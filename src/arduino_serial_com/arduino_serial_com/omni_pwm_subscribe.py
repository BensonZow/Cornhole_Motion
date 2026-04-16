"""Omni-wheel body velocity → per-wheel PWM and serial line for Arduino four-motor firmware.

Wheel kinematics (ω = 0): v_i = -sin(α_i)·ẋ + cos(α_i)·ý with
α_i ∈ {α, α+π/2, α-π, α-π/2} for wheels 1..4 in matrix order.

Motion heading φ (body frame): 0 rad = +x forward; +π/2 = +y (left in typical robot coords).
α (ALPHA_RAD) is wheel mounting geometry, not the motion heading.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, NamedTuple, Sequence, Tuple

import rclpy
import serial
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# --- Time horizon for distance → speed (plan: always 1 s) ---
DT_S = 1.0
# Decimal fraction of axial max body speed (0..1); CLI --speed-fraction overrides.
SPEED_LIMIT_FRACTION = 1.0

# --- Geometry placeholders (ω = 0: L / half-axle does not affect v_i) ---
ALPHA_RAD = 0.785  # first wheel angle vs robot +x (radians)
# max speed is 350 rpm
# min speed is theoretically 15 rpm, at some random pwm value we dont know yet.
# Wheel tangential speed (m/s) that maps to |PWM| = 255
PWM_REF_WHEEL_M_S = 1.778

PWM_MAX = 255
PWM_MIN = -255

# Arduino motor order FR,FL,RR,RL: motor_k uses matrix output vs[MOTOR_PERM[k]]
MOTOR_PERM: Tuple[int, int, int, int] = (0, 1, 2, 3)

# Offsets for the four rows (radians) relative to ALPHA_RAD
_WHEEL_ANGLE_OFFSETS = (0.0, math.pi / 2, -math.pi, -math.pi / 2)

# After each non-zero drive command, wait then send 0,0,0,0 (CLI default; override with --stop-after)
STOP_AFTER_COMMAND_S = 1.0


class SpeedLimitDerived(NamedTuple):
    v_axial_max_m_s: float
    v_body_cap_m_s: float
    v_wheel_peak_m_s: float
    max_distance_allowed_m: float


class TrackerSubscriber(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('tracker_subscriber')
        self.args = args
        self.ser = None
        self.stop_timer = None
        self._comm_seq = 0
        self._last_comm_mono: float | None = None

        # Use a Reentrant group so the timer and subscriber can run simultaneously
        self.callback_group = ReentrantCallbackGroup()

        # 1. Setup Serial once during initialization
        if not self.args.dry_run:
            try:
                self.ser = open_serial(self.args.port, self.args.baud)
                self.get_logger().info(f"Serial opened on {self.args.port}")
            except (OSError, serial.SerialException) as e:
                self.get_logger().error(f"Serial open failed: {e}")
                raise e

        # 2. Pre-calculate limits if using distance mode
        self.lim = derive_speed_limit_metrics(
            speed_fraction=self.args.speed_fraction,
            pwm_ref_m_s=self.args.pwm_ref,
            alpha_rad=self.args.alpha,
            dt_s=DT_S,
        )

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/bean_bag_trajectory',
            self.listener_callback,
            10,
            callback_group=self.callback_group,
        )

    def _comm_print(self, phase: str, **kwargs: object) -> None:
        if not self.args.comm_timing:
            return
        self._comm_seq += 1
        now = time.monotonic()
        delta_ms = 0.0
        if self._last_comm_mono is not None:
            delta_ms = (now - self._last_comm_mono) * 1000.0
        self._last_comm_mono = now
        parts = [
            f"[COMMDBG_PY] seq={self._comm_seq} phase={phase}",
            f"mono_s={now:.6f}",
            f"delta_since_prev_ms={delta_ms:.3f}",
        ]
        for k, v in kwargs.items():
            parts.append(f"{k}={v}")
        print(" ".join(parts), file=sys.stderr, flush=True)

    def _try_read_serial_response(self) -> None:
        if not self.args.comm_timing or self.ser is None or not self.ser.is_open:
            return
        t0 = time.monotonic()
        deadline = t0 + 0.08
        buf = bytearray()
        while time.monotonic() < deadline:
            n = self.ser.in_waiting
            if n:
                buf.extend(self.ser.read(n))
                if b"\n" in buf or b"\r" in buf:
                    break
            time.sleep(0.001)
        if b"\n" not in buf and b"\r" not in buf:
            buf.extend(self.ser.readline())
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        text = bytes(buf).decode("utf-8", errors="replace").strip()
        self._comm_print(
            "serial_read",
            ack_wait_ms=f"{elapsed_ms:.3f}",
            resp=repr(text[:200]),
        )

    def listener_callback(self, msg: Float32MultiArray):
        """Always interpret ``msg.data`` as ``[distance_m, heading_rad]`` (e.g. from bean_bag_trajectory)."""
        if len(msg.data) < 2:
            return

        self._comm_print("callback_enter", msg_len=len(msg.data))

        dist_m = float(msg.data[0])
        if self.args.heading_deg is not None:
            heading_rad = math.radians(self.args.heading_deg)
        else:
            heading_rad = float(msg.data[1])

        if abs(dist_m) > self.lim.max_distance_allowed_m:
            self.get_logger().warn(
                f"Rejected |dist|={abs(dist_m):g} (max {self.lim.max_distance_allowed_m:g})"
            )
            self._comm_print("reject_distance", dist_m=dist_m)
            return

        pwms = pwm_from_distance_heading(
            dist_m,
            heading_rad,
            alpha_rad=self.args.alpha,
            pwm_ref_m_s=self.args.pwm_ref,
        )

        self._comm_print("after_compute", pwms=str(pwms))

        # 3. Execution (replaces _send_pwms)
        out = format_line(pwms)
        if self.args.dry_run:
            print(f"DRY RUN: {out.strip()}")
        elif self.ser is not None:
            self._comm_print("before_serial_drive", nbytes=len(out.encode("utf-8")))
            t_drive = time.monotonic()
            _send_drive_line_delayed_stop(self.ser, out, pwms, self.args.stop_after)
            self._comm_print(
                "after_serial_drive",
                drive_block_ms=f"{(time.monotonic() - t_drive) * 1000.0:.3f}",
            )
            if self.args.stop_after > 0 and any(pwms):
                self.get_logger().info(f"Sent command; stop scheduled in {self.args.stop_after}s")

        # LOGGING: See if commands are actually being sent
        self.get_logger().debug(f"Sending PWM: {pwms}")
        self._comm_print("before_send_to_serial")
        self.send_to_serial(format_line(pwms))
        self._try_read_serial_response()

        # Reset the stop timer
        if self.stop_timer:
            self.stop_timer.cancel()

        # Create timer in the reentrant group
        self.stop_timer = self.create_timer(
            self.args.stop_after,
            self.timer_stop_callback,
            callback_group=self.callback_group,
        )

    def stop_motors_and_close(self):
        if self.ser is not None:
            send_motor_stop(self.ser)
            self.ser.close()
            self.get_logger().info("Serial port closed.")

    def timer_stop_callback(self):
        self.get_logger().info("!!! TIMER STOP TRIGGERED !!!")
        self._comm_print("timer_stop_callback")
        self.send_to_serial("0,0,0,0\n")
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

    def send_to_serial(self, cmd: str):
        if self.args.dry_run:
            print(f"DRY RUN: {cmd.strip()}")
        elif self.ser and self.ser.is_open:
            raw = cmd.encode("utf-8")
            self._comm_print("serial_write", nbytes=len(raw))
            self.ser.write(raw)
            self.ser.flush()  # Ensure it actually leaves the OS buffer
            self._comm_print("serial_flush_done")


def max_wheel_gain_for_heading(heading_rad: float, alpha_rad: float = ALPHA_RAD) -> float:
    """max_i |sin(heading - α_i)| for unit-speed body motion along heading_rad."""
    k = 0.0
    for off in _WHEEL_ANGLE_OFFSETS:
        a = alpha_rad + off
        k = max(k, abs(math.sin(heading_rad - a)))
    return k


def max_body_speed_m_s_for_heading(
    heading_rad: float,
    *,
    pwm_ref_m_s: float,
    alpha_rad: float = ALPHA_RAD,
) -> float:
    """Max body speed (m/s) before the fastest wheel hits pwm_ref tangential (|PWM|=255)."""
    k = max_wheel_gain_for_heading(heading_rad, alpha_rad)
    if k < 1e-12:
        return float("inf")
    return pwm_ref_m_s / k


def derive_speed_limit_metrics(
    *,
    speed_fraction: float,
    pwm_ref_m_s: float,
    alpha_rad: float,
    dt_s: float = DT_S,
) -> SpeedLimitDerived:
    """Axial (+x) reference: v_axial_max, then fraction-scaled caps and max |distance_m| over dt_s."""
    v_axial_max = max_body_speed_m_s_for_heading(0.0, pwm_ref_m_s=pwm_ref_m_s, alpha_rad=alpha_rad)
    v_body_cap = speed_fraction * v_axial_max
    v_wheel_peak = speed_fraction * pwm_ref_m_s
    max_dist = v_body_cap * dt_s
    return SpeedLimitDerived(v_axial_max, v_body_cap, v_wheel_peak, max_dist)


def speed_from_distance(distance_m: float, dt_s: float = DT_S) -> float:
    return distance_m / dt_s


def body_velocity_from_speed_heading(speed_m_s: float, heading_rad: float) -> Tuple[float, float]:
    """Body-frame velocities from speed and heading.

    heading_rad: 0 = +x forward; increasing toward +y (strafe).
    """
    return (speed_m_s * math.cos(heading_rad), speed_m_s * math.sin(heading_rad))


def wheel_linear_velocities(x_dot_m_s: float, y_dot_m_s: float, alpha_rad: float = ALPHA_RAD) -> List[float]:
    vs: List[float] = []
    for off in _WHEEL_ANGLE_OFFSETS:
        a = alpha_rad + off
        vs.append(-math.sin(a) * x_dot_m_s + math.cos(a) * y_dot_m_s)
    return vs


def format_line(pwms: Sequence[int]) -> str:
    """Formats PWM list into the string expected by the Arduino firmware."""
    ordered = [pwms[MOTOR_PERM[i]] for i in range(4)]
    return ",".join(map(str, ordered)) + "\n"


def open_serial(port: str, baud: int) -> serial.Serial:
    """Opens the serial port for Arduino communication."""
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2)  # Wait for Arduino reset
    return ser


def send_motor_stop(ser: serial.Serial):
    """Sends zero velocity to all motors."""
    ser.write(b"0,0,0,0\n")


def _send_drive_line_delayed_stop(ser: serial.Serial, line: str, pwms: List[int], delay: float):
    """Sends the command and handles the stop-after timer."""
    ser.write(line.encode("utf-8"))
    if delay > 0 and any(pwms):
        time.sleep(delay)
        send_motor_stop(ser)


def pwm_from_body_velocity(vx: float, vy: float, alpha_rad: float, pwm_ref_m_s: float) -> List[int]:
    """Converts x/y velocity to discrete PWM values [-255, 255]."""
    v_wheels = wheel_linear_velocities(vx, vy, alpha_rad)
    return [int((v / pwm_ref_m_s) * PWM_MAX) for v in v_wheels]


def pwm_from_distance_heading(dist_m: float, heading_rad: float, alpha_rad: float, pwm_ref_m_s: float) -> List[int]:
    """Converts distance/heading to PWM via a 1s time horizon."""
    speed = speed_from_distance(dist_m, DT_S)
    vx, vy = body_velocity_from_speed_heading(speed, heading_rad)
    return pwm_from_body_velocity(vx, vy, alpha_rad, pwm_ref_m_s)


def main(args=None):
    parser = argparse.ArgumentParser(description="Omni-wheel Serial Node")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of send")
    parser.add_argument("--speed-fraction", type=float, default=SPEED_LIMIT_FRACTION)
    parser.add_argument("--alpha", type=float, default=ALPHA_RAD)
    parser.add_argument("--pwm-ref", type=float, default=PWM_REF_WHEEL_M_S)
    parser.add_argument("--stop-after", type=float, default=STOP_AFTER_COMMAND_S)
    parser.add_argument(
        "--heading-deg",
        type=float,
        default=None,
        help="If set, use this heading (degrees) instead of msg.data[1] (radians)",
    )
    parser.add_argument(
        "--comm-timing",
        action="store_true",
        help="Print [COMMDBG_PY] seq=... timing lines to stderr (monotonic seq per node)",
    )

    parsed_args, _unknown = parser.parse_known_args()

    rclpy.init(args=args)
    node = TrackerSubscriber(parsed_args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.stop_motors_and_close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
