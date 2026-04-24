"""Omni-wheel body velocity → per-wheel PWM and serial line for Arduino four-motor firmware.

Wheel kinematics (ω = 0): v_i = -sin(α_i)·ẋ + cos(α_i)·ý with
α_i ∈ {α, α+π/2, α-π, α-π/2} for wheels 1..4 in matrix order.

Motion heading φ (body frame): 0 rad = +x forward; +π/2 = +y (left in typical robot coords).
α (ALPHA_RAD) is wheel mounting geometry, not the motion heading.

Serial round-trip: RTT is measured from immediately before the PWM ``write`` until the first
full line is read. Firmware that echoes an acknowledgement (e.g. a line starting with
``OK`` on four-motor boards) allows meaningful RTT. Use ``--ack-prefix none`` if the device
sends a different line. ``--rtt-stdin-commands`` enables h/s/c/v in a TTY for live stats.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

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

SERIAL_READ_DEADLINE_S = 0.08

ACK_OK = "ok"
ACK_ERR_LINE = "err"
ACK_NO_MATCH = "no_match"
ACK_TIMEOUT = "timeout"


@dataclass
class RttStats:
    count_ok: int = 0
    err_lines: int = 0
    timeouts: int = 0
    no_match: int = 0
    rtt_min_ms: float = float("inf")
    rtt_max_ms: float = 0.0
    rtt_sum_ms: float = 0.0
    last_line: str = ""
    last_rtt_ms: float = 0.0
    last_status: str = ""

    def record(self, rtt_ms: float, status: str, line: str) -> None:
        self.last_rtt_ms = rtt_ms
        self.last_status = status
        self.last_line = line[:200]
        if status == ACK_OK:
            self.count_ok += 1
            self.rtt_min_ms = min(self.rtt_min_ms, rtt_ms)
            self.rtt_max_ms = max(self.rtt_max_ms, rtt_ms)
            self.rtt_sum_ms += rtt_ms
        elif status == ACK_ERR_LINE:
            self.err_lines += 1
        elif status == ACK_TIMEOUT:
            self.timeouts += 1
        elif status == ACK_NO_MATCH:
            self.no_match += 1

    def mean_ms(self) -> float:
        if self.count_ok == 0:
            return 0.0
        return self.rtt_sum_ms / self.count_ok


class SpeedLimitDerived(NamedTuple):
    v_axial_max_m_s: float
    v_body_cap_m_s: float
    v_wheel_peak_m_s: float
    max_distance_allowed_m: float


def _classify_ack(line: str, prefix: str) -> str:
    s = line.strip()
    if s.upper().startswith("ERR"):
        return ACK_ERR_LINE
    if not prefix or prefix == "none":
        return ACK_OK
    if s.startswith(prefix):
        return ACK_OK
    return ACK_NO_MATCH


def _first_text_line(data: str) -> str:
    for seg in data.replace("\r", "\n").split("\n"):
        t = seg.strip()
        if t:
            return t
    return data.strip()


def _rtt_stdin_thread(node: "TrackerSubscriber") -> None:
    print(
        "RTT debug keys: h=help  s=stats  c=clear  v=verbose (COMMDBG-style traces)",
        file=sys.stderr,
    )
    while rclpy.ok() and not node._rtt_stop.is_set():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        line = sys.stdin.readline()
        if not line:
            break
        c = line.strip().lower()[:1]
        if c == "h" or c == "?":
            print(
                "  h / ?  This help\n"
                "  s     Print serial RTT stats (ok count, min/max/avg ms, timeouts)\n"
                "  c     Clear RTT stat counters (not seq)\n"
                "  v     Toggle extra COMMDBG phases without restarting\n",
                file=sys.stderr,
            )
        elif c == "s":
            st = node.rtt_stats
            with node._rtt_lock:
                n_ok = st.count_ok
                mean = st.mean_ms()
                tmo = st.timeouts
                nm = st.no_match
                nerr = st.err_lines
                rmin = st.rtt_min_ms if n_ok else 0.0
                rmax = st.rtt_max_ms if n_ok else 0.0
                last_ms = st.last_rtt_ms
                last_s = st.last_status
            print(
                f"[RTT_STATS] ok={n_ok} min_ms={rmin:.3f} max_ms={rmax:.3f} "
                f"mean_ms={mean:.3f} last_ms={last_ms:.3f} last_st={last_s!r} "
                f"err={nerr} timeouts={tmo} no_match={nm}",
                file=sys.stderr,
            )
        elif c == "c":
            with node._rtt_lock:
                node.rtt_stats = RttStats()
            print("[RTT_STATS] counters cleared", file=sys.stderr)
        elif c == "v":
            node._verbose_traces = not node._verbose_traces
            print(
                f"[RTT] verbose_traces={node._verbose_traces!r} "
                f"(enables [COMMDBG_PY] phases when not using --comm-timing only)",
                file=sys.stderr,
            )
        else:
            print("Unknown: use h, s, c, or v", file=sys.stderr)


class TrackerSubscriber(Node):
    def __init__(self, args: argparse.Namespace, rtt_stdin: bool) -> None:
        super().__init__("tracker_subscriber")
        self.args = args
        self.ser = None
        self._comm_seq = 0
        self._rtt_io_seq = 0
        self._last_comm_mono: float | None = None

        self._verbose_traces: bool = False
        self.rtt_stats = RttStats()
        self._rtt_lock = threading.Lock()
        self._rtt_stop = threading.Event()
        self._rtt_thread: Optional[threading.Thread] = None
        if rtt_stdin:
            t = threading.Thread(
                target=_rtt_stdin_thread,
                args=(self,),
                name="rtt-stdin",
                daemon=True,
            )
            t.start()
            self._rtt_thread = t

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
            "/bean_bag_trajectory",
            self.listener_callback,
            10,
            callback_group=self.callback_group,
        )

    @property
    def _traces(self) -> bool:
        return bool(self.args.comm_timing) or self._verbose_traces

    def _comm_print(self, phase: str, **kwargs: object) -> None:
        if not self._traces:
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

    def _ack_prefix_value(self) -> str:
        p = (self.args.ack_prefix or "none").strip()
        if p == "none":
            return "none"
        return p

    def _write_pwm_and_read_ack(self, cmd: str) -> None:
        """t0 = immediately before write; RTT to first line from device (or timeout)."""
        need_read = self.args.rtt or self._traces
        if self.args.dry_run:
            if self._traces or self.args.rtt:
                print(f"DRY RUN: {cmd.strip()}", file=sys.stderr, flush=True)
            return
        if not self.ser or not self.ser.is_open:
            if need_read:
                with self._rtt_lock:
                    self.rtt_stats.last_status = "closed"
            return

        self._rtt_io_seq += 1
        raw = cmd.encode("utf-8")
        t0 = time.monotonic()
        self._comm_print("serial_write", nbytes=len(raw))
        self.ser.write(raw)
        self.ser.flush()
        self._comm_print("serial_flush_done")

        if not need_read:
            return

        buf = bytearray()
        deadline = time.monotonic() + SERIAL_READ_DEADLINE_S
        while time.monotonic() < deadline:
            n = self.ser.in_waiting
            if n:
                buf.extend(self.ser.read(n))
                if b"\n" in buf or b"\r" in buf:
                    break
            time.sleep(0.001)
        if b"\n" not in buf and b"\r" not in buf:
            buf.extend(self.ser.readline())

        t1 = time.monotonic()
        rtt_ms = (t1 - t0) * 1000.0
        text_raw = bytes(buf).decode("utf-8", errors="replace")
        first = _first_text_line(text_raw)

        if not first:
            status = ACK_TIMEOUT
            line = ""
        else:
            pfx = self._ack_prefix_value()
            c = _classify_ack(first, pfx)
            if c == ACK_ERR_LINE:
                status = ACK_ERR_LINE
            elif c == ACK_NO_MATCH:
                status = ACK_NO_MATCH
            else:
                status = ACK_OK
            line = first

        self._comm_print("serial_read", rtt_ms=f"{rtt_ms:.3f}", resp=repr(text_raw[:200].strip()))
        with self._rtt_lock:
            self.rtt_stats.record(rtt_ms, status, line)
        rtt_s = f"{rtt_ms:.3f}"
        if self.args.rtt:
            if status == ACK_OK:
                print(
                    f"[RTT] seq={self._rtt_io_seq} rtt_ms={rtt_s} status=ok "
                    f"ack={line[:200]!r}",
                    file=sys.stderr,
                    flush=True,
                )
            elif status == ACK_ERR_LINE:
                print(
                    f"[RTT] seq={self._rtt_io_seq} rtt_ms={rtt_s} status=err line={line[:200]!r}",
                    file=sys.stderr,
                    flush=True,
                )
            elif status == ACK_NO_MATCH:
                print(
                    f"[RTT] seq={self._rtt_io_seq} rtt_ms={rtt_s} status=no_match first={line[:200]!r}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[RTT] seq={self._rtt_io_seq} rtt_ms={rtt_s} status=timeout",
                    file=sys.stderr,
                    flush=True,
                )

    def destroy_node(self) -> None:  # type: ignore[override]
        self._rtt_stop.set()
        if self._rtt_thread and self._rtt_thread.is_alive():
            self._rtt_thread.join(timeout=0.2)
        super().destroy_node()

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

        out = format_line(pwms)
        self.get_logger().debug(f"Sending PWM: {pwms}")
        t_drive = time.monotonic()
        self._write_pwm_and_read_ack(out)
        self._comm_print("after_send_and_ack", block_ms=f"{(time.monotonic() - t_drive) * 1000.0:.3f}")

    def stop_motors_and_close(self) -> None:
        if self.ser is not None:
            try:
                if self.ser.is_open:
                    send_motor_stop(self.ser)
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass
            self.get_logger().info("Serial port closed.")


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


def send_motor_stop(ser: serial.Serial) -> None:
    """Sends zero velocity to all motors."""
    ser.write(b"0,0,0,0\n")


def pwm_from_body_velocity(vx: float, vy: float, alpha_rad: float, pwm_ref_m_s: float) -> List[int]:
    """Converts x/y velocity to discrete PWM values [-255, 255]."""
    v_wheels = wheel_linear_velocities(vx, vy, alpha_rad)
    return [int((v / pwm_ref_m_s) * PWM_MAX) for v in v_wheels]


def pwm_from_distance_heading(
    dist_m: float, heading_rad: float, alpha_rad: float, pwm_ref_m_s: float
) -> List[int]:
    """Converts distance/heading to PWM via a 1s time horizon."""
    speed = speed_from_distance(dist_m, DT_S)
    vx, vy = body_velocity_from_speed_heading(speed, heading_rad)
    return pwm_from_body_velocity(vx, vy, alpha_rad, pwm_ref_m_s)


def main(args: Optional[object] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Omni-wheel distance/heading to PWM, serial to Arduino. "
            "Use --rtt to print command→ACK line RTT. Firmware should echo a line (e.g. starting with OK)."
        )
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of send")
    parser.add_argument("--speed-fraction", type=float, default=SPEED_LIMIT_FRACTION)
    parser.add_argument("--alpha", type=float, default=ALPHA_RAD)
    parser.add_argument("--pwm-ref", type=float, default=PWM_REF_WHEEL_M_S)
    parser.add_argument(
        "--heading-deg",
        type=float,
        default=None,
        help="If set, use this heading (degrees) instead of msg.data[1] (radians)",
    )
    parser.add_argument(
        "--comm-timing",
        action="store_true",
        help="Print [COMMDBG_PY] phase lines to stderr (monotonic seq per node).",
    )
    parser.add_argument(
        "--rtt",
        action="store_true",
        help="After each write, read one response line; print [RTT] with round-trip ms (t0=before write).",
    )
    parser.add_argument(
        "--rtt-stdin-commands",
        action="store_true",
        help="In a TTY, enable h/s/c/v on stderr for RTT help, stats, clear, verbose traces.",
    )
    parser.add_argument(
        "--ack-prefix",
        type=str,
        default="OK",
        help=(
            "Classify a good ACK: line must start with this; use 'none' to accept the first line as OK. "
            "Lines starting with ERR are always classified as err."
        ),
    )

    parsed_args, _unknown = parser.parse_known_args()

    rtt_stdin = bool(parsed_args.rtt_stdin_commands) and sys.stdin.isatty()
    if parsed_args.rtt_stdin_commands and not rtt_stdin:
        print(
            "Warning: --rtt-stdin-commands requires an interactive TTY; stdin thread not started",
            file=sys.stderr,
        )

    rclpy.init(args=args)
    node = TrackerSubscriber(parsed_args, rtt_stdin)
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
