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
from typing import Iterable, List, NamedTuple, Sequence, Tuple
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import serial

# --- Time horizon for distance → speed (plan: always 1 s) ---
DT_S = 1.0
# Decimal fraction of axial max body speed (0..1); CLI --speed-fraction overrides.
SPEED_LIMIT_FRACTION = 1.0

# --- Geometry placeholders (ω = 0: L / half-axle does not affect v_i) ---
ALPHA_RAD = 0.785  # first wheel angle vs robot +x (radians)
WHEEL_RADIUS_M = 0.0485  # placeholder; v = r·ω if you relate to motor shaft later
HALF_AXLE_LENGTH_M = 0.478  # placeholder for future yaw term
# max speed is 350 rpm
#min speed is theoretically 15 rpm, at some random pwm value we dont know yet.
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
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sys
import math
import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from typing import Iterable

class TrackerSubscriber(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('tracker_subscriber')
        self.args = args
        self.ser = None

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
            'result_topic',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg: Float32MultiArray):
        # We replace the 'sys.stdin' / 'argv' logic here.
        # Assuming msg.data contains [val1, val2]
        if len(msg.data) < 2:
            return

        pwms = [0, 0, 0, 0]
        
        # --- ALL YOUR ORIGINAL CONDITIONALS INTEGRATED ---
        
        # Scenario A: Interpreting data as [distance, heading] (replaces --stdin and --distance)
        if self.args.stdin or self.args.distance is not None:
            dist_m = msg.data[0]
            # Handle heading vs heading_deg logic
            if self.args.heading_deg is not None:
                heading_rad = math.radians(self.args.heading_deg)
            else:
                heading_rad = msg.data[1] # Assume second float is radians

            if abs(dist_m) > self.lim.max_distance_allowed_m:
                self.get_logger().warn(f"Rejected |dist|={abs(dist_m):g} (max {self.lim.max_distance_allowed_m:g})")
                return

            pwms = pwm_from_distance_heading(
                dist_m,
                heading_rad,
                alpha_rad=self.args.alpha,
                pwm_ref_m_s=self.args.pwm_ref,
            )

        # Scenario B: Interpreting data as [vx, vy]
        elif self.args.vx is not None or (len(msg.data) == 2 and self.args.distance is None):
            vx = msg.data[0]
            vy = msg.data[1]
            pwms = pwm_from_body_velocity(
                float(vx),
                float(vy),
                alpha_rad=self.args.alpha,
                pwm_ref_m_s=self.args.pwm_ref,
            )

        else:
            self.get_logger().error("Node configuration does not match incoming data format")
            return

        # 3. Execution (replaces _send_pwms)
        out = format_line(pwms)
        if self.args.dry_run:
            print(f"DRY RUN: {out.strip()}")
        elif self.ser is not None:
            _send_drive_line_delayed_stop(self.ser, out, pwms, self.args.stop_after)
            if self.args.stop_after > 0 and any(pwms):
                 self.get_logger().info(f"Sent command; stop scheduled in {self.args.stop_after}s")

    def stop_motors_and_close(self):
        if self.ser is not None:
            send_motor_stop(self.ser)
            self.ser.close()
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


def _print_speed_limit_banner(
    m: SpeedLimitDerived,
    speed_fraction: float,
    dt_s: float,
    *,
    file=sys.stderr,
) -> None:
    pct = speed_fraction * 100.0
    print(
        f"  Speed limit: {pct:g}% of axial max body speed "
        f"(v_axial_max ≈ {m.v_axial_max_m_s:g} m/s, +x ref)",
        file=file,
    )
    print(
        f"  Allowed body speed cap ≈ {m.v_body_cap_m_s:g} m/s; "
        f"peak wheel tangential at axial cap ≈ {m.v_wheel_peak_m_s:g} m/s",
        file=file,
    )
    print(
        f"  Max |distance_m| over {dt_s:g} s: {m.max_distance_allowed_m:g} m "
        "(larger inputs are rejected, no motion sent)",
        file=file,
    )


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


def _clamp_pwm(v: int) -> int:
    return max(PWM_MIN, min(PWM_MAX, v))


def velocities_to_pwm(vs: Sequence[float], max_speed_ref_m_s: float = PWM_REF_WHEEL_M_S) -> List[int]:
    if max_speed_ref_m_s <= 0:
        return [0, 0, 0, 0]
    out: List[int] = []
    for v in vs:
        out.append(_clamp_pwm(round(v / max_speed_ref_m_s * PWM_MAX)))
    return out


def apply_motor_perm(vs: Sequence[float], perm: Sequence[int] = MOTOR_PERM) -> List[float]:
    return [vs[perm[k]] for k in range(4)]


def format_line(pwms: Sequence[int]) -> str:
    a, b, c, d = (int(x) for x in pwms)
    return f"{a},{b},{c},{d}\n"


def open_serial(port: str, baud: int = 115200, timeout: float = 1.0) -> serial.Serial:
    return serial.Serial(port, baud, timeout=timeout)


def send_line(ser: serial.Serial, line: str) -> None:
    ser.write(line.encode("utf-8"))


def send_motor_stop(ser: serial.Serial) -> None:
    """Tell the Arduino to zero all channels (firmware holds last command until a new line)."""
    try:
        send_line(ser, format_line([0, 0, 0, 0]))
        ser.flush()
    except (OSError, serial.SerialException):
        pass


def _sleep_then_motor_stop(ser: serial.Serial, seconds: float) -> None:
    """Wait, then send stop. If interrupted during sleep, stop immediately and re-raise."""
    if seconds <= 0:
        return
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        send_motor_stop(ser)
        raise
    send_motor_stop(ser)


def _send_drive_line_delayed_stop(
    ser: serial.Serial,
    line: str,
    pwms: Sequence[int],
    stop_after_s: float,
) -> None:
    send_line(ser, line)
    ser.flush()
    if stop_after_s > 0 and any(pwms):
        _sleep_then_motor_stop(ser, stop_after_s)


def pwm_from_body_velocity(
    x_dot_m_s: float,
    y_dot_m_s: float,
    *,
    alpha_rad: float = ALPHA_RAD,
    pwm_ref_m_s: float = PWM_REF_WHEEL_M_S,
    perm: Sequence[int] = MOTOR_PERM,
) -> List[int]:
    vs = wheel_linear_velocities(x_dot_m_s, y_dot_m_s, alpha_rad)
    vs_ord = apply_motor_perm(vs, perm)
    return velocities_to_pwm(vs_ord, pwm_ref_m_s)


def pwm_from_distance_heading(
    distance_m: float,
    heading_rad: float,
    *,
    dt_s: float = DT_S,
    alpha_rad: float = ALPHA_RAD,
    pwm_ref_m_s: float = PWM_REF_WHEEL_M_S,
    perm: Sequence[int] = MOTOR_PERM,
) -> List[int]:
    v = speed_from_distance(distance_m, dt_s)
    x_dot, y_dot = body_velocity_from_speed_heading(v, heading_rad)
    return pwm_from_body_velocity(x_dot, y_dot, alpha_rad=alpha_rad, pwm_ref_m_s=pwm_ref_m_s, perm=perm)


def _parse_speed_fraction(s: str) -> float:
    x = float(s)
    if math.isnan(x) or x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("expect a decimal in [0, 1], e.g. 0.8 for 80%")
    return x


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Omni wheel PWM over serial (4 values, -255..255).")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial device")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--dry-run", action="store_true", help="Print line only, do not open serial")
    p.add_argument("--pwm-ref", type=float, default=PWM_REF_WHEEL_M_S, help="Wheel m/s for |PWM|=255")
    p.add_argument("--alpha", type=float, default=ALPHA_RAD, help="First wheel angle vs +x (radians)")
    p.add_argument(
        "--speed-fraction",
        type=_parse_speed_fraction,
        default=SPEED_LIMIT_FRACTION,
        metavar="F",
        help="Fraction of axial max body speed (0..1); |distance_m| above cap is rejected (no send)",
    )
    p.add_argument(
        "--stdin",
        action="store_true",
        help="Read lines: distance_m heading_rad (two floats per line) and send each",
    )
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt for distance_m and heading_rad (use if running without a TTY, e.g. some IDEs)",
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--distance", type=float, default=None, help="Distance (m) over DT_S → speed")
    g.add_argument("--vx", type=float, default=None, help="Body ẋ (m/s); use with --vy")

    p.add_argument("--heading", type=float, default=None, help="Heading (radians), with --distance")
    p.add_argument("--heading-deg", type=float, default=None, help="Heading (degrees), with --distance")
    p.add_argument("--vy", type=float, default=None, help="Body ý (m/s); use with --vx")
    p.add_argument(
        "--stop-after",
        type=float,
        default=STOP_AFTER_COMMAND_S,
        metavar="SEC",
        help=(
            "After each non-zero drive command, wait SEC seconds then send 0,0,0,0; "
            "0 disables (default: %(default)s)"
        ),
    )

    return p.parse_args(list(argv) if argv is not None else None)


def _send_pwms(args: argparse.Namespace, pwms: List[int]) -> None:
    line = format_line(pwms)
    if args.dry_run:
        print(line, end="")
        return
    try:
        ser = open_serial(args.port, args.baud)
    except (OSError, serial.SerialException) as e:
        print(f"Serial open failed ({args.port}): {e}", file=sys.stderr)
        raise SystemExit(1) from e
    try:
        _send_drive_line_delayed_stop(ser, line, pwms, args.stop_after)
    finally:
        ser.close()
    print(f'Sent to Arduino: "{line.strip()}"')
    if args.stop_after > 0 and any(pwms):
        print(f'Sent stop after {args.stop_after:g}s: "0,0,0,0"', file=sys.stderr)


def _interactive_loop(args: argparse.Namespace) -> int:
    """Prompt for distance_m and heading_rad until EOF or quit; reuse one serial port."""
    lim = derive_speed_limit_metrics(
        speed_fraction=args.speed_fraction,
        pwm_ref_m_s=args.pwm_ref,
        alpha_rad=args.alpha,
        dt_s=DT_S,
    )
    print(
        "Interactive mode: enter two numbers per line: distance_m heading_rad",
        file=sys.stderr,
    )
    print(
        "  (speed = distance_m / 1 s; heading_rad: 0 = +x forward, +π/2 = +y)",
        file=sys.stderr,
    )
    _print_speed_limit_banner(lim, args.speed_fraction, DT_S)
    sa = args.stop_after
    if sa > 0:
        print(
            f"  After each non-zero command: wait {sa:g}s, then send 0,0,0,0 (--stop-after 0 to disable).",
            file=sys.stderr,
        )
    print("  Empty line or 'q' to exit (motors zeroed on exit). Ctrl+C also stops.", file=sys.stderr)
    ser = None
    if not args.dry_run:
        try:
            ser = open_serial(args.port, args.baud)
        except (OSError, serial.SerialException) as e:
            print(f"Serial open failed ({args.port}): {e}", file=sys.stderr)
            return 1
    try:
        while True:
            try:
                raw = input("distance_m heading_rad> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nStopping motors…", file=sys.stderr)
                break
            if not raw or raw.lower() in ("q", "quit", "exit"):
                break
            parts = raw.split()
            if len(parts) < 2:
                print("Need two numbers: distance_m heading_rad", file=sys.stderr)
                continue
            try:
                dist_m = float(parts[0])
                heading_rad = float(parts[1])
            except ValueError:
                print("Invalid number(s)", file=sys.stderr)
                continue
            if abs(dist_m) > lim.max_distance_allowed_m:
                print(
                    f"Rejected: |distance_m|={abs(dist_m):g} exceeds max "
                    f"{lim.max_distance_allowed_m:g} m (no command sent).",
                    file=sys.stderr,
                )
                continue
            pwms = pwm_from_distance_heading(
                dist_m,
                heading_rad,
                alpha_rad=args.alpha,
                pwm_ref_m_s=args.pwm_ref,
            )
            out = format_line(pwms)
            if args.dry_run:
                print(out, end="")
            elif ser is not None:
                try:
                    _send_drive_line_delayed_stop(ser, out, pwms, args.stop_after)
                except KeyboardInterrupt:
                    print("\nStopping motors…", file=sys.stderr)
                    break
                print(f'Sent to Arduino: "{out.strip()}"')
                if args.stop_after > 0 and any(pwms):
                    print(f'Sent stop after {args.stop_after:g}s: "0,0,0,0"', file=sys.stderr)
    finally:
        if ser is not None:
            send_motor_stop(ser)
            ser.close()
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    # Use your existing _parse_args
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    
    try:
        node = TrackerSubscriber(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node initialization failed: {e}")
        return 1
    finally:
        if 'node' in locals():
            node.stop_motors_and_close()
            node.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
