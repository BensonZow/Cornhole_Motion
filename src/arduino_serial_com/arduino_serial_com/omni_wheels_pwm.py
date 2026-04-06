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
from typing import Iterable, List, Sequence, Tuple

import serial

# --- Time horizon for distance → speed (plan: always 1 s) ---
DT_S = 1.0

# --- Geometry placeholders (ω = 0: L / half-axle does not affect v_i) ---
ALPHA_RAD = 0.0  # first wheel angle vs robot +x (radians)
WHEEL_RADIUS_M = 0.05  # placeholder; v = r·ω if you relate to motor shaft later
HALF_AXLE_LENGTH_M = 0.15  # placeholder for future yaw term

# Wheel tangential speed (m/s) that maps to |PWM| = 255
PWM_REF_WHEEL_M_S = 1.0

PWM_MAX = 255
PWM_MIN = -255

# Arduino motor order FR,FL,RR,RL: motor_k uses matrix output vs[MOTOR_PERM[k]]
MOTOR_PERM: Tuple[int, int, int, int] = (0, 1, 2, 3)

# Offsets for the four rows (radians) relative to ALPHA_RAD
_WHEEL_ANGLE_OFFSETS = (0.0, math.pi / 2, -math.pi, -math.pi / 2)


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


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Omni wheel PWM over serial (4 values, -255..255).")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial device")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--dry-run", action="store_true", help="Print line only, do not open serial")
    p.add_argument("--pwm-ref", type=float, default=PWM_REF_WHEEL_M_S, help="Wheel m/s for |PWM|=255")
    p.add_argument("--alpha", type=float, default=ALPHA_RAD, help="First wheel angle vs +x (radians)")
    p.add_argument(
        "--stdin",
        action="store_true",
        help="Read lines: distance_m heading_rad (two floats per line) and send each",
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--distance", type=float, default=None, help="Distance (m) over DT_S → speed")
    g.add_argument("--vx", type=float, default=None, help="Body ẋ (m/s); use with --vy")

    p.add_argument("--heading", type=float, default=None, help="Heading (radians), with --distance")
    p.add_argument("--heading-deg", type=float, default=None, help="Heading (degrees), with --distance")
    p.add_argument("--vy", type=float, default=None, help="Body ý (m/s); use with --vx")

    return p.parse_args(list(argv) if argv is not None else None)


def _send_pwms(args: argparse.Namespace, pwms: List[int]) -> None:
    line = format_line(pwms)
    if args.dry_run:
        print(line, end="")
        return
    ser = open_serial(args.port, args.baud)
    try:
        send_line(ser, line)
    finally:
        ser.close()
    print(f'Sent to Arduino: "{line.strip()}"')


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.stdin:
        ser = None if args.dry_run else open_serial(args.port, args.baud)
        try:
            for raw in sys.stdin:
                line_in = raw.strip()
                if not line_in or line_in.startswith("#"):
                    continue
                parts = line_in.split()
                if len(parts) < 2:
                    print("stdin: need distance_m heading_rad", file=sys.stderr)
                    return 2
                dist_m = float(parts[0])
                heading_rad = float(parts[1])
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
                    send_line(ser, out)
                    print(f'Sent to Arduino: "{out.strip()}"')
        finally:
            if ser is not None:
                ser.close()
        return 0

    if args.distance is not None:
        if args.heading is not None and args.heading_deg is not None:
            print("Use only one of --heading or --heading-deg", file=sys.stderr)
            return 2
        if args.heading is None and args.heading_deg is None:
            print("--distance requires --heading or --heading-deg", file=sys.stderr)
            return 2
        heading_rad = math.radians(args.heading_deg) if args.heading_deg is not None else float(args.heading)
        pwms = pwm_from_distance_heading(
            args.distance,
            heading_rad,
            alpha_rad=args.alpha,
            pwm_ref_m_s=args.pwm_ref,
        )
    elif args.vx is not None:
        if args.vy is None:
            print("--vx requires --vy", file=sys.stderr)
            return 2
        pwms = pwm_from_body_velocity(
            float(args.vx),
            float(args.vy),
            alpha_rad=args.alpha,
            pwm_ref_m_s=args.pwm_ref,
        )
    else:
        print("Provide --distance and heading, or --vx and --vy, or --stdin", file=sys.stderr)
        return 2

    _send_pwms(args, pwms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
