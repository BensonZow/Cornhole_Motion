#!/usr/bin/env python3
import math
import serial
import time
import sys

# ========== CONFIGURATION ==========
SERIAL_PORT = '/dev/ttyUSB0'   # Adjust to your serial port (e.g., '/dev/ttyAMA0')
BAUD_RATE = 115200
MAX_PWM = 100                  # PWM range: -MAX_PWM .. +MAX_PWM

# Wheel order: front-left, front-right, rear-left, rear-right
# Kinematics for pure translation (no rotation) using 4 omni wheels at 45° corners:
#   w_FL = Vx + Vy
#   w_FR = Vx - Vy
#   w_RL = Vx - Vy
#   w_RR = Vx + Vy
# where (Vx, Vy) is the desired unit direction vector.
# ===================================

def angle_to_pwm(angle_deg):
    """
    Convert an angle (0‑360°, 0° = forward, 90° = left) into four PWM values.
    Returns a tuple of four integers (FL, FR, RL, RR) in the range [-MAX_PWM, MAX_PWM].
    """
    theta = math.radians(angle_deg)
    vx = math.cos(theta)
    vy = math.sin(theta)

    # Raw speeds (range about -1.414 .. +1.414)
    w_fl_raw = vx + vy
    w_fr_raw = vx - vy
    w_rl_raw = vx - vy
    w_rr_raw = vx + vy

    # Normalise so that the maximum absolute value equals MAX_PWM
    max_raw = max(abs(w_fl_raw), abs(w_fr_raw), abs(w_rl_raw), abs(w_rr_raw))
    if max_raw > 1e-6:          # avoid division by zero
        scale = MAX_PWM / max_raw
    else:
        scale = 0.0

    # Scale, round, and clamp (clamp for safety)
    pwm_fl = int(round(w_fl_raw * scale))
    pwm_fr = int(round(w_fr_raw * scale))
    pwm_rl = int(round(w_rl_raw * scale))
    pwm_rr = int(round(w_rr_raw * scale))

    # Clamp to [-MAX_PWM, MAX_PWM] (should already be within limits)
    pwm_fl = max(-MAX_PWM, min(MAX_PWM, pwm_fl))
    pwm_fr = max(-MAX_PWM, min(MAX_PWM, pwm_fr))
    pwm_rl = max(-MAX_PWM, min(MAX_PWM, pwm_rl))
    pwm_rr = max(-MAX_PWM, min(MAX_PWM, pwm_rr))

    return (pwm_fl, pwm_fr, pwm_rl, pwm_rr)

def send_pwm(ser, pwm_values):
    """Send four comma‑separated PWM values over serial, followed by newline."""
    cmd = f"{pwm_values[0]},{pwm_values[1]},{pwm_values[2]},{pwm_values[3]}\n"
    ser.write(cmd.encode())

def main():
    # Open serial port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        sys.exit(1)

    print("Omni‑wheel robot controller ready.")
    print("Enter an angle (0‑360°) – 0° = forward, 90° = left, 180° = backward, 270° = right.")
    print("The robot will move for 0.5 seconds and then stop.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            # Get angle from user
            user_input = input("Angle (0‑360): ").strip()
            if not user_input:
                continue
            try:
                angle = float(user_input)
                if angle < 0 or angle > 360:
                    print("Angle must be between 0 and 360.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            # Compute PWM values
            pwm = angle_to_pwm(angle)

            # Send motion command
            send_pwm(ser, pwm)
            print(f"Moving at {angle}° -> PWM: {pwm}")

            # Move for 0.5 seconds
            time.sleep(0.5)

            # Send stop command
            send_pwm(ser, (0, 0, 0, 0))
            print("Stop.")

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        # Ensure robot is stopped before closing
        send_pwm(ser, (0, 0, 0, 0))
        ser.close()

if __name__ == "__main__":
    main()