# Cornhole Motion — Mecanum Wheel Platform

Arduino Mega firmware for a 4-wheel mecanum robot using two HW-095 (L298N) motor drivers with encoder feedback.

## Hardware

| Component | Details |
|-----------|---------|
| Controller | Arduino Mega 2560 |
| Motor Drivers | 2x HW-095 (L298N dual H-bridge), 2A per channel |
| Wheels | 97mm mecanum, 9 rollers @ 45°, inverted-V pattern |
| Wheelbase | 133.35 mm (front-to-rear axle) |
| Track Width | 304.8 mm (left-to-right wheel) |

## Wiring

### Motor Driver Pins

Remove the ENA/ENB jumpers on each HW-095 to enable external PWM speed control.

| Motor | Driver | Enable (PWM) | IN1 / IN3 | IN2 / IN4 |
|-------|--------|-------------|-----------|-----------|
| Front Left | HW-095 #1 | Pin 6 | Pin 26 | Pin 27 |
| Front Right | HW-095 #1 | Pin 7 | Pin 28 | Pin 29 |
| Rear Left | HW-095 #2 | Pin 8 | Pin 30 | Pin 31 |
| Rear Right | HW-095 #2 | Pin 9 | Pin 32 | Pin 33 |

### Encoder Pins

Channel A uses hardware interrupt pins; Channel B is read in the ISR for direction.

| Encoder | Channel A (Interrupt) | Channel B |
|---------|----------------------|-----------|
| Front Left | Pin 2 (INT0) | Pin 22 |
| Front Right | Pin 3 (INT1) | Pin 23 |
| Rear Left | Pin 18 (INT5) | Pin 24 |
| Rear Right | Pin 19 (INT4) | Pin 25 |

### Power

- Connect motor supply voltage (5–12V) to the HW-095 `VS` / `+12V` terminal.
- Connect HW-095 `GND` to Arduino `GND` (common ground is required).
- If motor supply is under 12V, the onboard 5V regulator can power logic; otherwise supply 5V to the `VSS` / `+5V` terminal separately and remove the 5V jumper.

## Serial Commands (115200 baud)

Open the Arduino Serial Monitor. Set line ending to **Newline** or **Both NL & CR**.

### Movement Pulses

Type a direction and press Enter. The robot moves for 300 ms then stops.

| Command | Motion |
|---------|--------|
| `F` | Forward |
| `B` | Backward |
| `L` | Strafe left |
| `R` | Strafe right |
| `FL` | Diagonal front-left |
| `FR` | Diagonal front-right |
| `BL` | Diagonal back-left |
| `BR` | Diagonal back-right |
| `CW` | Rotate clockwise |
| `CCW` | Rotate counter-clockwise |

### Configuration

| Command | Description |
|---------|-------------|
| `p <0-255>` | Set pulse PWM power (default 150) |
| `t <ms>` | Set pulse duration in ms (default 300, range 50–5000) |

### Raw Motor Control

For direct wheel control and future ROS2 integration.

| Command | Description |
|---------|-------------|
| `m <FL> <FR> <RL> <RR>` | Set per-wheel PWM, -255 to 255 |
| `e` | Read encoder tick counts |
| `r` | Reset encoder counts to zero |
| `s` | Emergency stop |

Raw motor commands (`m`) auto-stop after 500 ms if no new command is received (watchdog).

## Kinematics

The robot uses an **inverted-V** roller pattern (viewed from above, front rollers angle outward: `/ \`).

Inverse kinematics (robot velocity → wheel speed):

```
FL = (1/r) * (Vx - Vy - K * ω)
FR = (1/r) * (Vx + Vy + K * ω)
RL = (1/r) * (Vx + Vy - K * ω)
RR = (1/r) * (Vx - Vy + K * ω)
```

Where `r = 48.5 mm`, `K = Lx + Ly = 219.075 mm`, `Vx` = forward, `Vy` = strafe left, `ω` = counter-clockwise.

For pulse commands these simplify to a sign table (+1 / -1 / 0) applied to a fixed PWM value:

| Direction | FL | FR | RL | RR |
|-----------|:--:|:--:|:--:|:--:|
| Forward | + | + | + | + |
| Backward | − | − | − | − |
| Strafe Left | − | + | + | − |
| Strafe Right | + | − | − | + |
| Diag FL | 0 | + | + | 0 |
| Diag FR | + | 0 | 0 | + |
| Diag BL | − | 0 | 0 | − |
| Diag BR | 0 | − | − | 0 |
| Rotate CW | + | − | + | − |
| Rotate CCW | − | + | − | + |

## Upload

1. Open `firmware/mecanum_firmware/mecanum_firmware.ino` in the Arduino IDE.
2. Select **Board → Arduino Mega or Mega 2560**.
3. Select the correct **Port**.
4. Click **Upload**.
5. Open **Serial Monitor** at 115200 baud, line ending set to Newline.
6. Type `F` and press Enter to test a forward pulse.
