# Omni BLDC (Arduino-Only) Integration

This document covers omni-wheel BLDC integration using:

- Arduino Mega 2560
- 4x BLD-515C drivers
- 4x brushless geared motors
- Host (e.g. ROS2) can send `distance` (inches) and `angle` (radians) over USB serial

## Firmware

- Sketch: [omni_bldc_firmware.ino](omni_bldc_firmware.ino)
- Serial baud: `115200`
- **SV**: PWM + RC low-pass → analog speed command  
- **EN**: digital output (sketch uses `HIGH` = enable when moving—match your wiring to the driver)  
- **F/R** and **BK**: Mega pins wired **directly** to the driver; see pin-mode behavior below.

## F/R and BK pin behavior (BLD-515C-style)

The driver text describes **GND** vs **not connected** for some lines. This sketch drives **F/R** and **BK** as **push-pull outputs** from the Mega:

| Meaning on driver | Mega configuration |
| ----------------- | ------------------ |
| Not grounding the line (forward / brake released while running) | `pinMode(pin, OUTPUT); digitalWrite(pin, HIGH)` |
| Line grounded (reverse / brake asserted) | `pinMode(pin, OUTPUT); digitalWrite(pin, LOW)` |

**Caveat:** **OUTPUT HIGH** sources about **5 V** on the pin. That is **not** the same as the manual’s “do not connect” for forward. Only use this wiring if the BLD-515C control inputs tolerate a logic high; otherwise use open-collector buffering or level shifting per the datasheet.

Constants in the sketch: `BRAKE_WHEN_STOPPED` (default **true**—assert BK when stopped), `FR_EN_OFF_DELAY_US` (EN off before F/R change).

## Serial contract (host → Arduino)

One maneuver per line, **IDLE only** (lines are ignored while a move is running):

- Format: `distance_in,angle_rad` then newline  
  - Example: `6.0,0.0` — 6 inches along angle 0 (robot +X).

Responses include: `ACK MOVE`, `ACK DONE`, `ERR BAD_LINE`, `ERR TIMEOUT`, `ERR CMD_OVERFLOW`.

Optional: set `#define DEBUG_TELEMETRY 1` for periodic `PG` total counts when idle.

## Mega 2560 pinout (per driver)

| Wheel | SV (PWM out) | F/R (Mega → driver) | EN  | BK (Mega → driver) | PG (input) |
| ----- | ------------ | ------------------- | --- | ------------------ | ---------- |
| FL    | D2           | D22                 | D30 | D34                | D18        |
| FR    | D3           | D23                 | D31 | D35                | D19        |
| RL    | D5           | D24                 | D32 | D36                | D20        |
| RR    | D6           | D25                 | D33 | D37                | D21        |

Shared wiring:

- Mega **GND** to every driver signal **GND** (common reference required).
- **SV**: one RC low-pass per channel from PWM pin to driver **SV**.
- **ALM** not used in this sketch unless you add handling.

## BLD-515C wiring groups

- Power: `VP`, `GND`
- Motor phases: `MA`, `MB`, `MC`
- Hall: `GND`, `HA`, `HB`, `HC`, `+5V`
- Control: `GND`, `F/R`, `EN`, `BK`
- Speed: `SV`
- Outputs: `PG`, `ALM`, `+5V`

## RC filter guidance for SV

- PWM pin → series `R` → node → `SV`
- Node → `C` to signal ground

Starter values: `R = 2.2 kΩ`, `C = 0.1–1.0 µF`.

## Control summary

- Closed-loop **scalar distance PID** along the commanded direction; **angle** sets direction only (`omega = 0` body command).
- Odometry from **PG** pulse counts and **F/R** sense (forward vs reverse).
- Tune `MOTOR_POLE_PAIRS`, PID gains, and geometry in the sketch.

## Bring-up procedure

1. One driver + one motor first; common ground.
2. Upload sketch; Serial Monitor `115200`.
3. Send a short move, e.g. `1.0,0.0` then newline; expect `ACK MOVE` then `ACK DONE` (or adjust PID/timing if it times out).
4. Confirm direction: forward drives F/R **HIGH**; reverse drives F/R **LOW** (ground).
5. Confirm brake: with `BRAKE_WHEN_STOPPED` true, motors stopped should drive BK **LOW** (grounded).
6. Repeat per channel, then all four.

## Deferred / optional

- `DEBUG_TELEMETRY` / `ALM` handling
- PWM-modulated BK for variable braking (hardware-dependent)
- EN polarity alignment if your wiring uses “grounded = run” literally
