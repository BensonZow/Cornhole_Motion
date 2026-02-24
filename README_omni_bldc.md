# Omni BLDC (Arduino-Only) Integration

This document covers the new omni-wheel BLDC integration path using:
- Arduino Mega 2560
- 4x BLD-515C drivers
- 4x brushless geared motors
- ROS2 output (`distance`, `angle`) sent to Arduino over USB serial

This path is intentionally open-loop for now (no PID/fusion in this phase).

## Firmware

- Sketch: `firmware/omni_bldc_firmware/omni_bldc_firmware.ino`
- Serial baud: `115200`
- Control mode: `SV` speed command via PWM+RC, with `F/R`, `EN`, and `BK`.

## Serial Contract (ROS2 -> Arduino)

ROS2 currently publishes:
- `distance` in inches
- `angle` in radians

Command lines accepted by Arduino:

- `DA <distance_in> <angle_rad>`
  - Example: `DA 15.2 -0.37`
- `STOP`
- `PING`
- `HELP`

Arduino responses:
- `ACK DA`
- `ACK STOP`
- `PONG`
- `ERR ...`
- `WATCHDOG STOP` when command stream times out
- `PG <fl_count> <fr_count> <rl_count> <rr_count>` telemetry

## Mega 2560 Pinout (Per Driver)

| Wheel | SV (PWM out) | DIR (`F/R`) | EN | BK | PG (input) |
|------|---------------|-------------|----|----|------------|
| FL | D2 | D22 | D30 | D34 | D18 |
| FR | D3 | D23 | D31 | D35 | D19 |
| RL | D5 | D24 | D32 | D36 | D20 |
| RR | D6 | D25 | D33 | D37 | D21 |

Shared wiring:
- Mega `GND` to every driver signal `GND` (common reference required).
- `SV` gets one RC low-pass per channel from each PWM pin.
- `BK` is held inactive in firmware during phase-1.
- `ALM` is not used in phase-1.

## BLD-515C Wiring Groups

From driver terminal groups:
- Power: `VP`, `GND`
- Motor phases: `MA`, `MB`, `MC`
- Hall group: `GND`, `HA`, `HB`, `HC`, `+5V`
- Control: `GND`, `F/R`, `EN`, `BK`
- Speed input: `SV`
- Outputs: `PG`, `ALM`, `+5V`

For this firmware:
- Arduino drives `F/R`, `EN`, `BK`, `SV`.
- Arduino reads `PG`.
- `ALM` is left unconnected in this phase.

## RC Filter Guidance For SV

Use one low-pass per `SV` channel:
- PWM pin -> `R` series -> node -> `SV`
- Node -> `C` to signal ground

Starter values:
- `R = 2.2k ohm`
- `C = 0.1 uF` to `1.0 uF`

Tune RC cutoff so ripple is acceptable while command response remains fast enough.

## Basic Onboard Kinematics (Current Phase)

Arduino computes:
- `vx = k_d * distance * cos(angle)`
- `vy = k_d * distance * sin(angle)`
- `omega = k_a * angle`

Then applies omni inverse kinematics to get wheel commands and maps to signed PWM.

Global parameters are in the sketch for:
- Board geometry
- Wheel positions
- Wheel radius
- Gains and limits
- Control and watchdog timing

## Bring-Up Procedure

1. Power only one driver + one motor first.
2. Verify common ground between Mega and driver logic ground.
3. Upload sketch and open Serial Monitor at `115200`.
4. Send `PING` and verify `PONG`.
5. Send small command: `DA 2.0 0.0`.
6. Confirm expected wheel direction and smooth response.
7. Send `STOP` and verify immediate stop.
8. Unplug USB or stop streaming commands; verify `WATCHDOG STOP`.
9. Repeat for each channel.
10. Connect all four drivers after single-channel behavior is confirmed.

## Deferred Items

- Closed-loop PID
- Camera + motor feedback fusion
- Encoder/PG feedback in control loop
- Advanced trajectory tracking
