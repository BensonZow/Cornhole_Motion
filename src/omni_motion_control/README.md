# omni_motion_control

ROS 2 package for kinematics and serial control around an existing
`bean_bag_tracker` node.

## Plain-English Overview

- `omni_kinematics_node`: takes target distance/angle and converts it into robot motion and wheel commands.
- `serial_motor_bridge_node`: sends those commands to your firmware over serial and reads responses back.
- `motor_telemetry_node`: converts raw serial text into structured ROS topics (feedback/events/safety).

## Topic Contracts

- `/motion/target_da` (`geometry_msgs/Vector3`)
  - `x = distance_in`
  - `y = angle_rad`
  - `z = confidence` (optional)
- `/motion/wheel_cmd` (`std_msgs/Int16MultiArray`, 4 values)
  - `[FL, FR, RL, RR]`, range `-255..255`
- `/motion/safety_stop` (`std_msgs/Bool`)
  - `true` => bridge sends stop commands
- `/motion/serial_rx` (`std_msgs/String`)
  - raw lines from firmware, e.g. `ACK DA`, `PG ...`, `WATCHDOG STOP`
- `/motion/motor_feedback` (`std_msgs/Int32MultiArray`, 4 values)
  - parsed PG counts `[FL, FR, RL, RR]`

## Executables

- `omni_kinematics_node`
- `serial_motor_bridge_node`
- `motor_telemetry_node`

## Bridge Modes

- `raw_wheels`: sends `m <FL> <FR> <RL> <RR>`
- `da`: sends `DA <distance_in> <angle_rad>`

Set mode via launch arg:

```bash
ros2 launch omni_motion_control omni_motion_control.launch.py command_mode:=da
```
