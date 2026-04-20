# Copyright 2026 Cornhole_Motion contributors.
#
# Depth-only manual-label tracker + omni PWM (no YOLO). Start RealSense separately.
#
#   source install/setup.bash
#   ros2 launch bean_bag_tracker cornhole_motion_stack_manual_depth.launch.py
#
# Labeling uses OpenCV GUI windows (requires DISPLAY).

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bag_tracker = Node(
        package='bean_bag_tracker',
        executable='ros2_bag_sense_manual_depth',
        name='ros2_bag_sense_manual_depth',
        output='screen',
        parameters=[
            {
                'min_z_meters': 0.2,
                'max_z_meters': 4.0,
            }
        ],
    )

    omni_pwm = Node(
        package='arduino_serial_com',
        executable='omni_pwm_subscribe',
        name='omni_pwm_subscribe',
        output='screen',
    )

    return LaunchDescription([bag_tracker, omni_pwm])
