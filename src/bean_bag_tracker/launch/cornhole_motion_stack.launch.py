# Copyright 2026 Cornhole_Motion contributors.
#
# Stack: RealSense driver + bean bag tracker + Arduino omni PWM subscriber.
#
# One terminal (after ``colcon build`` and ``source install/setup.bash``)::
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack.launch.py
#
# Same stack split across three terminals (separate logs / easier to restart one part)::
#
#   # Terminal 1
#   source install/setup.bash
#   ros2 launch realsense2_camera rs_launch.py initial_reset:=true
#
#   # Terminal 2  (executable name has no ``.py`` suffix)
#   source install/setup.bash
#   ros2 run bean_bag_tracker ros2_bag_sense_fast
#
# Optional: require a minimum purple blob size in pixels (OpenCV contour area), e.g.::
#
#   ros2 run bean_bag_tracker ros2_bag_sense_fast --ros-args -p min_purple_contour_area_px:=300.0
#
# Or add ``parameters=[{'min_purple_contour_area_px': 300.0}]`` to the ``bag_tracker`` Node below.
#
#   # Terminal 3
#   source install/setup.bash
#   ros2 run arduino_serial_com omni_pwm_subscribe

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py',
                ]
            )
        ),
        launch_arguments=[('initial_reset', 'true')],
    )

    # Default min_purple_contour_area_px is 0 (disabled); see header comment to enable.
    bag_tracker = Node(
        package='bean_bag_tracker',
        executable='ros2_bag_sense_fast',
        name='ros2_bag_sense_fast',
        output='screen',
    )

    omni_pwm = Node(
        package='arduino_serial_com',
        executable='omni_pwm_subscribe',
        name='omni_pwm_subscribe',
        output='screen',
    )

    return LaunchDescription(
        [
            realsense,
            bag_tracker,
            omni_pwm,
        ]
    )
