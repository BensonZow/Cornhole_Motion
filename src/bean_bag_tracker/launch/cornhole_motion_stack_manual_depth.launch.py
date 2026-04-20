# Copyright 2026 Cornhole_Motion contributors.
#
# Default: opens **three separate GNOME Terminal windows** (camera, manual-depth tracker, omni PWM)
# so each process has its own scrollback and visibility.
#
# Prerequisites: ``source install/setup.bash`` in the shell where you run ``ros2 launch`` (so the
# workspace install path is discoverable). Requires ``gnome-terminal`` and a desktop ``DISPLAY``.
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack_manual_depth.launch.py
#
# Single-terminal / headless (camera + nodes in this launch, no extra windows)::
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack_manual_depth.launch.py spawn_split_terminals:=false
#
# If ament cannot find ``bean_bag_tracker`` (unsourced workspace), pass setup explicitly::
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack_manual_depth.launch.py \\
#     install_setup_bash:=/home/you/ros2_jazzy/install/setup.bash

import os
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
def _resolve_install_setup_bash(context) -> str:
    override = (context.launch_configurations.get('install_setup_bash') or '').strip()
    if override:
        return override
    try:
        prefix = Path(get_package_prefix('bean_bag_tracker')).resolve()
        return str(prefix.parent / 'setup.bash')
    except Exception:
        return ''


def _opaque_launch_setup(context, *args, **kwargs):
    split = (context.launch_configurations.get('spawn_split_terminals') or 'true').lower() in (
        '1',
        'true',
        'yes',
    )
    if split:
        return _actions_split_terminals(context)
    return _actions_inline_stack(context)


def _actions_split_terminals(context):
    setup_bash = _resolve_install_setup_bash(context)
    if not setup_bash:
        raise RuntimeError(
            'spawn_split_terminals is true but install/setup.bash could not be resolved. '
            'Source your workspace (`source install/setup.bash`) before `ros2 launch`, or pass '
            'install_setup_bash:=/absolute/path/to/ros2_jazzy/install/setup.bash'
        )
    if not os.path.isfile(setup_bash):
        raise RuntimeError(f'install_setup_bash does not exist: {setup_bash!r}')

    ros_distro = os.environ.get('ROS_DISTRO', 'jazzy')
    ros_setup = f'/opt/ros/{ros_distro}/setup.bash'
    q_setup = shlex.quote(setup_bash)
    q_ros = shlex.quote(ros_setup)

    min_z = context.launch_configurations.get('min_z_meters', '0.2')
    max_z = context.launch_configurations.get('max_z_meters', '4.0')

    def _bash_lc(title: str, middle_cmd: str) -> str:
        # Keep shell open after the ROS command exits so logs stay visible.
        return (
            f'set -e; echo "=== {title} ==="; '
            f'source {q_ros} && source {q_setup} && {middle_cmd}; '
            f'echo "=== {title} exited with code $? ==="; exec bash'
        )

    cam_cmd = _bash_lc(
        'RealSense',
        'ros2 launch realsense2_camera rs_launch.py initial_reset:=true',
    )
    tracker_cmd = _bash_lc(
        'ros2_bag_sense_manual_depth',
        (
            'ros2 run bean_bag_tracker ros2_bag_sense_manual_depth --ros-args '
            f'-p min_z_meters:={min_z} -p max_z_meters:={max_z}'
        ),
    )
    pwm_cmd = _bash_lc('omni_pwm_subscribe', 'ros2 run arduino_serial_com omni_pwm_subscribe')

    return [
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--title=RealSense (rs_launch)',
                '--',
                'bash',
                '-lc',
                cam_cmd,
            ],
            name='gnome_terminal_realsense',
        ),
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--title=manual-depth bag tracker',
                '--',
                'bash',
                '-lc',
                tracker_cmd,
            ],
            name='gnome_terminal_manual_depth',
        ),
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--title=omni_pwm_subscribe',
                '--',
                'bash',
                '-lc',
                pwm_cmd,
            ],
            name='gnome_terminal_omni_pwm',
        ),
    ]


def _actions_inline_stack(context):
    share = get_package_share_directory('realsense2_camera')
    rs_launch = os.path.join(share, 'rs_launch.py')

    min_z = float(context.launch_configurations.get('min_z_meters', '0.2'))
    max_z = float(context.launch_configurations.get('max_z_meters', '4.0'))

    from launch_ros.actions import Node

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rs_launch),
            launch_arguments={'initial_reset': 'true'}.items(),
        ),
        Node(
            package='bean_bag_tracker',
            executable='ros2_bag_sense_manual_depth',
            name='ros2_bag_sense_manual_depth',
            output='screen',
            parameters=[{'min_z_meters': min_z, 'max_z_meters': max_z}],
        ),
        Node(
            package='arduino_serial_com',
            executable='omni_pwm_subscribe',
            name='omni_pwm_subscribe',
            output='screen',
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'spawn_split_terminals',
                default_value='true',
                description=(
                    'If true, start RealSense, manual-depth tracker, and omni PWM each in its own '
                    'gnome-terminal window. If false, run all nodes in this launch (single terminal).'
                ),
            ),
            DeclareLaunchArgument(
                'install_setup_bash',
                default_value='',
                description='Path to workspace install/setup.bash (optional if environment is already sourced).',
            ),
            DeclareLaunchArgument(
                'min_z_meters',
                default_value='0.2',
                description='Passed to ros2_bag_sense_manual_depth.',
            ),
            DeclareLaunchArgument(
                'max_z_meters',
                default_value='4.0',
                description='Passed to ros2_bag_sense_manual_depth.',
            ),
            OpaqueFunction(function=_opaque_launch_setup),
        ]
    )
