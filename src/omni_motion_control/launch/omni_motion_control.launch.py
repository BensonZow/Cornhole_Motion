from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # User-editable launch arguments:
    # - serial_port: which USB/serial device the firmware is on
    # - baud_rate: must match firmware serial speed
    # - command_mode: "da" (distance/angle) or "raw_wheels" (direct wheel commands)
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port connected to Arduino firmware',
    )
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Arduino serial baud rate',
    )
    command_mode_arg = DeclareLaunchArgument(
        'command_mode',
        default_value='da',
        description='Bridge mode: raw_wheels or da',
    )

    # Node 1: Converts target distance/angle into robot/wheel commands.
    kinematics_node = Node(
        package='omni_motion_control',
        executable='omni_kinematics_node',
        name='omni_kinematics_node',
        output='screen',
    )

    # Node 2: Sends commands to firmware over serial and reads serial lines back.
    serial_bridge_node = Node(
        package='omni_motion_control',
        executable='serial_motor_bridge_node',
        name='serial_motor_bridge_node',
        output='screen',
        parameters=[
            {
                'serial_port': LaunchConfiguration('serial_port'),
                'baud_rate': LaunchConfiguration('baud_rate'),
                'command_mode': LaunchConfiguration('command_mode'),
            }
        ],
    )

    # Node 3: Parses serial text into structured telemetry topics.
    telemetry_node = Node(
        package='omni_motion_control',
        executable='motor_telemetry_node',
        name='motor_telemetry_node',
        output='screen',
    )

    # Launch all arguments and all three cooperating nodes.
    return LaunchDescription(
        [
            serial_port_arg,
            baud_rate_arg,
            command_mode_arg,
            kinematics_node,
            serial_bridge_node,
            telemetry_node,
        ]
    )
