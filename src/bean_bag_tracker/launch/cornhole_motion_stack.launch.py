# Copyright 2026 Cornhole_Motion contributors.
#
# Stack: YOLO ``.pt`` detector + bag trajectory + Arduino omni PWM (start RealSense / camera separately).
#
# From your workspace (after ``colcon build``)::
#
#   source install/setup.bash
#   ros2 launch bean_bag_tracker cornhole_motion_stack.launch.py
#
# With explicit weights (otherwise default is ``share/bean_bag_tracker/models/yolo26n.pt``)::
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack.launch.py \
#     model_path:=/full/path/to/yolo26n.pt
#
# Optional detector tuning::
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack.launch.py \
#     model_path:=/full/path/to/weights.pt device:=cpu confidence_threshold:=0.5
#
# Requires ``ultralytics`` (PyTorch) in the Python environment used by ``ros2 launch``.
# One-time (Debian/Ubuntu): ``pip3 install ultralytics --break-system-packages``
# If ``cv_bridge`` fails with NumPy 2 vs 1.x: ``pip3 install 'numpy>=1.26,<2' --break-system-packages``
# ``ros2_bag_sense_fast`` time-syncs color + depth + ``/bean_bag_detection``, so the
# detector node must be running for the tracker to receive synchronized callbacks.
#
# Depth-only manual labeling (no NN / PyTorch)::
#
#   ros2 launch bean_bag_tracker cornhole_motion_stack.launch.py manual_depth_tracker:=true
#
# Same stack split across terminals::
#
#   # Terminal 1
#   source install/setup.bash
#   ros2 launch realsense2_camera rs_launch.py initial_reset:=true
#
#   # Terminal 2
#   source install/setup.bash
#   ros2 run bean_bag_tracker bean_bag_nn_detector --ros-args -p model_path:=/path/to/weights.pt
#   ros2 run bean_bag_tracker ros2_bag_sense_fast
#
#   # Terminal 3
#   source install/setup.bash
#   ros2 run arduino_serial_com omni_pwm_subscribe

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description=(
            'Path to YOLO ``.pt`` weights. Empty uses share/bean_bag_tracker/models/yolo26n.pt '
            'after install.'
        ),
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Torch device for the detector (e.g. cpu, cuda:0).',
    )
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Minimum box confidence before considering detections.',
    )
    manual_depth_arg = DeclareLaunchArgument(
        'manual_depth_tracker',
        default_value='false',
        description=(
            'If true, launch ``ros2_bag_sense_manual_depth`` (depth probe + manual ROI) '
            'and skip YOLO + ``ros2_bag_sense_fast``.'
        ),
    )

    bag_detector = Node(
        package='bean_bag_tracker',
        executable='bean_bag_nn_detector',
        name='bean_bag_nn_detector',
        output='screen',
        parameters=[
            {
                'detection_topic': '/bean_bag_detection',
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            }
        ],
        condition=UnlessCondition(LaunchConfiguration('manual_depth_tracker')),
    )

    bag_tracker = Node(
        package='bean_bag_tracker',
        executable='ros2_bag_sense_fast',
        name='ros2_bag_sense_fast',
        output='screen',
        parameters=[
            {
                'bag_detection_topic': '/bean_bag_detection',
                'min_z_meters': 0.2,
                'max_z_meters': 4.0,
            }
        ],
        condition=UnlessCondition(LaunchConfiguration('manual_depth_tracker')),
    )

    bag_tracker_manual = Node(
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
        condition=IfCondition(LaunchConfiguration('manual_depth_tracker')),
    )

    omni_pwm = Node(
        package='arduino_serial_com',
        executable='omni_pwm_subscribe',
        name='omni_pwm_subscribe',
        output='screen',
    )

    return LaunchDescription(
        [
            model_path_arg,
            device_arg,
            confidence_threshold_arg,
            manual_depth_arg,
            bag_detector,
            bag_tracker,
            bag_tracker_manual,
            omni_pwm,
        ]
    )
