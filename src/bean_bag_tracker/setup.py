import os

from glob import glob

from setuptools import find_packages, setup

package_name = 'bean_bag_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    # NumPy 2.x breaks ROS Jazzy ``cv_bridge`` (Boost bindings built against NumPy 1.x).
    install_requires=['setuptools', 'matplotlib', 'numpy>=1.23.0,<2', 'ultralytics'],
    zip_safe=True,
    maintainer='cornholio',
    maintainer_email='cornholio@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_listener = bean_bag_tracker.camera_listener:main',
            'ros2_bag_sense_1 = bean_bag_tracker.ros2_bag_sense_1:main',
            'ros2_bag_sense_fast = bean_bag_tracker.ros2_bag_sense_fast:main',
            'ros2_bag_sense_manual_depth = bean_bag_tracker.ros2_bag_sense_manual_depth:main',
            'bean_bag_nn_detector = bean_bag_tracker.bean_bag_nn_detector:main',
            'realsense_preview_to_file = bean_bag_tracker.realsense_preview_to_file:main',
            'train_unlabelled_capture = bean_bag_tracker.train_unlabelled_capture:main',
            'tracker_node = bean_bag_tracker.tracker_node:main'
        ],
    },
)
