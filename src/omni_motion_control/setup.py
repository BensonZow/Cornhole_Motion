from setuptools import find_packages, setup

# Package name is reused in multiple setup fields below.
package_name = 'omni_motion_control'

setup(
    # Basic package metadata used by ROS 2 and Python packaging tools.
    name=package_name,
    version='0.1.0',
    # Include Python modules, excluding tests from install package.
    packages=find_packages(exclude=['test']),
    data_files=[
        # Register package with ament index.
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Install ROS package manifest.
        ('share/' + package_name, ['package.xml']),
        # Install launch file so "ros2 launch" can find it.
        ('share/' + package_name + '/launch', ['launch/omni_motion_control.launch.py']),
    ],
    # Python runtime dependencies.
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='cornholio',
    maintainer_email='cornholio@todo.todo',
    description='Omni kinematics and serial bridge nodes for Cornhole Motion.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        # ROS 2 command-line executables created from Python entry points.
        'console_scripts': [
            'omni_kinematics_node = omni_motion_control.omni_kinematics_node:main',
            'serial_motor_bridge_node = omni_motion_control.serial_motor_bridge_node:main',
            'motor_telemetry_node = omni_motion_control.motor_telemetry_node:main',
        ],
    },
)
