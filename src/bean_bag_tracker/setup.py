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
    ],
    install_requires=['setuptools', 'matplotlib'],
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
            'ros2_bag_sense_fast = bean_bag_tracker.ros2_bag_sense_fast:main'
        ],
    },
)
