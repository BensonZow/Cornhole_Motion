from setuptools import find_packages, setup

package_name = 'arduino_serial_com'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='cornholio',
    maintainer_email='corn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sender_node = arduino_serial_com.serial_sender:main',
            'omni_wheels_pwm = arduino_serial_com.omni_wheels_pwm:main',
            'omni_pwm_subscribe = arduino_serial_com.omni_pwm_subscribe:main',
        ],
    },
)
