ROS2 Tutorial 
https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html

ip address 

10.201.82.18

Realsense 
https://github.com/realsenseai/realsense-ros?tab=readme-ov-file

    reset camera
   ros2 launch realsense2_camera rs_launch.py initial_reset:=true



Update dependencies 
rosdep install --from-paths src --ignore-src -r -y


Check SSH 
sudo systemctl status ssh

Build ROS2 Package
colcon build --packages-select <package_name> --symlink-install
source install/local_setup.bash



Start up 

source /opt/ros/jazzy/setup.bash

in ~/ros2_jazzy/

source install/setup.bash


Serial Write 

Pi -> Arduino 
check Arduino is connected to Pi serial port 
 
sudo dmesg | grep tty 
ls /dev/ttyACM* 

to write across serial port 
On Pi in arduino_serial_com pkg 
python3 serial_fast.py 
