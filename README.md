ROS2 Tutorial 
https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html

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
