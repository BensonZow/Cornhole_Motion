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


Serial Write 

Pi -> Arduino 
check Arduino is connected to Pi serial port 
 
sudo dmesg | grep tty 
ls /dev/ttyACM* 

to write across serial port 
On Pi in arduino_serial_com pkg 
python3 serial_fast.py 


ultralytics env 
# Create and activate the environment
python3 -m venv ultralytics_env
source ultralytics_env/bin/activate

ros2 run bean_bag_tracker bean_bag_nn_detector

# NN segfault (Pi / mixed pip): keep numpy<2 for cv_bridge; the node sets BLAS threads=1.
# If it still crashes, remove pip OpenCV so ROS uses apt cv2 only:
#   pip3 uninstall -y opencv-python opencv-python-headless
# Then: colcon build --packages-select bean_bag_tracker --symlink-install

# NN install / weights
# After ``pip install -U ultralytics``, pin NumPy again for ROS cv_bridge:
#   pip3 install 'numpy>=1.26.0,<2' --break-system-packages
# Official yolo26n.pt must be an Ultralytics checkpoint (not a random .pt renamed).
# Download with Ultralytics, then copy into src/bean_bag_tracker/models/ and rebuild:
#   mkdir -p /tmp/yolodl && cd /tmp/yolodl && python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
#   cp yolo26n.pt ~/ros2_jazzy/src/bean_bag_tracker/models/
#   cd ~/ros2_jazzy && colcon build --packages-select bean_bag_tracker --symlink-install