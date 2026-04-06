# Write data to serial port
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

serial_port = serial.Serial('/dev/ttyACM1', 115200, timeout=1)

while(True):
    msg=input()
<<<<<<< HEAD
    serial_port.write((msg+ '\n').encode('utf-8'))
=======
    serial_port.write((msg + '\n').encode('utf-8'))
>>>>>>> refs/remotes/origin/main
    print(f'Sent to Arduino: "{msg}"')