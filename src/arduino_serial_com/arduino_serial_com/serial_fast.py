# Write data to serial port
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

while(True):
    msg=input()
    serial_port.write(msg.encode('utf-8'))
    print(f'Sent to Arduino: "{msg}"')