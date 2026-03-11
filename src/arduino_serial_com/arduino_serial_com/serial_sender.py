import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

class SerialSenderNode(Node):
    def __init__(self):
        super().__init__('serial_sender_node')
        # Configure serial port (match Arduino baud rate)
        self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        
        # Subscribe to a ROS topic
        self.subscription = self.create_subscription(
            String,
            'serial_data',
            self.listener_callback,
            10)
        self.get_logger().info('Serial Sender Node has started.')

    def listener_callback(self, msg):
        # Write data to serial port
        self.serial_port.write(msg.data.encode('utf-8'))
        self.get_logger().info(f'Sent to Arduino: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = SerialSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.serial_port.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
