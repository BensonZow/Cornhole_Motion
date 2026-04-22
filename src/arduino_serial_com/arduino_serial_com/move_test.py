import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math

class QuadrantTestPublisher(Node):
    def __init__(self):
        super().__init__('quadrant_test_publisher')
        # Using the parameter-based topic name from your snippet
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        topic_name = self.get_parameter('result_topic').value
        
        self.publisher = self.create_publisher(Float32MultiArray, topic_name, 10)
        
        # Quadrant centers: 45, 135, 225, 315 degrees
        self.test_angles = [math.radians(a) for a in [45, 135, 225, 315]]
        self.range_val = 0.1
        self.index = 0
        
        # Publish every 2 seconds to allow for observation
        self.timer = self.create_timer(2.0, self.timer_callback)

    def timer_callback(self):
        if self.index >= len(self.test_angles):
            self.get_logger().info('Test sequence complete.')
            self.index = 0  # Loop back for continuous testing
            return

        msg = Float32MultiArray()
        current_rad = self.test_angles[self.index]
        
        # Your specific [range, radian] format
        msg.data = [self.range_val, current_rad]
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: Range={msg.data[0]}, Radian={msg.data[1]:.4f} (Quadrant {self.index + 1})')
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = QuadrantTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
