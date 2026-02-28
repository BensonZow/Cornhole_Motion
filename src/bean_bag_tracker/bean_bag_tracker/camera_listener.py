import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Subscribes to 'topic' with String messages, queue size 10
        self.subscription = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.listener_callback, 10)
        self.subscription
        self.br = CvBridge()

    def listener_callback(self, data):
        #self.get_logger().info('I heard: "%s"' % msg.data)
        frame = self.br.imgmsg_to_cv2(data)
        cv2.imshow("RealSense", frame)

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()