import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

class ByteTrackNode(Node):
    def __init__(self):
        super().__init__('bytetrack_forecaster')
        self.bridge = CvBridge()
        # Initialize YOLO26 with ByteTrack
        self.model = YOLO('/home/cornholio/ros2_jazzy/src/bean_bag_tracker/models/yolo26n.pt') 
        
        # Synchronized Subscriptions
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        # ApproximateTimeSynchronizer matches color/depth frames with similar timestamps
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

    def sync_callback(self, color_msg, depth_msg):
        # Convert ROS to OpenCV
        color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        # 1. Run YOLO26 Tracking with ByteTrack
        # 'persist=True' maintains tracking IDs across frames
        results = self.model.track(color_img, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                cx, cy, w, h = box
                # 2. Extract Depth for the tracked object
                # Depth frames from aligned_depth are in millimeters
                depth_val = depth_img[int(cy), int(cx)] / 1000.0 # Convert to meters

                # 3. Add to your history for trajectory forecasting
                self.get_logger().info(f"ID: {track_id} is at depth: {depth_val:.2f}m")

def main():
    rclpy.init()
    node = ByteTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
