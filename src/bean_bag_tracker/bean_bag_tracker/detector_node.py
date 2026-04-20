import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os
import time

class YOLOSaveDetector(Node):
    def __init__(self):
        super().__init__('yolo_save_detector')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Detection2DArray, '/detections', 10)
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.model = YOLO('yolo11n.pt') 
        
        # Setup Save Folder
        self.save_path = 'detected_images'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image, verbose=False)

        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        found_any = False

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > 0.5:
                    found_any = True
                    # 1. Prepare ROS Message
                    det = Detection2D()
                    xywh = box.xywh[0].tolist()
                    det.bbox.center.position.x, det.bbox.center.position.y = xywh[0], xywh[1]
                    det.bbox.size_x, det.bbox.size_y = xywh[2], xywh[3]
                    detection_array.detections.append(det)

                    # 2. Draw on image for saving
                    xyxy = box.xyxy[0].tolist()
                    label = f"{self.model.names[int(box.cls[0])]}: {conf:.2f}"
                    cv2.rectangle(cv_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(cv_image, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. Save Image if objects were found
        if found_any:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.save_path, f"detection_{timestamp}.jpg")
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f"Saved detection to {filename}")

        self.publisher.publish(detection_array)

def main():
    rclpy.init()
    node = YOLOSaveDetector()
    rclpy.spin(node)
    rclpy.shutdown()
