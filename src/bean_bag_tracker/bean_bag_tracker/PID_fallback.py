#!/usr/bin/env python3
"""
Red Ball Tracking and Horizontal Offset Publisher (ROS2)
For Raspberry Pi 5 with ROS2 Jazzy and Intel RealSense D415
Subscribes to existing RealSense ROS2 node for camera feed.
Uses color detection (red/purple) and publishes:
    - absolute horizontal distance from center (non‑negative)
    - radians: 0.0 (left side) or π (right side)
Publishes as std_msgs/Float32MultiArray to 'bean_bag_tracker'.
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional, Tuple

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# Configuration
@dataclass
class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS = 30

    # ROS2 topic for RealSense color image
    CAMERA_TOPIC = '/camera/camera/color/image_raw'  # Adjust to your actual topic

    # ROS2 topic for publishing offset data
    OUTPUT_TOPIC = 'bean_bag_tracker'

    # Red color detection (HSV ranges)
    RED_LOWER_1 = np.array([0, 120, 70])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 120, 70])
    RED_UPPER_2 = np.array([180, 255, 255])

    # Purple color detection (commented out for future use)
    # PURPLE_LOWER = np.array([125, 50, 50])
    # PURPLE_UPPER = np.array([150, 255, 255])

    # Ball detection
    MIN_RADIUS = 10
    MAX_RADIUS = 10000
    MIN_AREA = 100

    # System parameters
    TARGET_X = CAMERA_WIDTH // 2

    # Prediction settings
    PREDICTION_WINDOW = 5
    SMOOTHING_WINDOW = 3

    # Debug settings
    SHOW_VIDEO = True
    PRINT_STATS = True

class BallTrackerNode(Node):
    """
    ROS2 Node that subscribes to RealSense camera topic and performs color-based ball tracking.
    Publishes horizontal offset and direction flag.
    """

    def __init__(self, config):
        super().__init__('ball_tracker_node')
        self.config = config
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

        # Create subscription to RealSense color topic
        self.subscription = self.create_subscription(
            Image,
            self.config.CAMERA_TOPIC,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to {self.config.CAMERA_TOPIC}")

        # Create publisher for offset data
        self.publisher = self.create_publisher(Float32MultiArray, self.config.OUTPUT_TOPIC, 10)
        self.get_logger().info(f"Publishing to {self.config.OUTPUT_TOPIC}")

        # Position tracking
        self.positions = deque(maxlen=config.SMOOTHING_WINDOW)
        self.velocities = deque(maxlen=config.PREDICTION_WINDOW)

        # Tracking state
        self.running = True

    def image_callback(self, msg):
        """ROS2 callback for incoming camera images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
            self.new_frame_event.set()
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def get_frame(self, timeout=1.0):
        """Wait for a new frame and return it."""
        if self.new_frame_event.wait(timeout=timeout):
            self.new_frame_event.clear()
            with self.frame_lock:
                return self.latest_frame.copy() if self.latest_frame is not None else None
        return None

    def detect_ball(self, frame):
        """Detect red ball in the frame."""
        if frame is None:
            return None, None

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.config.RED_LOWER_1, self.config.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.config.RED_LOWER_2, self.config.RED_UPPER_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Uncomment for purple detection if needed:
        # purple_mask = cv2.inRange(hsv, self.config.PURPLE_LOWER, self.config.PURPLE_UPPER)
        # red_mask = purple_mask  # swap mask

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, red_mask

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if contour is large enough
        if cv2.contourArea(largest_contour) < self.config.MIN_AREA:
            return None, red_mask

        # Find enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Convert to integers
        x, y, radius = int(x), int(y), int(radius)

        # Check radius limits
        if radius < self.config.MIN_RADIUS or radius > self.config.MAX_RADIUS:
            return None, red_mask

        return (x, y, radius), red_mask

    def predict_position(self, position: Tuple[int, int, int]) -> Tuple[int, int]:
        """Predict future position based on recent positions."""
        self.positions.append(position)

        if len(self.positions) < 2:
            return position[0], position[1]

        # Calculate velocity
        if len(self.positions) >= 2:
            prev_pos = self.positions[-2]
            curr_pos = self.positions[-1]
            dt = 1.0 / self.config.FPS

            velocity = (
                (curr_pos[0] - prev_pos[0]) / dt,
                (curr_pos[1] - prev_pos[1]) / dt
            )
            self.velocities.append(velocity)

        # Simple linear prediction
        if len(self.velocities) > 0:
            avg_vx = np.mean([v[0] for v in self.velocities])
            avg_vy = np.mean([v[1] for v in self.velocities])

            predict_time = 0.1
            pred_x = position[0] + avg_vx * predict_time
            pred_y = position[1] + avg_vy * predict_time

            pred_x = int(max(0, min(self.config.CAMERA_WIDTH - 1, pred_x)))
            pred_y = int(max(0, min(self.config.CAMERA_HEIGHT - 1, pred_y)))

            return pred_x, pred_y

        return position[0], position[1]

    def draw_detection(self, frame, ball_data, pred_position):
        if frame is None:
            return frame

        if ball_data is not None:
            x, y, radius = ball_data
            pred_x, pred_y = pred_position

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(pred_x), int(pred_y)), 10, (255, 255, 0), 2)

        cv2.line(frame, (self.config.TARGET_X, 0),
                 (self.config.TARGET_X, self.config.CAMERA_HEIGHT),
                 (255, 0, 0), 2)

        return frame

    def draw_text(self, frame, ball_data, pred_position):
        if frame is None:
            return frame

        if ball_data is not None:
            x, y, radius = ball_data
            pred_x, pred_y = pred_position
            cv2.putText(frame, f"Ball: ({x}, {y}, r={radius})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pred: ({pred_x}, {pred_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No ball detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Target X: {self.config.TARGET_X}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def process_frame(self, frame):
        """Process a single frame and return detection result and annotated frame."""
        ball_data, mask = self.detect_ball(frame)
        pred_x = pred_y = radius = None
        annotated_frame = frame.copy() if frame is not None else None

        if ball_data is not None:
            x, y, radius = ball_data
            pred_x, pred_y = self.predict_position((x, y, radius))

            if self.config.SHOW_VIDEO and annotated_frame is not None:
                annotated_frame = self.draw_detection(annotated_frame, ball_data, (pred_x, pred_y))
                annotated_frame = self.draw_text(annotated_frame, ball_data, (pred_x, pred_y))

                if mask is not None:
                    mask_resized = cv2.resize(mask, (160, 120))
                    mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                    annotated_frame[0:120, 0:160] = mask_colored
        else:
            if self.config.SHOW_VIDEO and annotated_frame is not None:
                annotated_frame = self.draw_text(annotated_frame, None, (0, 0))
                cv2.line(annotated_frame, (self.config.TARGET_X, 0),
                         (self.config.TARGET_X, self.config.CAMERA_HEIGHT),
                         (255, 0, 0), 2)

        return pred_x, pred_y, radius, annotated_frame

    def stop(self):
        self.running = False

class BallTrackerSystem:
    """Main system for tracking balls and publishing offset data."""

    def __init__(self):
        self.config = Config()

        # Initialize ROS2
        rclpy.init()
        self.tracker_node = BallTrackerNode(self.config)

        # State
        self.running = True
        self.publish_enabled = False  # Start in pause mode
        self.ball_detected = False

        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

    def calculate_offset_data(self, ball_x: float) -> Tuple[float, float]:
        """
        Compute absolute horizontal distance from center and radians direction flag.
        Returns (distance, radians) where radians is 0.0 for left, π for right.
        """
        distance = abs(ball_x - self.config.TARGET_X)
        if ball_x < self.config.TARGET_X:
            radians = 0.0       # left side
        else:
            radians = math.pi   # right side
        return distance, radians

    def run(self):
        print("Starting ROS2 color-based tracking system (offset publisher)...")
        print("Controls:")
        print("  ENTER : Start publishing")
        print("  SPACE : Pause publishing")
        print("  'q'   : Quit")
        print("  's'   : Stop publishing (same as pause)")
        print("System starts in PAUSE mode. Press ENTER to begin publishing.")

        last_print_time = time.time()

        try:
            while self.running and rclpy.ok():
                # Spin ROS2 once to process callbacks
                rclpy.spin_once(self.tracker_node, timeout_sec=0.01)

                frame_start_time = time.time()

                # Get latest frame from ROS2 topic
                frame = self.tracker_node.get_frame(timeout=0.1)
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Process frame through color detection
                ball_x, ball_y, radius, annotated_frame = self.tracker_node.process_frame(frame)

                self.frame_count += 1

                if ball_x is not None:
                    self.detection_count += 1
                    self.ball_detected = True

                    # Calculate offset data
                    distance, radians = self.calculate_offset_data(float(ball_x))

                    if self.publish_enabled:
                        # Publish as Float32MultiArray
                        msg = Float32MultiArray()
                        msg.data = [distance, radians]
                        self.tracker_node.publisher.publish(msg)

                    # Stats print
                    current_time = time.time()
                    if current_time - last_print_time > 1.0 and self.config.PRINT_STATS:
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        detect_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                        mode_str = "PUBLISH" if self.publish_enabled else "PAUSE"
                        side_str = "LEFT" if ball_x < self.config.TARGET_X else "RIGHT"
                        print(f"[{mode_str}] FPS: {fps:.1f}, Ball X: {int(ball_x)} ({side_str}), "
                              f"Dist: {distance:.1f}, Rad: {radians:.2f}, Detect: {detect_rate:.1f}%")
                        last_print_time = current_time
                else:
                    self.ball_detected = False

                # Display
                if self.config.SHOW_VIDEO and annotated_frame is not None:
                    # Show mode status
                    mode_text = "MODE: PUBLISH" if self.publish_enabled else "MODE: PAUSE (Press ENTER to start)"
                    mode_color = (0, 255, 0) if self.publish_enabled else (0, 0, 255)
                    cv2.putText(annotated_frame, mode_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                    # Status text
                    status = "BALL TRACKING" if self.ball_detected else "SEARCHING"
                    status_color = (0, 255, 0) if self.ball_detected else (0, 0, 255)
                    cv2.putText(annotated_frame, f"Status: {status}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    if self.ball_detected and ball_x is not None:
                        distance, radians = self.calculate_offset_data(float(ball_x))
                        cv2.putText(annotated_frame, f"Dist: {distance:.1f} px, Rad: {radians:.2f}", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # FPS
                    frame_time = time.time() - frame_start_time
                    fps_disp = 1.0 / frame_time if frame_time > 0 else 0
                    cv2.putText(annotated_frame, f"FPS: {fps_disp:.1f}", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Ball Tracker - Offset Publisher (ROS2)", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.publish_enabled = False
                        print("Publishing paused")
                    elif key == 13 or key == 10:  # Enter key
                        self.publish_enabled = True
                        print("Publishing ENABLED")
                    elif key == 32:  # Space key
                        self.publish_enabled = False
                        print("Publishing PAUSED")

                # Maintain approximate frame rate
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                target_time = 1.0 / self.config.FPS
                if processing_time < target_time:
                    time.sleep(target_time - processing_time)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        print("\nCleaning up...")
        self.running = False
        self.tracker_node.stop()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

        if self.config.PRINT_STATS:
            total_time = time.time() - self.start_time
            print(f"\nFinal Statistics:")
            print(f"Total time: {total_time:.1f} s")
            print(f"Total frames: {self.frame_count}")
            print(f"Total detections: {self.detection_count}")
            if total_time > 0:
                print(f"Average FPS: {self.frame_count/total_time:.1f}")
            if self.frame_count > 0:
                print(f"Detection rate: {(self.detection_count/self.frame_count)*100:.1f}%")

def main():
    system = BallTrackerSystem()

    print("=" * 60)
    print("RED BALL OFFSET PUBLISHER (ROS2)")
    print("Subscribing to RealSense camera topic")
    print("=" * 60)
    print(f"ROS2 Input Topic:  {system.config.CAMERA_TOPIC}")
    print(f"ROS2 Output Topic: {system.config.OUTPUT_TOPIC}")
    print(f"Camera: {system.config.CAMERA_WIDTH}x{system.config.CAMERA_HEIGHT} @ {system.config.FPS}FPS")
    print(f"Target X: {system.config.TARGET_X}")
    print("=" * 60)
    print("Publishes: [distance_from_center (px), radians (0.0 left, π right)]")
    print("System starts in PAUSE mode.")
    print("Press ENTER to start publishing, SPACE to pause.")
    print("=" * 60)

    system.run()

if __name__ == "__main__":
    main()
