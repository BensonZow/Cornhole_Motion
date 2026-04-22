#!/usr/bin/env python3
"""
Red Ball Tracking and Metric X Distance Publisher (ROS2)
Uses RealSense depth + intrinsics to compute horizontal offset in meters.
Publishes [x_distance_meters, radians] to 'bean_bag_tracker'.
"""

import cv2
import numpy as np
import time
import threading
import sys
import select
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional, Tuple

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# pyrealsense2 for deprojection
import pyrealsense2 as rs

# Configuration
@dataclass
class Config:
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    FPS = 30

    # ROS2 topics (adjust if needed)
    COLOR_TOPIC = '/camera/camera/color/image_raw'
    DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
    CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'

    # Output topic
    OUTPUT_TOPIC = 'bean_bag_tracker'

    # Red color detection (HSV ranges)
    RED_LOWER_1 = np.array([0, 120, 70])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 120, 70])
    RED_UPPER_2 = np.array([180, 255, 255])

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
    """ROS2 Node that subscribes to color, depth, and camera info to compute metric X offset."""

    def __init__(self, config):
        super().__init__('ball_tracker_node')
        self.config = config
        self.bridge = CvBridge()

        # Synchronized data storage
        self.color_frame = None
        self.depth_frame = None
        self.camera_info = None
        self.frame_lock = threading.Lock()
        self.new_color_event = threading.Event()

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, config.COLOR_TOPIC, self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, config.DEPTH_TOPIC, self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, config.CAMERA_INFO_TOPIC, self.info_callback, 10)

        self.get_logger().info(f"Subscribed to {config.COLOR_TOPIC}, {config.DEPTH_TOPIC}, {config.CAMERA_INFO_TOPIC}")

        # Publisher
        self.publisher = self.create_publisher(Float32MultiArray, config.OUTPUT_TOPIC, 10)
        self.get_logger().info(f"Publishing to {config.OUTPUT_TOPIC}")

        # Position tracking
        self.positions = deque(maxlen=config.SMOOTHING_WINDOW)
        self.velocities = deque(maxlen=config.PREDICTION_WINDOW)

        self.running = True

    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.color_frame = cv_image
            self.new_color_event.set()
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            # Depth is 16UC1 in millimeters
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.frame_lock:
                self.depth_frame = depth_image
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def info_callback(self, msg):
        with self.frame_lock:
            self.camera_info = msg

    def get_synchronized_data(self, timeout=1.0):
        """Wait for a new color frame and return current color, depth, and intrinsics."""
        if self.new_color_event.wait(timeout=timeout):
            self.new_color_event.clear()
            with self.frame_lock:
                color = self.color_frame.copy() if self.color_frame is not None else None
                depth = self.depth_frame.copy() if self.depth_frame is not None else None
                info = self.camera_info
            return color, depth, info
        return None, None, None

    def detect_ball(self, frame):
        if frame is None:
            return None, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.config.RED_LOWER_1, self.config.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.config.RED_LOWER_2, self.config.RED_UPPER_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, red_mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.config.MIN_AREA:
            return None, red_mask

        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        x, y, radius = int(x), int(y), int(radius)
        if radius < self.config.MIN_RADIUS or radius > self.config.MAX_RADIUS:
            return None, red_mask

        return (x, y, radius), red_mask

    def pixel_to_metric_x(self, pixel_x, pixel_y, depth_frame, camera_info):
        """
        Convert pixel coordinates to metric X coordinate using RealSense deprojection.
        Returns X in meters, or None if depth is invalid.
        """
        if depth_frame is None or camera_info is None:
            return None

        # Get depth value in millimeters
        depth_mm = depth_frame[pixel_y, pixel_x]
        if depth_mm == 0:
            return None  # Invalid depth

        # Build rs2_intrinsics from CameraInfo
        intrinsics = rs.intrinsics()
        intrinsics.width = camera_info.width
        intrinsics.height = camera_info.height
        intrinsics.ppx = camera_info.k[2]  # cx
        intrinsics.ppy = camera_info.k[5]  # cy
        intrinsics.fx = camera_info.k[0]
        intrinsics.fy = camera_info.k[4]
        # Assume Brown Conrady model with zero distortion for simplicity
        intrinsics.model = rs.distortion.brown_conrady
        intrinsics.coeffs = camera_info.d  # distortion coefficients

        # Deproject pixel to point
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_mm / 1000.0)
        # point is [x, y, z] in meters
        return point[0]  # X coordinate

    def predict_position(self, position: Tuple[int, int, int]) -> Tuple[int, int]:
        self.positions.append(position)
        if len(self.positions) < 2:
            return position[0], position[1]

        prev_pos = self.positions[-2]
        curr_pos = self.positions[-1]
        dt = 1.0 / self.config.FPS
        velocity = ((curr_pos[0] - prev_pos[0]) / dt,
                    (curr_pos[1] - prev_pos[1]) / dt)
        self.velocities.append(velocity)

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

    def draw_detection(self, frame, ball_data, pred_position, metric_x=None):
        if frame is None or ball_data is None:
            return frame
        x, y, radius = ball_data
        pred_x, pred_y = pred_position
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (pred_x, pred_y), 10, (255, 255, 0), 2)
        cv2.line(frame, (self.config.TARGET_X, 0),
                 (self.config.TARGET_X, self.config.CAMERA_HEIGHT), (255, 0, 0), 2)
        if metric_x is not None:
            cv2.putText(frame, f"X: {metric_x:.3f} m", (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame

    def draw_text(self, frame, ball_data, pred_position, metric_x=None):
        if frame is None:
            return frame
        if ball_data is not None:
            x, y, radius = ball_data
            pred_x, pred_y = pred_position
            cv2.putText(frame, f"Ball: ({x}, {y}, r={radius})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pred: ({pred_x}, {pred_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if metric_x is not None:
                cv2.putText(frame, f"Metric X: {metric_x:.3f} m", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No ball detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Target X: {self.config.TARGET_X}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def process_frame(self, color_frame, depth_frame, camera_info):
        ball_data, mask = self.detect_ball(color_frame)
        pred_x = pred_y = radius = metric_x = None
        annotated_frame = color_frame.copy() if color_frame is not None else None

        if ball_data is not None:
            x, y, radius = ball_data
            pred_x, pred_y = self.predict_position((x, y, radius))

            # Compute metric X using depth
            if depth_frame is not None and camera_info is not None:
                metric_x = self.pixel_to_metric_x(x, y, depth_frame, camera_info)

            if self.config.SHOW_VIDEO and annotated_frame is not None:
                annotated_frame = self.draw_detection(annotated_frame, ball_data, (pred_x, pred_y), metric_x)
                annotated_frame = self.draw_text(annotated_frame, ball_data, (pred_x, pred_y), metric_x)
                if mask is not None:
                    mask_resized = cv2.resize(mask, (160, 120))
                    mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                    annotated_frame[0:120, 0:160] = mask_colored
        else:
            if self.config.SHOW_VIDEO and annotated_frame is not None:
                annotated_frame = self.draw_text(annotated_frame, None, (0, 0))
                cv2.line(annotated_frame, (self.config.TARGET_X, 0),
                         (self.config.TARGET_X, self.config.CAMERA_HEIGHT), (255, 0, 0), 2)

        return pred_x, pred_y, radius, metric_x, annotated_frame

    def stop(self):
        self.running = False

class BallTrackerSystem:
    """Main system with terminal command input."""

    def __init__(self):
        self.config = Config()
        rclpy.init()
        self.tracker_node = BallTrackerNode(self.config)

        self.running = True
        self.publish_enabled = False
        self.ball_detected = False

        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

    def calculate_offset_data(self, metric_x: float) -> Tuple[float, float]:
        """
        Compute distance (absolute) and radians direction flag from metric X.
        Since target is at X=0 in camera frame, distance = abs(metric_x).
        Radians = 0.0 for negative X (left), pi for positive X (right).
        """
        distance = abs(metric_x)
        radians = 0.0 if metric_x < 0.0 else math.pi
        return distance, radians

    def terminal_input_thread(self):
        """Thread to read commands from stdin."""
        print("Terminal commands: [Enter] = toggle publish, 's' = pause, 'q' = quit")
        while self.running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline().strip().lower()
                if line == '':
                    self.publish_enabled = not self.publish_enabled
                    state = "ENABLED" if self.publish_enabled else "PAUSED"
                    print(f"Publishing {state}")
                elif line == 'q':
                    print("Quit command received.")
                    self.running = False
                    break
                elif line == 's':
                    self.publish_enabled = False
                    print("Publishing PAUSED")
                else:
                    print(f"Unknown command: '{line}'. Use Enter, 's', or 'q'.")
            else:
                time.sleep(0.05)

    def run(self):
        print("=" * 60)
        print("RED BALL METRIC X PUBLISHER (ROS2) - Terminal Control")
        print("=" * 60)
        print(f"Color topic:  {self.config.COLOR_TOPIC}")
        print(f"Depth topic:  {self.config.DEPTH_TOPIC}")
        print(f"Info topic:   {self.config.CAMERA_INFO_TOPIC}")
        print(f"Output topic: {self.config.OUTPUT_TOPIC}")
        print("Publishes: [abs_x_meters, radians (0.0 left, π right)]")
        print("=" * 60)
        print("TERMINAL COMMANDS:")
        print("  [Enter]       : toggle publishing on/off")
        print("  's' + Enter   : pause publishing")
        print("  'q' + Enter   : quit")
        print("OpenCV window also accepts 'q' (quit) and SPACE (pause).")
        print("=" * 60)

        input_thread = threading.Thread(target=self.terminal_input_thread, daemon=True)
        input_thread.start()

        last_print_time = time.time()

        try:
            while self.running and rclpy.ok():
                rclpy.spin_once(self.tracker_node, timeout_sec=0.01)
                frame_start_time = time.time()

                # Get synchronized color, depth, and camera info
                color, depth, info = self.tracker_node.get_synchronized_data(timeout=0.1)
                if color is None:
                    time.sleep(0.01)
                    continue

                ball_x, ball_y, radius, metric_x, annotated_frame = self.tracker_node.process_frame(color, depth, info)
                self.frame_count += 1

                if ball_x is not None and metric_x is not None:
                    self.detection_count += 1
                    self.ball_detected = True
                    distance, radians = self.calculate_offset_data(metric_x)

                    if self.publish_enabled:
                        msg = Float32MultiArray()
                        msg.data = [distance, radians]
                        self.tracker_node.publisher.publish(msg)

                    current_time = time.time()
                    if current_time - last_print_time > 1.0 and self.config.PRINT_STATS:
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        detect_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                        mode_str = "PUBLISH" if self.publish_enabled else "PAUSE"
                        side_str = "LEFT" if metric_x < 0.0 else "RIGHT"
                        print(f"[{mode_str}] FPS: {fps:.1f}, X: {metric_x:.3f} m ({side_str}), "
                              f"Dist: {distance:.3f}, Rad: {radians:.2f}, Detect: {detect_rate:.1f}%")
                        last_print_time = current_time
                else:
                    self.ball_detected = False

                # OpenCV window
                if self.config.SHOW_VIDEO and annotated_frame is not None:
                    mode_text = "MODE: PUBLISH" if self.publish_enabled else "MODE: PAUSE"
                    mode_color = (0, 255, 0) if self.publish_enabled else (0, 0, 255)
                    cv2.putText(annotated_frame, mode_text, (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                    status = "BALL TRACKING" if self.ball_detected else "SEARCHING"
                    status_color = (0, 255, 0) if self.ball_detected else (0, 0, 255)
                    cv2.putText(annotated_frame, f"Status: {status}", (10, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    frame_time = time.time() - frame_start_time
                    fps_disp = 1.0 / frame_time if frame_time > 0 else 0
                    cv2.putText(annotated_frame, f"FPS: {fps_disp:.1f}", (10, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Ball Tracker - Metric Offset", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                    elif key == 32:  # Space
                        self.publish_enabled = False
                        print("Publishing PAUSED (via OpenCV window)")

                # Maintain frame rate
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                target_time = 1.0 / self.config.FPS
                if processing_time < target_time:
                    time.sleep(target_time - processing_time)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
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
    system.run()

if __name__ == "__main__":
    main()
