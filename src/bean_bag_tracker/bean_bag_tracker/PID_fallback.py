#!/usr/bin/env python3
"""
<<<<<<< HEAD
Red Ball Tracking and Metric X Distance Publisher (ROS2)
Uses RealSense depth + intrinsics to compute horizontal offset in meters.
Publishes [x_distance_meters, radians] to 'bean_bag_tracker'.

Terminal: h=help, d=debug (publish interval & frame→publish ms), p=one [PUB] line per message.
=======
Red Ball Tracking and Horizontal Offset Publisher (ROS2)
Terminal-controlled publishing: Enter toggles run/pause, 'q' quits, 's' pauses.
>>>>>>> 186951d (Revert "Refactor PID_fallback.py for metric X distance tracking")
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
import statistics
from typing import Deque, Optional, Tuple

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
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    FPS = 30

    # ROS2 topic for RealSense color image
    CAMERA_TOPIC = '/camera/camera/color/image_raw'

    # ROS2 topic for publishing offset data
    OUTPUT_TOPIC = '/bean_bag_trajectory'

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
    """ROS2 Node that subscribes to camera and publishes offset data."""

    def __init__(self, config):
        super().__init__('ball_tracker_node')
        self.config = config
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

        self.subscription = self.create_subscription(
            Image, config.CAMERA_TOPIC, self.image_callback, 10)
        self.get_logger().info(f"Subscribed to {config.CAMERA_TOPIC}")

        self.publisher = self.create_publisher(Float32MultiArray, config.OUTPUT_TOPIC, 10)
        self.get_logger().info(f"Publishing to {config.OUTPUT_TOPIC}")

        self.positions = deque(maxlen=config.SMOOTHING_WINDOW)
        self.velocities = deque(maxlen=config.PREDICTION_WINDOW)
        self.running = True

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
            self.new_frame_event.set()
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def get_frame(self, timeout=1.0):
        if self.new_frame_event.wait(timeout=timeout):
            self.new_frame_event.clear()
            with self.frame_lock:
                return self.latest_frame.copy() if self.latest_frame is not None else None
        return None

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

    def draw_detection(self, frame, ball_data, pred_position):
        if frame is None or ball_data is None:
            return frame
        x, y, radius = ball_data
        pred_x, pred_y = pred_position
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (pred_x, pred_y), 10, (255, 255, 0), 2)
        cv2.line(frame, (self.config.TARGET_X, 0),
                 (self.config.TARGET_X, self.config.CAMERA_HEIGHT), (255, 0, 0), 2)
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
                         (self.config.TARGET_X, self.config.CAMERA_HEIGHT), (255, 0, 0), 2)

        return pred_x, pred_y, radius, annotated_frame

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

        self._stats_lock = threading.Lock()
        self._last_publish_mono: Optional[float] = None
        self._inter_publish_ms: Deque[float] = deque(maxlen=30)
        self._last_frame_to_pub_ms: float = 0.0
        self._per_publish_log: bool = False

    def calculate_offset_data(self, metric_x: float) -> Tuple[float, float]:
        """
        Compute distance (absolute) and radians direction flag from metric X.
        Since target is at X=0 in camera frame, distance = abs(metric_x).
        Radians = 0.0 for negative X (left), pi for positive X (right).
        """
        distance = abs(metric_x)
        radians = 0.0 if metric_x < 0.0 else math.pi
        return distance, radians

    def _print_terminal_help(self) -> None:
        print("Commands: [Enter]=toggle publish  s=pause  q=quit  h=help  d=debug  p=per-pub line")

    def _print_debug_snapshot(self) -> None:
        now = time.monotonic()
        with self._stats_lock:
            last_m = self._last_publish_mono
            dts = list(self._inter_publish_ms)
            f2p = self._last_frame_to_pub_ms
            ppl = self._per_publish_log
        age_ms = (now - last_m) * 1000.0 if last_m is not None else None
        n = len(dts)
        if n > 0:
            mn, mx, avg = min(dts), max(dts), statistics.mean(dts)
            inter = f"n={n} min_ms={mn:.2f} max_ms={mx:.2f} mean_ms={avg:.2f}"
        else:
            inter = "n=0 (no inter-publish samples yet)"
        det = self.ball_detected
        pe = self.publish_enabled
        print("--- [DEBUG] publish pipeline ---")
        print(f"  publish_enabled={pe!r}  ball_detected={det!r}  per_pub_log={ppl!r}")
        if last_m is not None:
            print(f"  last_publish_mono_s={last_m:.6f}  age_ms={age_ms:.2f}")
        else:
            print("  last publish: (never while stats collected)")
        print(f"  inter_publish_ms: {inter}")
        print(f"  last frame_start→publish_ms: {f2p:.2f} (most recent loop with a publish)")

    def terminal_input_thread(self):
        """Thread to read commands from stdin."""
        print("Terminal: Enter=toggle  s=pause  h=help  d=debug  p=per-pub  q=quit")
        while self.running:
            # Check if there's input available (non-blocking)
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
                elif line in ("h", "?"):
                    self._print_terminal_help()
                elif line == "d":
                    self._print_debug_snapshot()
                elif line == "p":
                    with self._stats_lock:
                        self._per_publish_log = not self._per_publish_log
                        on = self._per_publish_log
                    print(f"Per-publish one-liner: {'ON' if on else 'OFF'}")
                else:
                    print(
                        f"Unknown command: '{line}'. Use Enter, s, q, h, d, p "
                        f"(or run 'h' for help)."
                    )
            else:
                time.sleep(0.05)

    def run(self):
        print("=" * 60)
        print("RED BALL OFFSET PUBLISHER (ROS2) - Terminal Control")
        print("=" * 60)
        print(f"Input topic:  {self.config.CAMERA_TOPIC}")
        print(f"Output topic: {self.config.OUTPUT_TOPIC}")
        print(f"Camera: {self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT} @ {self.config.FPS}FPS")
        print("Publishes: [distance_from_center (px), radians (0.0 left, π right)]")
        print("=" * 60)
        print("TERMINAL COMMANDS:")
        print("  [Enter]       : toggle publishing on/off")
        print("  's' + Enter   : pause publishing")
        print("  'h' or '?'    : help (short list)")
        print("  'd' + Enter   : debug snapshot (publish interval & frame→publish ms)")
        print("  'p' + Enter   : toggle one line per [PUB] on each publish")
        print("  'q' + Enter   : quit")
        print("OpenCV window also accepts 'q' (quit) and SPACE (pause).")
        print("=" * 60)

        # Start terminal input thread
        input_thread = threading.Thread(target=self.terminal_input_thread, daemon=True)
        input_thread.start()

        last_print_time = time.time()

        try:
            while self.running and rclpy.ok():
                rclpy.spin_once(self.tracker_node, timeout_sec=0.1)
                frame_start_mono = time.monotonic()
                frame_start_time = time.time()

                frame = self.tracker_node.get_frame(timeout=0.1)
                if frame is None:
                    time.sleep(0.01)
                    continue

                ball_x, ball_y, radius, annotated_frame = self.tracker_node.process_frame(frame)
                self.frame_count += 1

                if ball_x is not None:
                    self.detection_count += 1
                    self.ball_detected = True
                    distance, radians = self.calculate_offset_data(float(ball_x))

                    if self.publish_enabled:
                        t_pub = time.monotonic()
                        f2p_ms = (t_pub - frame_start_mono) * 1000.0
                        inter_ms = 0.0
                        with self._stats_lock:
                            if self._last_publish_mono is not None:
                                inter_ms = (t_pub - self._last_publish_mono) * 1000.0
                                self._inter_publish_ms.append(inter_ms)
                            self._last_publish_mono = t_pub
                            self._last_frame_to_pub_ms = f2p_ms
                            do_log = self._per_publish_log
                        msg = Float32MultiArray()
                        msg.data = [distance, radians]
                        self.tracker_node.publisher.publish(msg)
                        if do_log:
                            print(
                                f"[PUB] mono_s={t_pub:.6f} inter_ms={inter_ms:.2f} "
                                f"f2p_ms={f2p_ms:.2f} dist={distance:.3f} rad={radians:.2f}"
                            )

                    current_time = time.time()
                    if current_time - last_print_time > 1.0 and self.config.PRINT_STATS:
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        detect_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                        mode_str = "PUBLISH" if self.publish_enabled else "PAUSE"
                        side_str = "LEFT" if ball_x < self.config.TARGET_X else "RIGHT"
                        print(f"[{mode_str}] FPS: {fps:.1f}, X: {int(ball_x)} ({side_str}), "
                              f"Dist: {distance:.1f}, Rad: {radians:.2f}, Detect: {detect_rate:.1f}%")
                        last_print_time = current_time
                else:
                    self.ball_detected = False

                # OpenCV window handling
                if self.config.SHOW_VIDEO and annotated_frame is not None:
                    mode_text = "MODE: PUBLISH" if self.publish_enabled else "MODE: PAUSE"
                    mode_color = (0, 255, 0) if self.publish_enabled else (0, 0, 255)
                    cv2.putText(annotated_frame, mode_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                    status = "BALL TRACKING" if self.ball_detected else "SEARCHING"
                    status_color = (0, 255, 0) if self.ball_detected else (0, 0, 255)
                    cv2.putText(annotated_frame, f"Status: {status}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    if self.ball_detected and ball_x is not None:
                        distance, radians = self.calculate_offset_data(float(ball_x))
                        cv2.putText(annotated_frame, f"Dist: {distance:.1f} px, Rad: {radians:.2f}", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    frame_time = time.time() - frame_start_time
                    fps_disp = 1.0 / frame_time if frame_time > 0 else 0
                    cv2.putText(annotated_frame, f"FPS: {fps_disp:.1f}", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Ball Tracker - Offset Publisher", annotated_frame)

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
