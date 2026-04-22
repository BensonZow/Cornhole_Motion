#!/usr/bin/env python3
"""
Red Ball Tracking and Horizontal Strafing System with PID Control (ROS2 Subscriber)
For Raspberry Pi 5 with ROS2 Jazzy and Intel RealSense D415
Subscribes to existing RealSense ROS2 node for camera feed.
Uses color detection (red/purple) and serial motor commands.
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional, Tuple
import serial

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Configuration
@dataclass
class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS = 30

    # ROS2 topic for RealSense color image
    CAMERA_TOPIC = '/camera/color/image_raw'  # Adjust to your actual topic

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
    MAX_RADIUS = 100
    MIN_AREA = 100

    # PID Controller gains (X-axis only)
    KP_X = 0.8
    KI_X = 0.05
    KD_X = 0.15

    # Motor control (serial)
    SERIAL_PORT = '/dev/ttyACM1'
    SERIAL_BAUD = 115200

    # Motor limits
    MAX_MOTOR_SPEED = 100
    MIN_MOTOR_SPEED = 20

    # System parameters
    TARGET_X = CAMERA_WIDTH // 2

    # Prediction settings
    PREDICTION_WINDOW = 5
    SMOOTHING_WINDOW = 3

    # Debug settings
    SHOW_VIDEO = True
    PRINT_STATS = True

class PIDController:
    """PID Controller for motor position control"""

    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.reset()

    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()

    def calculate(self, measurement: float, dt: Optional[float] = None) -> float:
        current_time = time.time()
        if dt is None:
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 0.01

        error = self.setpoint - measurement
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        if dt > 0:
            d_term = self.kd * (error - self.previous_error) / dt
        else:
            d_term = 0

        output = p_term + i_term + d_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self.previous_error = error
        self.previous_time = current_time
        return output

class MotorController:
    """Motor Controller via Serial to Arduino"""

    def __init__(self, config):
        self.config = config
        self.serial_port = None
        self.initialized = False

        try:
            self.serial_port = serial.Serial(
                port=self.config.SERIAL_PORT,
                baudrate=self.config.SERIAL_BAUD,
                timeout=1
            )
            self.initialized = True
            print(f"Serial port {self.config.SERIAL_PORT} opened at {self.config.SERIAL_BAUD} baud")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            print("Motor controller in simulation mode (serial output to console)")
            self.initialized = False

    def set_motor_speeds(self, fl: float, fr: float, rr: float, rl: float):
        fl = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, fl))
        fr = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, fr))
        rr = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, rr))
        rl = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, rl))

        cmd_str = f"{fl:.0f},{fr:.0f},{rr:.0f},{rl:.0f}\n"

        if self.initialized and self.serial_port:
            try:
                self.serial_port.write(cmd_str.encode())
                self.serial_port.flush()
            except Exception as e:
                print(f"Serial write error: {e}")
        else:
            print(f"Motors: FL={fl:.0f}, FR={fr:.0f}, RR={rr:.0f}, RL={rl:.0f}")

    def stop_all(self):
        self.set_motor_speeds(0, 0, 0, 0)

    def cleanup(self):
        if self.initialized and self.serial_port:
            self.serial_port.close()
            print("Serial port closed")

class BallTrackerNode(Node):
    """
    ROS2 Node that subscribes to RealSense camera topic and performs color-based ball tracking.
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
        """Process a single frame and return detection result."""
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

class BallCatchingSystem:
    """Main system for horizontal strafing using color detection from ROS2."""

    def __init__(self):
        self.config = Config()

        # Initialize ROS2
        rclpy.init()
        self.tracker_node = BallTrackerNode(self.config)

        self.motor_controller = MotorController(self.config)

        self.pid_x = PIDController(
            kp=self.config.KP_X,
            ki=self.config.KI_X,
            kd=self.config.KD_X,
            setpoint=self.config.TARGET_X,
            output_limits=(-self.config.MAX_MOTOR_SPEED, self.config.MAX_MOTOR_SPEED)
        )

        self.running = True
        self.movement_enabled = False  # Start in pause mode
        self.ball_detected = False

        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

    def calculate_motor_speeds(self, ball_x: float) -> Tuple[float, float, float, float]:
        speed = self.pid_x.calculate(ball_x)
        if abs(speed) > 0 and abs(speed) < self.config.MIN_MOTOR_SPEED:
            speed = self.config.MIN_MOTOR_SPEED if speed > 0 else -self.config.MIN_MOTOR_SPEED
        return speed, speed, speed, speed

    def run(self):
        print("Starting ROS2 color-based horizontal strafing system...")
        print("Controls:")
        print("  ENTER : Start movement (run mode)")
        print("  SPACE : Pause movement")
        print("  'q'   : Quit")
        print("  'r'   : Reset PID")
        print("  's'   : Stop motors (also pauses)")
        print("System starts in PAUSE mode. Press ENTER to begin moving.")

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

                    if self.movement_enabled:
                        fl, fr, rr, rl = self.calculate_motor_speeds(float(ball_x))
                        self.motor_controller.set_motor_speeds(fl, fr, rr, rl)
                    else:
                        self.motor_controller.stop_all()

                    # Stats print
                    current_time = time.time()
                    if current_time - last_print_time > 1.0 and self.config.PRINT_STATS:
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        detect_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                        mode_str = "RUN" if self.movement_enabled else "PAUSE"
                        print(f"[{mode_str}] FPS: {fps:.1f}, Ball X: {int(ball_x)}, "
                              f"Radius: {radius}, Detect: {detect_rate:.1f}%")
                        last_print_time = current_time
                else:
                    self.ball_detected = False
                    self.motor_controller.stop_all()

                # Display
                if self.config.SHOW_VIDEO and annotated_frame is not None:
                    # Show mode status
                    mode_text = "MODE: RUN" if self.movement_enabled else "MODE: PAUSE (Press ENTER to start)"
                    mode_color = (0, 255, 0) if self.movement_enabled else (0, 0, 255)
                    cv2.putText(annotated_frame, mode_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                    # Status text
                    status = "BALL TRACKING" if self.ball_detected else "SEARCHING"
                    status_color = (0, 255, 0) if self.ball_detected else (0, 0, 255)
                    cv2.putText(annotated_frame, f"Status: {status}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    # FPS
                    frame_time = time.time() - frame_start_time
                    fps_disp = 1.0 / frame_time if frame_time > 0 else 0
                    cv2.putText(annotated_frame, f"FPS: {fps_disp:.1f}", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Ball Tracker - Horizontal Strafing (ROS2)", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.pid_x.reset()
                        print("PID controller reset")
                    elif key == ord('s'):
                        self.motor_controller.stop_all()
                        self.movement_enabled = False
                        print("Motors stopped, movement paused")
                    elif key == 13 or key == 10:  # Enter key
                        self.movement_enabled = True
                        print("Movement ENABLED (RUN mode)")
                    elif key == 32:  # Space key
                        self.movement_enabled = False
                        self.motor_controller.stop_all()
                        print("Movement PAUSED")

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
        self.motor_controller.stop_all()
        self.motor_controller.cleanup()
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
    system = BallCatchingSystem()

    print("=" * 60)
    print("RED BALL HORIZONTAL STRAFING SYSTEM (ROS2)")
    print("Subscribing to RealSense camera topic")
    print("=" * 60)
    print(f"ROS2 Topic: {system.config.CAMERA_TOPIC}")
    print(f"Camera: {system.config.CAMERA_WIDTH}x{system.config.CAMERA_HEIGHT} @ {system.config.FPS}FPS")
    print(f"Target X: {system.config.TARGET_X}")
    print(f"PID Gains: KP={system.config.KP_X}, KI={system.config.KI_X}, KD={system.config.KD_X}")
    print(f"Serial: {system.config.SERIAL_PORT} @ {system.config.SERIAL_BAUD} baud")
    print("=" * 60)
    print("System starts in PAUSE mode.")
    print("Press ENTER to start movement, SPACE to pause.")
    print("=" * 60)

    system.run()

if __name__ == "__main__":
    main()
