#!/usr/bin/env python3
"""
YOLO Object Detection Based Horizontal Strafing System with PID Control
For Raspberry Pi 5 with Intel RealSense D415 and ROS2 Jazzy
Uses Ultralytics YOLO for object detection, pyserial for motor commands to Arduino
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

# Ultralytics YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Please install: pip install ultralytics")

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available, using OpenCV fallback")

# Configuration
@dataclass
class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS = 30

    # YOLO model settings
    # CHANGE THIS PATH TO YOUR TRAINED MODEL FILE
    YOLO_MODEL_PATH = "path/to/your/model.pt"  # <--- PUT YOUR .pt FILE NAME HERE
    CONFIDENCE_THRESHOLD = 0.4                 # Minimum confidence to consider a detection

    # PID Controller gains (X-axis only)
    KP_X = 0.8
    KI_X = 0.05
    KD_X = 0.15

    # Motor control (serial)
    SERIAL_PORT = '/dev/ttyACM0'
    SERIAL_BAUD = 115200

    # Motor limits
    MAX_MOTOR_SPEED = 100
    MIN_MOTOR_SPEED = 20

    # System parameters
    TARGET_X = CAMERA_WIDTH // 2               # Center of frame

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

class YOLOTracker:
    """Tracks objects using YOLO model (e.g., red/purple balls)"""

    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.align = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

        # Load YOLO model
        self.model = None
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(config.YOLO_MODEL_PATH)
                print(f"YOLO model loaded from {config.YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = None
        else:
            print("YOLO not available. Detection disabled.")

        # Position tracking
        self.positions = deque(maxlen=config.SMOOTHING_WINDOW)
        self.velocities = deque(maxlen=config.PREDICTION_WINDOW)

        self._init_camera()

    def _init_camera(self):
        if REALSENSE_AVAILABLE:
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color,
                                     self.config.CAMERA_WIDTH,
                                     self.config.CAMERA_HEIGHT,
                                     rs.format.bgr8,
                                     self.config.FPS)
                self.pipeline.start(config)
                self.align = rs.align(rs.stream.color)
                print("RealSense D415 initialized successfully")
                return
            except Exception as e:
                print(f"Error initializing RealSense: {e}")
                self.pipeline = None

        print("Using OpenCV camera fallback")
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.FPS)
            print("OpenCV camera initialized")
        else:
            print("Error: Could not open any camera")

    def get_frame(self):
        if REALSENSE_AVAILABLE and self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    return np.asanyarray(color_frame.get_data())
            except Exception as e:
                print(f"RealSense frame error: {e}")
                return None
        elif hasattr(self, 'camera') and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None

    def detect_object(self, frame):
        """
        Run YOLO inference and return centroid of highest confidence detection.
        Returns (x, y, confidence, bbox) or None if no detection meets threshold.
        """
        if frame is None or self.model is None:
            return None

        # Run inference
        results = self.model(frame, verbose=False)

        if not results or len(results) == 0:
            return None

        # Get detections from first result
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        # Find detection with highest confidence
        best_conf = -1
        best_box = None
        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_conf and conf >= self.config.CONFIDENCE_THRESHOLD:
                best_conf = conf
                best_box = box

        if best_box is None:
            return None

        # Get bounding box coordinates (xyxy format)
        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy

        # Calculate centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Bounding box width and height (for info)
        w = int(x2 - x1)
        h = int(y2 - y1)

        return (cx, cy, best_conf, (x1, y1, x2, y2))

    def predict_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Predict future centroid position based on recent positions."""
        self.positions.append(position)

        if len(self.positions) < 2:
            return position

        # Calculate velocity
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

        return position

    def draw_detection(self, frame, detection_data, pred_position):
        if frame is None:
            return frame

        if detection_data is not None:
            cx, cy, conf, bbox = detection_data
            x1, y1, x2, y2 = bbox
            pred_x, pred_y = pred_position

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw centroid
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            # Draw confidence
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw predicted position
            cv2.circle(frame, (int(pred_x), int(pred_y)), 10, (255, 255, 0), 2)

        # Draw target vertical line
        cv2.line(frame, (self.config.TARGET_X, 0),
                 (self.config.TARGET_X, self.config.CAMERA_HEIGHT),
                 (255, 0, 0), 2)

        return frame

    def draw_text(self, frame, detection_data, pred_position):
        if frame is None:
            return frame

        if detection_data is not None:
            cx, cy, conf, _ = detection_data
            pred_x, pred_y = pred_position
            cv2.putText(frame, f"Object: ({cx}, {cy}) conf={conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pred: ({pred_x}, {pred_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Target X: {self.config.TARGET_X}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def run_tracking(self):
        """Main tracking loop - returns object centroid and annotated frame."""
        self.running = True

        while self.running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame.copy()

            detection = self.detect_object(frame)
            if detection is not None:
                cx, cy, conf, bbox = detection
                pred_x, pred_y = self.predict_position((cx, cy))

                if self.config.SHOW_VIDEO:
                    frame = self.draw_detection(frame, detection, (pred_x, pred_y))
                    frame = self.draw_text(frame, detection, (pred_x, pred_y))

                return pred_x, pred_y, conf, frame
            else:
                if self.config.SHOW_VIDEO:
                    frame = self.draw_text(frame, None, (0, 0))
                    cv2.line(frame, (self.config.TARGET_X, 0),
                             (self.config.TARGET_X, self.config.CAMERA_HEIGHT),
                             (255, 0, 0), 2)
                return None, None, None, frame

    def stop(self):
        self.running = False
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()

class StrafingSystem:
    """Main system for horizontal strafing using YOLO detections"""

    def __init__(self):
        self.config = Config()
        self.tracker = YOLOTracker(self.config)
        self.motor_controller = MotorController(self.config)

        self.pid_x = PIDController(
            kp=self.config.KP_X,
            ki=self.config.KI_X,
            kd=self.config.KD_X,
            setpoint=self.config.TARGET_X,
            output_limits=(-self.config.MAX_MOTOR_SPEED, self.config.MAX_MOTOR_SPEED)
        )

        self.running = True           # Main loop control
        self.movement_enabled = False # Pause/run mode (start paused)
        self.object_detected = False

        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

    def calculate_motor_speeds(self, object_x: float) -> Tuple[float, float, float, float]:
        speed = self.pid_x.calculate(object_x)
        if abs(speed) > 0 and abs(speed) < self.config.MIN_MOTOR_SPEED:
            speed = self.config.MIN_MOTOR_SPEED if speed > 0 else -self.config.MIN_MOTOR_SPEED
        return speed, speed, speed, speed

    def run(self):
        print("Starting YOLO-based horizontal strafing system...")
        print("Controls:")
        print("  ENTER : Start movement (run mode)")
        print("  SPACE : Pause movement")
        print("  'q'   : Quit")
        print("  'r'   : Reset PID")
        print("  's'   : Stop motors (also pauses)")
        print("System starts in PAUSE mode. Press ENTER to begin moving.")

        last_print_time = time.time()

        try:
            while self.running:
                frame_start_time = time.time()

                # Run detection
                obj_x, obj_y, conf, frame = self.tracker.run_tracking()
                self.frame_count += 1

                if obj_x is not None:
                    self.detection_count += 1
                    self.object_detected = True

                    if self.movement_enabled:
                        fl, fr, rr, rl = self.calculate_motor_speeds(float(obj_x))
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
                        print(f"[{mode_str}] FPS: {fps:.1f}, Obj X: {int(obj_x)}, "
                              f"Conf: {conf:.2f}, Detect: {detect_rate:.1f}%")
                        last_print_time = current_time
                else:
                    self.object_detected = False
                    self.motor_controller.stop_all()

                # Display
                if self.config.SHOW_VIDEO and frame is not None:
                    # Show mode status
                    mode_text = "MODE: RUN" if self.movement_enabled else "MODE: PAUSE (Press ENTER to start)"
                    mode_color = (0, 255, 0) if self.movement_enabled else (0, 0, 255)
                    cv2.putText(frame, mode_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                    # FPS
                    frame_time = time.time() - frame_start_time
                    fps_disp = 1.0 / frame_time if frame_time > 0 else 0
                    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("YOLO Tracker - Horizontal Strafing", frame)

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

                # Maintain frame rate
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
        self.tracker.stop()
        self.motor_controller.stop_all()
        self.motor_controller.cleanup()
        cv2.destroyAllWindows()

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
    system = StrafingSystem()

    print("=" * 60)
    print("YOLO OBJECT DETECTION STRAFING SYSTEM")
    print("RealSense D415 + Mecanum Wheels")
    print("=" * 60)
    print(f"Model: {system.config.YOLO_MODEL_PATH}")
    print(f"Confidence threshold: {system.config.CONFIDENCE_THRESHOLD}")
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
