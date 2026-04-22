#!/usr/bin/env python3
"""
Red Ball Tracking and Horizontal Strafing System with PID Control
For Raspberry Pi 5 with Intel RealSense D415 and ROS2 Jazzy
Uses pyrealsense2 for camera, pyserial for motor commands to Arduino
Omnidirectional (Mecanum) wheels – strafing only (horizontal movement)
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

# Try to import RealSense
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
    KP_X = 0.8      # Proportional gain
    KI_X = 0.05     # Integral gain
    KD_X = 0.15     # Derivative gain
    
    # Motor control (serial)
    SERIAL_PORT = '/dev/ttyUSB0'  # Adjust to your Arduino port
    SERIAL_BAUD = 115200
    
    # Motor limits
    MAX_MOTOR_SPEED = 100   # Maximum PWM duty cycle (0-100%)
    MIN_MOTOR_SPEED = 20    # Minimum speed to overcome friction
    
    # System parameters
    TARGET_X = CAMERA_WIDTH // 2  # Target horizontal position (center)
    
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
        """Calculate PID output based on measurement"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 0.01
        
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            d_term = self.kd * (error - self.previous_error) / dt
        else:
            d_term = 0
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Update state
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
        """
        Set speeds for four motors.
        fl: front left, fr: front right, rr: rear right, rl: rear left
        Values are in range [-100, 100]
        """
        # Clamp to limits
        fl = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, fl))
        fr = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, fr))
        rr = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, rr))
        rl = max(-self.config.MAX_MOTOR_SPEED, min(self.config.MAX_MOTOR_SPEED, rl))
        
        # Format as comma-separated string
        cmd_str = f"{fl:.0f},{fr:.0f},{rr:.0f},{rl:.0f}\n"
        
        if self.initialized and self.serial_port:
            try:
                self.serial_port.write(cmd_str.encode())
                self.serial_port.flush()
            except Exception as e:
                print(f"Serial write error: {e}")
        else:
            # Simulation mode: print to console
            print(f"Motors: FL={fl:.0f}, FR={fr:.0f}, RR={rr:.0f}, RL={rl:.0f}")
    
    def stop_all(self):
        """Stop all motors"""
        self.set_motor_speeds(0, 0, 0, 0)
    
    def cleanup(self):
        """Close serial port"""
        if self.initialized and self.serial_port:
            self.serial_port.close()
            print("Serial port closed")

class BallTracker:
    """Tracks red ball using computer vision with RealSense D415"""
    
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.align = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Position tracking
        self.positions = deque(maxlen=config.SMOOTHING_WINDOW)
        self.velocities = deque(maxlen=config.PREDICTION_WINDOW)
        
        # Initialize camera
        self._init_camera()
    
    def _init_camera(self):
        """Initialize RealSense D415 or fallback to OpenCV"""
        if REALSENSE_AVAILABLE:
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 
                                     self.config.CAMERA_WIDTH, 
                                     self.config.CAMERA_HEIGHT, 
                                     rs.format.bgr8, 
                                     self.config.FPS)
                
                # Start streaming
                profile = self.pipeline.start(config)
                
                # Create align object to align depth to color (not used but available)
                self.align = rs.align(rs.stream.color)
                
                print("RealSense D415 initialized successfully")
                return
            except Exception as e:
                print(f"Error initializing RealSense: {e}")
                self.pipeline = None
        
        # Fallback to OpenCV camera
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
        """Get a frame from the camera"""
        if REALSENSE_AVAILABLE and self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    return frame
            except Exception as e:
                print(f"RealSense frame error: {e}")
                return None
        elif hasattr(self, 'camera') and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
    
    def detect_ball(self, frame):
        """Detect red ball in the frame"""
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
        """Predict future position based on recent positions"""
        self.positions.append(position)
        
        if len(self.positions) < 2:
            return position[0], position[1]  # Return (x, y)
        
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
            
            # Predict position 0.1 seconds ahead
            predict_time = 0.1
            pred_x = position[0] + avg_vx * predict_time
            pred_y = position[1] + avg_vy * predict_time
            
            # Constrain to frame bounds
            pred_x = int(max(0, min(self.config.CAMERA_WIDTH - 1, pred_x)))
            pred_y = int(max(0, min(self.config.CAMERA_HEIGHT - 1, pred_y)))
            
            return pred_x, pred_y
        
        return position[0], position[1]
    
    def draw_detection(self, frame, ball_data, pred_position):
        """Draw detection results on frame"""
        if frame is None:
            return frame
        
        if ball_data is not None:
            x, y, radius = ball_data
            pred_x, pred_y = pred_position
            
            # Draw detected ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Draw predicted position
            cv2.circle(frame, (int(pred_x), int(pred_y)), 10, (255, 255, 0), 2)
        
        # Draw target horizontal line
        cv2.line(frame, (int(self.config.TARGET_X), 0), 
                 (int(self.config.TARGET_X), self.config.CAMERA_HEIGHT), 
                 (255, 0, 0), 2)
        
        return frame
    
    def draw_text(self, frame, ball_data, pred_position):
        """Draw text information on frame"""
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
        
        # Draw target information
        cv2.putText(frame, f"Target X: {self.config.TARGET_X}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run_tracking(self):
        """Main tracking loop - returns ball position and frame"""
        self.running = True
        
        while self.running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            with self.lock:
                self.frame = frame
            
            ball_data, mask = self.detect_ball(frame)
            
            if ball_data is not None:
                x, y, radius = ball_data
                
                # Predict position (only X is used for control, Y is kept for visualization)
                pred_x, pred_y = self.predict_position((x, y, radius))
                
                if self.config.SHOW_VIDEO:
                    frame = self.draw_detection(frame, ball_data, (pred_x, pred_y))
                    frame = self.draw_text(frame, ball_data, (pred_x, pred_y))
                    
                    if mask is not None:
                        mask_resized = cv2.resize(mask, (160, 120))
                        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                        frame[0:120, 0:160] = mask_colored
                
                return pred_x, pred_y, radius, frame
            
            # No ball detected
            if self.config.SHOW_VIDEO and frame is not None:
                frame = self.draw_text(frame, None, (0, 0))
                cv2.line(frame, (self.config.TARGET_X, 0), 
                         (self.config.TARGET_X, self.config.CAMERA_HEIGHT), 
                         (255, 0, 0), 2)
            
            return None, None, None, frame
    
    def stop(self):
        """Stop tracking"""
        self.running = False
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()

class BallCatchingSystem:
    """Main system for horizontal strafing to catch balls"""
    
    def __init__(self):
        self.config = Config()
        self.tracker = BallTracker(self.config)
        self.motor_controller = MotorController(self.config)
        
        # Initialize PID controller for X-axis only
        self.pid_x = PIDController(
            kp=self.config.KP_X,
            ki=self.config.KI_X,
            kd=self.config.KD_X,
            setpoint=self.config.TARGET_X,
            output_limits=(-self.config.MAX_MOTOR_SPEED, self.config.MAX_MOTOR_SPEED)
        )
        
        # State variables
        self.running = False
        self.ball_detected = False
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
    
    def calculate_motor_speeds(self, ball_x: float) -> Tuple[float, float, float, float]:
        """
        Calculate motor speeds for four Mecanum wheels to strafe horizontally.
        All wheels rotate in the same direction for pure translation left/right.
        Positive speed = strafe right, negative = strafe left.
        Returns (front_left, front_right, rear_right, rear_left)
        """
        # Get PID output for X correction
        speed = self.pid_x.calculate(ball_x)
        
        # Apply minimum speed if moving
        if abs(speed) > 0 and abs(speed) < self.config.MIN_MOTOR_SPEED:
            speed = self.config.MIN_MOTOR_SPEED if speed > 0 else -self.config.MIN_MOTOR_SPEED
        
        # For Mecanum wheels with rollers at 45° (front-left & rear-right) and
        # -45° (front-right & rear-left), all wheels turning same direction
        # produces pure strafing.
        fl = speed
        fr = speed
        rr = speed
        rl = speed
        
        return fl, fr, rr, rl
    
    def run(self):
        """Main system loop"""
        self.running = True
        print("Starting horizontal ball tracking system...")
        print("Press 'q' to quit, 'r' to reset PID, 's' to stop motors")
        
        last_print_time = time.time()
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # Track ball
                ball_x, ball_y, radius, frame = self.tracker.run_tracking()
                
                self.frame_count += 1
                
                if ball_x is not None:
                    self.detection_count += 1
                    self.ball_detected = True
                    
                    # Calculate motor speeds (strafe only)
                    fl, fr, rr, rl = self.calculate_motor_speeds(float(ball_x))
                    
                    # Send to motors
                    self.motor_controller.set_motor_speeds(fl, fr, rr, rl)
                    
                    # Print statistics occasionally
                    current_time = time.time()
                    if current_time - last_print_time > 1.0 and self.config.PRINT_STATS:
                        elapsed_time = current_time - self.start_time
                        fps = self.frame_count / elapsed_time
                        detection_rate = (self.detection_count / self.frame_count) * 100
                        print(f"FPS: {fps:.1f}, Ball X: {int(ball_x)}, "
                              f"Speed: {fl:.0f}, Detect: {detection_rate:.1f}%")
                        last_print_time = current_time
                else:
                    self.ball_detected = False
                    self.motor_controller.stop_all()
                
                # Display video
                if self.config.SHOW_VIDEO and frame is not None:
                    status = "TRACKING" if self.ball_detected else "SEARCHING"
                    color = (0, 255, 0) if self.ball_detected else (0, 0, 255)
                    cv2.putText(frame, f"Status: {status}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # FPS
                    frame_time = time.time() - frame_start_time
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("Ball Tracker (Horizontal Strafing)", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.pid_x.reset()
                        print("PID controller reset")
                    elif key == ord('s'):
                        self.motor_controller.stop_all()
                        print("Motors stopped")
                    elif key == ord(' '):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"snapshot_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Snapshot saved as {filename}")
                
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
        """Cleanup resources"""
        print("\nCleaning up...")
        self.running = False
        self.tracker.stop()
        self.motor_controller.stop_all()
        self.motor_controller.cleanup()
        cv2.destroyAllWindows()
        
        if self.config.PRINT_STATS:
            total_time = time.time() - self.start_time
            print(f"\nFinal Statistics:")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Total frames: {self.frame_count}")
            print(f"Total detections: {self.detection_count}")
            if total_time > 0:
                print(f"Average FPS: {self.frame_count/total_time:.1f}")
            if self.frame_count > 0:
                detection_rate = (self.detection_count / self.frame_count) * 100
                print(f"Detection rate: {detection_rate:.1f}%")

def main():
    """Main entry point"""
    system = BallCatchingSystem()
    
    print("=" * 60)
    print("RED BALL HORIZONTAL STRAFING SYSTEM")
    print("RealSense D415 + Mecanum Wheels")
    print("=" * 60)
    print(f"Camera: {system.config.CAMERA_WIDTH}x{system.config.CAMERA_HEIGHT} @ {system.config.FPS}FPS")
    print(f"Target X: {system.config.TARGET_X}")
    print(f"PID Gains: KP={system.config.KP_X}, KI={system.config.KI_X}, KD={system.config.KD_X}")
    print(f"Serial: {system.config.SERIAL_PORT} @ {system.config.SERIAL_BAUD} baud")
    print("=" * 60)
    print("Controls: 'q'=quit, 'r'=reset PID, 's'=stop motors, SPACE=snapshot")
    print("=" * 60)
    
    system.run()

if __name__ == "__main__":
    main()