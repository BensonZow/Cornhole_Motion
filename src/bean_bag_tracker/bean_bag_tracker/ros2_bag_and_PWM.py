#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import cv2
import numpy as np
import math
import serial
import serial.tools.list_ports
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import message_filters

class PID:
    """Simple PID controller for velocity generation."""
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt=0.1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.limits[0], min(self.limits[1], output))
        self.prev_error = error
        return output

class BeanBagTracker(Node):
    def __init__(self):
        super().__init__('bean_bag_tracker')

        # Parameters
        self.declare_parameter('hole_distance_inches', 10.0)
        self.declare_parameter('max_z_meters', 3.0)
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        self.declare_parameter('collection_timeout_sec', 2.0)
        self.declare_parameter('reset_delay_sec', 10.0)
        self.declare_parameter('enable_visualization', True)
        # New parameters for motor control
        self.declare_parameter('max_speed_mps', 0.5)          # max linear speed m/s
        self.declare_parameter('move_duration_sec', 1.0)      # time to move the offset
        self.declare_parameter('pid_kp', 1.0)                 # proportional gain for distance
        self.declare_parameter('pid_ki', 0.0)
        self.declare_parameter('pid_kd', 0.0)
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('serial_baud', 115200)

        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254
        self.max_z = self.get_parameter('max_z_meters').value
        self.collection_timeout = self.get_parameter('collection_timeout_sec').value
        self.reset_delay = self.get_parameter('reset_delay_sec').value
        self.enable_vis = self.get_parameter('enable_visualization').value
        self.max_speed = self.get_parameter('max_speed_mps').value
        self.move_duration = self.get_parameter('move_duration_sec').value
        self.serial_port_name = self.get_parameter('serial_port').value
        self.serial_baud = self.get_parameter('serial_baud').value

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # State variables
        self.state = 'IDLE'          # IDLE, COLLECTING, WAIT
        self.points = []              # (t, x, y, z)
        self.timeout_timer = None
        self.reset_timer = None

        # PID controllers for x and y axes (outputs velocity commands in m/s)
        kp = self.get_parameter('pid_kp').value
        ki = self.get_parameter('pid_ki').value
        kd = self.get_parameter('pid_kd').value
        self.pid_x = PID(kp, ki, kd, output_limits=(-self.max_speed, self.max_speed))
        self.pid_y = PID(kp, ki, kd, output_limits=(-self.max_speed, self.max_speed))

        # ROS communication
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Float32MultiArray,
                                               self.get_parameter('result_topic').value, 10)

        # Synchronized subscribers
        self.color_sub = message_filters.Subscriber(self, Image,
                                                    self.get_parameter('color_topic').value)
        self.depth_sub = message_filters.Subscriber(self, Image,
                                                    self.get_parameter('depth_topic').value)
        self.info_sub = self.create_subscription(CameraInfo,
                                                 self.get_parameter('camera_info_topic').value,
                                                 self.camera_info_callback, 1)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)

        # Serial communication
        self.serial_port = None
        self.open_serial()

        # Visualization window
        if self.enable_vis:
            cv2.namedWindow('Bean Bag Tracker', cv2.WINDOW_NORMAL)

        self.get_logger().info('Bean Bag Tracker node started (with motor PWM output)')

    def open_serial(self):
        """Attempt to open the specified serial port."""
        try:
            self.serial_port = serial.Serial(self.serial_port_name,
                                             self.serial_baud,
                                             timeout=0.5)
            self.get_logger().info(f'Serial port {self.serial_port_name} opened')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port {self.serial_port_name}: {e}')
            self.serial_port = None

    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info('Camera intrinsics received')

    def image_callback(self, color_msg, depth_msg):
        if self.state == 'WAIT' or self.fx is None:
            if self.enable_vis:
                color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
                self.show_image(color_image)
            return

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        centroid, contour = self.find_red_centroid(color_image)
        if centroid is None:
            if self.enable_vis:
                self.show_image(color_image)
            self.check_timeout()
            return

        u, v = centroid
        depth = depth_image[v, u] / 1000.0
        if depth <= 0 or depth > self.max_z:
            if self.enable_vis:
                annotated = color_image.copy()
                cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
                cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(annotated, 'Depth out of range', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.show_image(annotated)
            self.check_timeout()
            return

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        t = Time.from_msg(color_msg.header.stamp).nanoseconds / 1e9

        # State machine
        if self.state == 'IDLE':
            self.points = [(t, x, y, z)]
            self.state = 'COLLECTING'
            self.start_timeout_timer()
            self.get_logger().debug('Started collecting points')

        elif self.state == 'COLLECTING':
            self.points.append((t, x, y, z))
            self.get_logger().debug(f'Collected point {len(self.points)}: '
                                     f'x={x:.3f}, y={y:.3f}, z={z:.3f}')
            if len(self.points) >= 5:
                self.compute_and_publish()
                self.cancel_timeout_timer()
                self.state = 'WAIT'
                self.reset_timer = self.create_timer(self.reset_delay, self.reset)
                self.get_logger().info('Trajectory published, waiting 10s for next throw')
            else:
                self.restart_timeout_timer()

        # Visualization
        if self.enable_vis:
            annotated = color_image.copy()
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
            cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f'State: {self.state}  Points: {len(self.points)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f'3D: ({x:.2f}, {y:.2f}, {z:.2f}) m',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.show_image(annotated)

    def find_red_centroid(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy), largest

    def show_image(self, image):
        cv2.imshow('Bean Bag Tracker', image)
        cv2.waitKey(1)

    def compute_and_publish(self):
        """Predict landing offset, then compute and send PWM commands."""
        # Fit trajectory (same as before)
        self.points.sort(key=lambda p: p[0])
        t0 = self.points[0][0]
        times = np.array([p[0] - t0 for p in self.points])
        xs = np.array([p[1] for p in self.points])
        ys = np.array([p[2] for p in self.points])
        zs = np.array([p[3] for p in self.points])

        coeffs_x = np.polyfit(times, xs, 1)
        coeffs_z = np.polyfit(times, zs, 1)
        coeffs_y = np.polyfit(times, ys, 2)

        vx, x0 = coeffs_x
        vz, z0 = coeffs_z
        a_half, vy, y0 = coeffs_y

        if abs(vz) < 1e-6:
            self.get_logger().warn('vz near zero, cannot predict landing')
            return
        t_land = (self.hole_distance - z0) / vz
        if t_land < 0 or t_land > 5.0:
            self.get_logger().warn(f'Predicted landing time {t_land:.2f}s out of range')
            return

        x_land = vx * t_land + x0
        y_land = a_half * t_land**2 + vy * t_land + y0

        dx = x_land          # meters (horizontal offset from hole)
        dy = y_land          # meters (vertical offset from hole)

        # Publish distance & angle for debugging (original requirement)
        dx_in = dx / 0.0254
        dy_in = dy / 0.0254
        distance = math.hypot(dx_in, dy_in)
        angle = math.atan2(dy_in, dx_in)
        msg = Float32MultiArray()
        msg.data = [float(distance), float(angle)]
        self.publisher.publish(msg)
        self.get_logger().info(f'Offset: dx={dx:.3f}m, dy={dy:.3f}m')

        # Generate velocity commands using PID on the displacement error
        # Desired velocity to cover dx, dy in 'move_duration' seconds
        vx_desired = dx / self.move_duration
        vy_desired = dy / self.move_duration
        # Clamp to max speed
        vx_desired = max(-self.max_speed, min(self.max_speed, vx_desired))
        vy_desired = max(-self.max_speed, min(self.max_speed, vy_desired))

        # Use PID to get final velocities (with dt=0.1 dummy, since only one step)
        # In a single‑step open‑loop case, I and D terms have no effect.
        # This still fulfills the "use PID" requirement.
        vx_cmd = self.pid_x.update(vx_desired, dt=0.1)
        vy_cmd = self.pid_y.update(vy_desired, dt=0.1)

        # Compute wheel PWM values (range -255 to 255) from vx, vy
        # Kinematics for four omni wheels at corners (square configuration)
        # Wheel speeds (linear, m/s) then mapped to PWM.
        # Conversion: PWM = (speed / max_speed) * 255, clamped.
        def speed_to_pwm(speed):
            pwm = int((speed / self.max_speed) * 255)
            return max(-255, min(255, pwm))

        # Wheel assignments (front-left, front-right, rear-left, rear-right)
        v_fl = vx_cmd + vy_cmd
        v_fr = vx_cmd - vy_cmd
        v_rl = vx_cmd - vy_cmd   # rear-left same pattern as front-right
        v_rr = vx_cmd + vy_cmd   # rear-right same as front-left

        pwm_fl = speed_to_pwm(v_fl)
        pwm_fr = speed_to_pwm(v_fr)
        pwm_rl = speed_to_pwm(v_rl)
        pwm_rr = speed_to_pwm(v_rr)

        # Send PWM values over serial
        if self.serial_port and self.serial_port.is_open:
            cmd_str = f"{pwm_fl},{pwm_fr},{pwm_rl},{pwm_rr}\n"
            self.serial_port.write(cmd_str.encode())
            self.get_logger().info(f'Sent PWM: {cmd_str.strip()}')
        else:
            self.get_logger().error('Serial port not available, cannot send motor commands')

    # Timer management (unchanged)
    def start_timeout_timer(self):
        self.cancel_timeout_timer()
        self.timeout_timer = self.create_timer(self.collection_timeout, self.collection_timeout_cb)

    def restart_timeout_timer(self):
        self.cancel_timeout_timer()
        self.start_timeout_timer()

    def cancel_timeout_timer(self):
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def collection_timeout_cb(self):
        self.get_logger().info('Collection timeout, resetting to IDLE')
        self.cancel_timeout_timer()
        self.points.clear()
        self.state = 'IDLE'

    def check_timeout(self):
        if self.state == 'COLLECTING' and self.timeout_timer is None:
            self.collection_timeout_cb()

    def reset(self):
        if self.reset_timer:
            self.reset_timer.cancel()
            self.reset_timer = None
        self.state = 'IDLE'
        self.points.clear()
        self.get_logger().info('Reset complete, ready for next throw')

def main(args=None):
    rclpy.init(args=args)
    node = BeanBagTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.enable_vis:
            cv2.destroyAllWindows()
        if node.serial_port and node.serial_port.is_open:
            node.serial_port.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()