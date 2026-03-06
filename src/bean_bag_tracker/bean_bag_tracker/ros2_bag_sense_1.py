#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import message_filters

class BeanBagTracker(Node):
    def __init__(self):
        super().__init__('bean_bag_tracker')

        # Parameters (adjustable via ROS params)
        self.declare_parameter('hole_distance_inches', 10.0)
        self.declare_parameter('max_z_meters', 3.0)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        self.declare_parameter('collection_timeout_sec', 2.0)
        self.declare_parameter('reset_delay_sec', 10.0)

        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254  # convert to meters
        self.max_z = self.get_parameter('max_z_meters').value
        self.collection_timeout = self.get_parameter('collection_timeout_sec').value
        self.reset_delay = self.get_parameter('reset_delay_sec').value

        # Camera intrinsics (filled when CameraInfo received)
        self.fx = self.fy = self.cx = self.cy = None

        # State variables
        self.state = 'IDLE'          # IDLE, COLLECTING, WAIT
        self.points = []              # list of (timestamp, x, y, z) in meters
        self.collection_start_time = None
        self.timeout_timer = None
        self.reset_timer = None

        # ROS communication
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Float32MultiArray, 
                                               self.get_parameter('result_topic').value, 
                                               10)

        # Subscribers with synchronization
        self.color_sub = message_filters.Subscriber(self, Image, 
                                                    self.get_parameter('color_topic').value)
        self.depth_sub = message_filters.Subscriber(self, Image, 
                                                    self.get_parameter('depth_topic').value)
        self.info_sub = self.create_subscription(CameraInfo,
                                                 self.get_parameter('camera_info_topic').value,
                                                 self.camera_info_callback,
                                                 1)

        # Approximate time synchronizer (slop 0.1s)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info('Bean Bag Tracker node started')

    def camera_info_callback(self, msg):
        """Extract camera intrinsics once."""
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info('Camera intrinsics received')

    def image_callback(self, color_msg, depth_msg):
        """Process synchronized color and depth images."""
        # Skip processing if in WAIT state or intrinsics not ready
        if self.state == 'WAIT' or self.fx is None:
            return

        # Convert ROS images to OpenCV
        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        # Find red bean bag centroid
        centroid = self.find_red_centroid(color_image)
        if centroid is None:
            # No bean bag detected
            self.check_timeout()
            return

        u, v = centroid
        # Get depth at pixel (u,v)
        depth = depth_image[v, u] / 1000.0  # convert mm to meters
        if depth <= 0 or depth > self.max_z:
            # Depth invalid or too far
            self.check_timeout()
            return

        # Convert to 3D point in camera coordinates (x right, y down, z forward)
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        # Get timestamp in seconds (ROS time)
        t = Time.from_msg(color_msg.header.stamp).nanoseconds / 1e9

        # State machine
        if self.state == 'IDLE':
            # Start new collection
            self.points = [(t, x, y, z)]
            self.collection_start_time = t
            self.state = 'COLLECTING'
            self.start_timeout_timer()
            self.get_logger().debug('Started collecting points')

        elif self.state == 'COLLECTING':
            # Add point and check if we have 5
            self.points.append((t, x, y, z))
            self.get_logger().debug(f'Collected point {len(self.points)}: '
                                     f'x={x:.3f}, y={y:.3f}, z={z:.3f}')
            if len(self.points) >= 5:
                # We have enough points, compute trajectory
                self.compute_and_publish()
                self.cancel_timeout_timer()
                self.state = 'WAIT'
                # Start 10s reset timer
                self.reset_timer = self.create_timer(self.reset_delay, self.reset)
                self.get_logger().info('Trajectory published, waiting 10s for next throw')
            else:
                # Reset timeout on each new point
                self.restart_timeout_timer()

    def find_red_centroid(self, bgr_image):
        """Detect the largest red blob and return its centroid (u, v) or None."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        # Red has two hue ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Optional: morphological ops to clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour by area
        largest = max(contours, key=cv2.contourArea)
        # Compute centroid using moments
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    def compute_and_publish(self):
        """Fit trajectory to collected points, predict landing, publish result."""
        # Sort points by time (just in case)
        self.points.sort(key=lambda p: p[0])
        t0 = self.points[0][0]
        times = np.array([p[0] - t0 for p in self.points])
        xs = np.array([p[1] for p in self.points])
        ys = np.array([p[2] for p in self.points])
        zs = np.array([p[3] for p in self.points])

        # Fit linear models for x and z
        coeffs_x = np.polyfit(times, xs, 1)   # [vx, x0]
        coeffs_z = np.polyfit(times, zs, 1)   # [vz, z0]
        # Fit quadratic for y (gravity)
        coeffs_y = np.polyfit(times, ys, 2)   # [a/2, vy, y0]

        # Extract coefficients
        vx, x0 = coeffs_x
        vz, z0 = coeffs_z
        a_half, vy, y0 = coeffs_y   # a is 2*a_half (acceleration)

        # Find time when z reaches hole distance (0.254 m)
        # z(t) = vz * t + z0
        # Solve vz * t + z0 = hole_distance
        if abs(vz) < 1e-6:
            self.get_logger().warn('vz near zero, cannot predict landing')
            return
        t_land = (self.hole_distance - z0) / vz

        # Check if t_land is positive and reasonable (e.g., < 5 sec)
        if t_land < 0 or t_land > 5.0:
            self.get_logger().warn(f'Predicted landing time {t_land:.2f}s out of range')
            return

        # Compute landing coordinates
        x_land = vx * t_land + x0
        y_land = a_half * t_land**2 + vy * t_land + y0

        # Offsets from hole (hole at x=0, y=0)
        dx = x_land
        dy = y_land

        # Convert to inches
        dx_in = dx / 0.0254
        dy_in = dy / 0.0254
        distance = math.hypot(dx_in, dy_in)
        angle = math.atan2(dy_in, dx_in)   # radians, range -pi to pi

        # Publish as Float32MultiArray
        msg = Float32MultiArray()
        msg.data = [float(distance), float(angle)]
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: distance={distance:.2f} in, angle={angle:.3f} rad')

    def start_timeout_timer(self):
        """Start or restart the timeout timer for collection."""
        self.cancel_timeout_timer()
        self.timeout_timer = self.create_timer(self.collection_timeout, self.collection_timeout_cb)

    def restart_timeout_timer(self):
        self.cancel_timeout_timer()
        self.start_timeout_timer()

    def cancel_timeout_timer(self):
        if self.timeout_timer is not None:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def collection_timeout_cb(self):
        """Called when no new point arrives within timeout."""
        self.get_logger().info('Collection timeout, resetting to IDLE')
        self.cancel_timeout_timer()
        self.points.clear()
        self.state = 'IDLE'

    def check_timeout(self):
        """Check if we are in COLLECTING and too much time passed since last point."""
        if self.state == 'COLLECTING' and self.timeout_timer is None:
            # Shouldn't happen, but safety
            self.collection_timeout_cb()

    def reset(self):
        """Reset after 10s wait period."""
        if self.reset_timer is not None:
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()