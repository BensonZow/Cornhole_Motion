#!/usr/bin/env python3
import math
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import message_filters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.time import Time


class BeanBagTracker(Node):
    def __init__(self):
        super().__init__('bean_bag_tracker')

        # Parameters (adjustable via ROS params)
        self.declare_parameter('hole_distance_inches', 10.0)
        self.declare_parameter('max_z_meters', 5.0)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        self.declare_parameter('reset_delay_sec', 10.0)
        self.declare_parameter('min_publish_interval_sec', 5.0)
        # Max horizontal miss (m, Euclidean from hole) for publishing / motion; larger → no publish, keyboard still arms.
        self.declare_parameter('max_publish_distance_m', 0.5)

        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254  # convert to meters
        self.min_publish_interval = float(self.get_parameter('min_publish_interval_sec').value)
        self.max_publish_distance_m = float(self.get_parameter('max_publish_distance_m').value)

        self._last_publish_mono = float('-inf')
        self._state_lock = threading.Lock()
        self._pending_keyboard_reset = False
        self._stdin_arm_thread: threading.Thread | None = None
        # Camera intrinsics (filled when CameraInfo received)
        self.fx = self.fy = self.cx = self.cy = None

        # State variables
        self.required_points = 3 
        self.state = 'IDLE'          # IDLE, COLLECTING, WAIT
        self.points = []              # list of (timestamp, x, y, z) in meters
        self.reset_timer = None
        # Use a Reentrant group so the timer and subscriber can run simultaneously
        self.callback_group = ReentrantCallbackGroup()
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

        # stdin thread sets _pending_keyboard_reset; this timer runs on the executor thread.
        self._keyboard_poll_timer = self.create_timer(
            0.05, self._keyboard_poll_callback, callback_group=self.callback_group)

        self.get_logger().info('Bean Bag Tracker node started')

    def camera_info_callback(self, msg):
        """Extract camera intrinsics once."""
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def image_callback(self, color_msg, depth_msg):
        """Process only the first 3 frames, then stop."""
        # IF we are in WAIT state, we do zero work (saves CPU)
        if self.state == 'WAIT' or self.fx is None:
            return

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        centroid = self.find_red_centroid(color_image)

        if centroid is None:
            return

        # 2. Extract 3D data
        u, v = centroid
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        depth = int(depth_image[v, u]) / 1000.0

        if not (0.2 < depth < 4.0):
            return

        t = Time.from_msg(color_msg.header.stamp).nanoseconds / 1e9

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy

        with self._state_lock:
            # Start collecting (respect min interval since last publish)
            if self.state == 'IDLE':
                elapsed = time.monotonic() - self._last_publish_mono
                if elapsed < self.min_publish_interval:
                    return
                self.points = [(t, x, y, depth)]
                self.state = 'COLLECTING'

            # Add subsequent points
            elif self.state == 'COLLECTING':
                self.points.append((t, x, y, depth))

                # TRIGGER: Once we hit 3, calculate and SHUT DOWN processing
                if len(self.points) >= self.required_points:
                    published, arm_keyboard = self.compute_and_publish()

                    if arm_keyboard:
                        self.state = 'WAIT'
                        self._spawn_stdin_arm_thread()
                    else:
                        self.points.clear()
                        self.state = 'IDLE'

    def _spawn_stdin_arm_thread(self) -> None:
        """Wait for Enter (TTY stdin), then remaining publish cooldown; arm IDLE via poll timer."""
        min_interval = self.min_publish_interval

        def worker() -> None:
            try:
                # Requires a TTY when launched with `ros2 run` in a terminal.
                sys.stdout.write('Press Enter when ready for the next bag...\n')
                sys.stdout.flush()
                input()
            except EOFError:
                pass
            rem = max(0.0, min_interval - (time.monotonic() - self._last_publish_mono))
            if rem > 0:
                time.sleep(rem)
            with self._state_lock:
                self._pending_keyboard_reset = True

        self._stdin_arm_thread = threading.Thread(target=worker, daemon=True)
        self._stdin_arm_thread.start()

    def _keyboard_poll_callback(self) -> None:
        with self._state_lock:
            if not self._pending_keyboard_reset:
                return
            self._apply_idle_reset()

    def _apply_idle_reset(self) -> None:
        """Clear wait flag and collection state; caller must hold ``_state_lock``."""
        self._pending_keyboard_reset = False
        self.state = 'IDLE'
        self.points.clear()

    def reset_state(self):
        """Re-enable image processing for the next throw."""
        with self._state_lock:
            self._apply_idle_reset()

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

    def _backtrack_sample_point_lines(self, t0: float) -> list[str]:
        lines: list[str] = ['  samples (camera frame, hole at origin in board plane):']
        for i, (t_abs, xi, yi, zi) in enumerate(self.points):
            tr = t_abs - t0
            lines.append(
                f'    pt[{i}] t_ros={t_abs:.6f}s  t_rel={tr:.6f}s  '
                f'x={xi:.6f}m  y={yi:.6f}m  z={zi:.6f}m'
            )
        return lines

    def _emit_bag_distance_backtrack(self, lines: list[str]) -> None:
        """Single INFO log per completed 3-point bag cycle (distance math only)."""
        self.get_logger().info('\n'.join(lines))

    def compute_and_publish(self) -> tuple[bool, bool]:
        """Fit trajectory, predict landing. Returns (published, arm_keyboard).

        ``published`` is True only if a message was sent. ``arm_keyboard`` is True
        to enter WAIT + stdin (all finished cycles except compute aborted on cooldown).
        """
        now = time.monotonic()
        self.points.sort(key=lambda p: p[0])
        t0 = self.points[0][0]

        if now - self._last_publish_mono < self.min_publish_interval:
            lines = [
                '--- bag distance backtrack ---',
                'outcome: publish_cooldown (no trajectory message; landing math not run)',
                f'  elapsed_since_last_publish_s={now - self._last_publish_mono:.6f}  '
                f'min_publish_interval_s={self.min_publish_interval:.6f}',
            ]
            lines.extend(self._backtrack_sample_point_lines(t0))
            self._emit_bag_distance_backtrack(lines)
            return (False, False)

        times = np.array([p[0] - t0 for p in self.points])
        xs = np.array([p[1] for p in self.points])
        ys = np.array([p[2] for p in self.points])
        zs = np.array([p[3] for p in self.points])

        coeffs_x = np.polyfit(times, xs, 1)   # [vx, x0]  => x(t)=vx*t+x0
        coeffs_z = np.polyfit(times, zs, 1)   # [vz, z0]  => z(t)=vz*t+z0
        coeffs_y = np.polyfit(times, ys, 2)   # [a/2, vy, y0] => y(t)=(a/2)*t^2+vy*t+y0

        vx, x0 = coeffs_x
        vz, z0 = coeffs_z
        a_half, vy, y0 = coeffs_y

        zh = float(self.hole_distance)
        lines = [
            '--- bag distance backtrack ---',
        ]
        lines.extend(self._backtrack_sample_point_lines(t0))
        lines.append(
            f'  polyfit (t_rel from first sample): '
            f'x(t)={vx:.6f}*t+{x0:.6f}  z(t)={vz:.6f}*t+{z0:.6f}  '
            f'y(t)={a_half:.6f}*t^2+{vy:.6f}*t+{y0:.6f}'
        )
        lines.append(
            f'  backtrack z to hole plane: z(t_land)=z_hole={zh:.6f}m  '
            f'=> vz*t_land+z0=zh  => t_land=(zh-z0)/vz'
        )

        if abs(vz) < 1e-6:
            lines.append(f'  vz={vz:.6e}  => |vz|<1e-6, t_land undefined (division by zero)')
            lines.append('  outcome: vz_near_zero (no /bean_bag_trajectory publish)')
            self._emit_bag_distance_backtrack(lines)
            return (False, True)

        t_land = (zh - z0) / vz
        x_land = vx * t_land + x0
        y_land = a_half * t_land**2 + vy * t_land + y0
        dx = x_land
        dy = y_land
        distance_m = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)

        lines.append(
            f'  t_land = (zh-z0)/vz = ({zh:.6f}-{z0:.6f})/{vz:.6f} = {t_land:.6f} s'
        )
        lines.append(
            f'  x_land = vx*t_land+x0 = {vx:.6f}*{t_land:.6f}+{x0:.6f} = {x_land:.6f} m'
        )
        lines.append(
            f'  y_land = (a/2)*t_land^2+vy*t_land+y0 = '
            f'{a_half:.6f}*{t_land:.6f}^2+{vy:.6f}*{t_land:.6f}+{y0:.6f} = {y_land:.6f} m'
        )
        lines.append(
            f'  miss from hole (origin): dx=x_land={dx:.6f} m, dy=y_land={dy:.6f} m'
        )
        lines.append(
            f'  distance_m = hypot(dx,dy) = sqrt({dx:.6f}^2+{dy:.6f}^2) = {distance_m:.6f} m'
        )
        lines.append(f'  angle_rad = atan2(dy,dx) = {angle:.6f}')
        lines.append(f'  max_publish_distance_m = {self.max_publish_distance_m:.6f}')

        topic = self.get_parameter('result_topic').value
        if distance_m > self.max_publish_distance_m:
            lines.append(
                '  outcome: withheld_over_distance_cap '
                f'(distance_m > max_publish_distance_m; no publish on {topic!r})'
            )
            self._emit_bag_distance_backtrack(lines)
            return (False, True)

        msg = Float32MultiArray()
        msg.data = [float(distance_m), float(angle)]
        self.publisher.publish(msg)
        self._last_publish_mono = time.monotonic()
        lines.append(
            f'  outcome: published std_msgs/Float32MultiArray on {topic!r} '
            f'data=[distance_m, angle_rad]=[{distance_m:.6f}, {angle:.6f}]'
        )
        self._emit_bag_distance_backtrack(lines)

        return (True, True)

    def reset(self):
        """Reset after 10s wait period."""
        if self.reset_timer is not None:
            self.reset_timer.cancel()
            self.reset_timer = None
        with self._state_lock:
            self._apply_idle_reset()

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