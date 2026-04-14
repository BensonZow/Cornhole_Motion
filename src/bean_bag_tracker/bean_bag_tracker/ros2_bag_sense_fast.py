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

# Horizontal offsets from the fitted trajectory share this length unit; miss magnitude is converted to SI.
INCH_TO_METERS = 0.0254


class BeanBagTracker(Node):
    def __init__(self):
        super().__init__('bean_bag_tracker')

        # Parameters (adjustable via ROS params)
        self.declare_parameter('comm_timing', False)
        self.declare_parameter('hole_distance_inches', 10.0)
        self.declare_parameter('max_z_meters', 3.0)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        self.declare_parameter('reset_delay_sec', 10.0)
        self.declare_parameter('min_publish_interval_sec', 5.0)
        # Max horizontal miss (m, Euclidean from hole) for publishing / motion; larger → no publish, keyboard still arms.
        self.declare_parameter('max_publish_distance_m', 0.5)

        self.hole_distance = self.get_parameter('hole_distance_inches').value * INCH_TO_METERS
        self.max_z = self.get_parameter('max_z_meters').value
        self.min_publish_interval = float(self.get_parameter('min_publish_interval_sec').value)
        self.max_publish_distance_m = float(self.get_parameter('max_publish_distance_m').value)

        self._comm_timing = bool(self.get_parameter('comm_timing').value)
        self._comm_seq = 0
        self._last_comm_mono: float | None = None
        self._last_skip_log_mono = 0.0
        self._last_publish_mono = float('-inf')
        self._state_lock = threading.Lock()
        self._pending_keyboard_reset = False
        self._stdin_arm_thread: threading.Thread | None = None

        # Camera intrinsics (filled when CameraInfo received)
        self.fx = self.fy = self.cx = self.cy = None

        # State variables
        self.required_points = 3
        # Begin in WAIT until first Enter (same path as post-trajectory keyboard arm).
        self.state = 'WAIT'  # IDLE, COLLECTING, WAIT
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

        self.get_logger().info(
            'Bean Bag Tracker node started; waiting for Enter before bag detection.'
        )
        if self._comm_timing:
            self._comm_print('node_started', result_topic=self.get_parameter('result_topic').value)
        self._spawn_stdin_arm_thread(startup=True)

    def _comm_print(self, phase: str, **kwargs: object) -> None:
        if not self._comm_timing:
            return
        self._comm_seq += 1
        now = time.monotonic()
        delta_ms = 0.0
        if self._last_comm_mono is not None:
            delta_ms = (now - self._last_comm_mono) * 1000.0
        self._last_comm_mono = now
        parts = [
            f'[COMMDBG_BAG] seq={self._comm_seq} phase={phase}',
            f'mono_s={now:.6f}',
            f'delta_since_prev_ms={delta_ms:.3f}',
        ]
        for k, v in kwargs.items():
            parts.append(f'{k}={v}')
        print(' '.join(parts), file=sys.stderr, flush=True)

    def camera_info_callback(self, msg):
        """Extract camera intrinsics once."""
        if self.fx is None:
            t0 = time.monotonic()
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info('Camera intrinsics received')
            self._comm_print(
                'camera_intrinsics_set',
                parse_ms=f'{(time.monotonic() - t0) * 1000.0:.3f}',
                fx=self.fx,
                fy=self.fy,
            )

    def image_callback(self, color_msg, depth_msg):
        """Process only the first 3 frames, then stop."""
        stamp = f'{color_msg.header.stamp.sec}.{color_msg.header.stamp.nanosec:09d}'
        # IF we are in WAIT state, we do zero work (saves CPU)
        if self.state == 'WAIT' or self.fx is None:
            if self._comm_timing:
                now = time.monotonic()
                if now - self._last_skip_log_mono >= 2.0:
                    self._last_skip_log_mono = now
                    self._comm_print(
                        'image_cb_skip_throttled',
                        stamp=stamp,
                        reason='wait' if self.state == 'WAIT' else 'no_intrinsics',
                    )
            return

        t_frame0 = time.monotonic()
        # 1. Convert images only if we are actually looking for a bag
        t0 = time.monotonic()
        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        t1 = time.monotonic()
        centroid = self.find_red_centroid(color_image)
        t2 = time.monotonic()

        if centroid is None:
            self._comm_print(
                'image_cb_no_centroid',
                stamp=stamp,
                state=self.state,
                color_cv_ms=f'{(t1 - t0) * 1000.0:.3f}',
                red_detect_ms=f'{(t2 - t1) * 1000.0:.3f}',
            )
            return

        # 2. Extract 3D data
        u, v = centroid
        t3 = time.monotonic()
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        depth = depth_image[v, u] / 1000.0
        t4 = time.monotonic()

        if not (0.2 < depth < 4.0):
            self._comm_print(
                'image_cb_depth_reject',
                stamp=stamp,
                depth_m=depth,
                depth_cv_ms=f'{(t4 - t3) * 1000.0:.3f}',
            )
            return

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        t = Time.from_msg(color_msg.header.stamp).nanoseconds / 1e9

        self._comm_print(
            'image_cb_frame_timing',
            stamp=stamp,
            state_before=self.state,
            color_cv_ms=f'{(t1 - t0) * 1000.0:.3f}',
            red_detect_ms=f'{(t2 - t1) * 1000.0:.3f}',
            depth_cv_sample_ms=f'{(t4 - t3) * 1000.0:.3f}',
            frame_total_ms=f'{(t4 - t_frame0) * 1000.0:.3f}',
            depth_m=f'{depth:.4f}',
        )

        with self._state_lock:
            # Start collecting (respect min interval since last publish)
            if self.state == 'IDLE':
                elapsed = time.monotonic() - self._last_publish_mono
                if elapsed < self.min_publish_interval:
                    self._comm_print(
                        'collect_skipped_cooldown',
                        elapsed_since_publish_s=f'{elapsed:.3f}',
                        min_interval_s=self.min_publish_interval,
                    )
                    return
                self.get_logger().info('Bag detected! Starting 3-frame capture...')
                self._comm_print('collect_start_idle_to_collecting', u=u, v=v)
                self.points = [(t, x, y, depth)]
                self.state = 'COLLECTING'

            # Add subsequent points
            elif self.state == 'COLLECTING':
                self.points.append((t, x, y, depth))
                self._comm_print(
                    'collect_point_appended',
                    n=len(self.points),
                    u=u,
                    v=v,
                )

                # TRIGGER: Once we hit 3, calculate and SHUT DOWN processing
                if len(self.points) >= self.required_points:
                    self.get_logger().info(
                        f'Captured {self.required_points} frames. Calculating trajectory...')
                    self._comm_print('collect_complete_trigger_compute', n=len(self.points))
                    t_compute = time.monotonic()
                    published, arm_keyboard = self.compute_and_publish()
                    self._comm_print(
                        'compute_and_publish_returned',
                        compute_wall_ms=f'{(time.monotonic() - t_compute) * 1000.0:.3f}',
                        published=published,
                        arm_keyboard=arm_keyboard,
                    )

                    if arm_keyboard:
                        self.state = 'WAIT'
                        self._comm_print('state_wait_keyboard_arm')
                        self._spawn_stdin_arm_thread()
                        if not published:
                            self.get_logger().info(
                                'Trajectory not published (safety); press Enter for next bag.')
                    else:
                        self.points.clear()
                        self.state = 'IDLE'
                        self.get_logger().info(
                            'Trajectory compute skipped (cooldown); reset to IDLE for next detection.')

    def _spawn_stdin_arm_thread(self, startup: bool = False) -> None:
        """Wait for Enter (TTY stdin), then remaining publish cooldown; arm IDLE via poll timer."""
        min_interval = self.min_publish_interval
        prompt = (
            'Press Enter when ready to start tracking...\n'
            if startup
            else 'Press Enter when ready for the next bag...\n'
        )

        def worker() -> None:
            try:
                # Requires a TTY when launched with `ros2 run` in a terminal.
                sys.stdout.write(prompt)
                sys.stdout.flush()
                input()
            except EOFError:
                self.get_logger().warn(
                    'stdin closed (no TTY); arming next bag after cooldown only.')
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
        self._comm_print('reset_state_enter')
        self.get_logger().info('System reset. Ready for next bag.')
        self._comm_print('reset_state_done')

    def _apply_idle_reset(self) -> None:
        """Clear wait flag and collection state; caller must hold ``_state_lock``."""
        self._pending_keyboard_reset = False
        self.state = 'IDLE'
        self.points.clear()

    def reset_state(self):
        """Re-enable image processing for the next throw."""
        self._comm_print('reset_state_enter')
        with self._state_lock:
            self._apply_idle_reset()
        self.get_logger().info('System reset. Ready for next bag.')
        self._comm_print('reset_state_done')

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

    def compute_and_publish(self) -> tuple[bool, bool]:
        """Fit trajectory, predict landing. Returns (published, arm_keyboard).

        ``published`` is True only if a message was sent. ``arm_keyboard`` is True
        to enter WAIT + stdin (all finished cycles except compute aborted on cooldown).
        """
        t_compute0 = time.monotonic()
        self._comm_print('compute_enter', n_points=len(self.points))
        now = time.monotonic()
        if now - self._last_publish_mono < self.min_publish_interval:
            self.get_logger().warn(
                f'Publish cooldown active ({now - self._last_publish_mono:.2f}s '
                f'< {self.min_publish_interval}s), skipping publish')
            self._comm_print(
                'compute_abort_cooldown',
                elapsed_since_publish_s=f'{now - self._last_publish_mono:.3f}',
            )
            return (False, False)
        # Sort points by time (just in case)
        self.points.sort(key=lambda p: p[0])
        t0 = self.points[0][0]
        times = np.array([p[0] - t0 for p in self.points])
        xs = np.array([p[1] for p in self.points])
        ys = np.array([p[2] for p in self.points])
        zs = np.array([p[3] for p in self.points])

        # Fit linear models for x and z
        t_fit0 = time.monotonic()
        coeffs_x = np.polyfit(times, xs, 1)   # [vx, x0]
        coeffs_z = np.polyfit(times, zs, 1)   # [vz, z0]
        # Fit quadratic for y (gravity)
        coeffs_y = np.polyfit(times, ys, 2)   # [a/2, vy, y0]
        self._comm_print(
            'compute_polyfit_done',
            polyfit_ms=f'{(time.monotonic() - t_fit0) * 1000.0:.3f}',
        )

        # Extract coefficients
        vx, x0 = coeffs_x
        vz, z0 = coeffs_z
        a_half, vy, y0 = coeffs_y   # a is 2*a_half (acceleration)

        # Find time when z reaches hole distance (0.254 m)
        # z(t) = vz * t + z0
        # Solve vz * t + z0 = hole_distance
        if abs(vz) < 1e-6:
            self.get_logger().warn('vz near zero, cannot predict landing')
            self._comm_print('compute_abort_vz_near_zero', vz=vz)
            return (False, True)
        t_land = (self.hole_distance - z0) / vz

        if t_land < 0:
            self.get_logger().warn(f'Predicted landing time {t_land:.2f}s invalid (negative)')
            self._comm_print('compute_abort_t_land_negative', t_land_s=t_land)
            return (False, True)

        # Compute landing coordinates
        x_land = vx * t_land + x0
        y_land = a_half * t_land**2 + vy * t_land + y0

        # Offsets from hole (hole at x=0, y=0); x/y miss from the fit is in inches here.
        dx = x_land
        dy = y_land
        distance_in = math.hypot(dx, dy)
        distance_m = distance_in * INCH_TO_METERS
        angle = math.atan2(dy, dx)  # radians

        if distance_m > self.max_publish_distance_m:
            self.get_logger().warn(
                f'Landing miss {distance_m:.3f} m exceeds max {self.max_publish_distance_m:.3f} m; '
                'not publishing'
            )
            self._comm_print(
                'compute_abort_distance_m',
                distance_in=distance_in,
                distance_m=distance_m,
                max_publish_distance_m=self.max_publish_distance_m,
                angle_rad=angle,
                t_land_s=t_land,
            )
            return (False, True)

        msg = Float32MultiArray()
        msg.data = [float(distance_m), float(angle)]
        self._comm_print(
            'compute_before_publish',
            distance_in=distance_in,
            distance_m=distance_m,
            angle_rad=angle,
            t_land_s=t_land,
        )
        t_pub = time.monotonic()
        self.publisher.publish(msg)
        self._last_publish_mono = time.monotonic()
        self._comm_print(
            'compute_after_publish',
            publish_ms=f'{(time.monotonic() - t_pub) * 1000.0:.3f}',
            compute_total_ms=f'{(time.monotonic() - t_compute0) * 1000.0:.3f}',
        )

        self.get_logger().info(
            f'Published: distance={distance_m:.3f} m, angle={angle:.3f} rad'
        )

        return (True, True)

    def reset(self):
        """Reset after 10s wait period."""
        self._comm_print('reset_method_enter')
        if self.reset_timer is not None:
            self.reset_timer.cancel()
            self.reset_timer = None
        with self._state_lock:
            self._apply_idle_reset()
        self.get_logger().info('Reset complete, ready for next throw')
        self._comm_print('reset_method_done')

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