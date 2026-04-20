#!/usr/bin/env python3
"""Bean bag trajectory from color + depth + NN detection (``/bean_bag_detection``).

Depth is read only at the NN box centroid. Samples are taken only on a strict
approach toward the camera (depth decreasing), after a frame beyond ``max_z_meters``.
"""
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

from bean_bag_tracker.trajectory_common import compute_and_publish_points


def _wait_for_confirm_y_line() -> None:
    """Block until the user types ``y`` and ends the line (Enter / Return).

    Uses line-oriented ``input()`` so the same behavior applies on Windows
    (CRLF), macOS, and Linux (LF). Empty lines and other text are ignored until
    a line whose stripped value is ``y`` (case-insensitive).
    """
    sys.stdout.write('Type y then press Enter when ready for the next bag...\n')
    sys.stdout.flush()
    while True:
        try:
            line = input()
        except EOFError:
            return
        if line.strip().casefold() == 'y':
            return


class BeanBagTracker(Node):
    def __init__(self):
        super().__init__('bean_bag_tracker')

        # Parameters (adjustable via ROS params)
        self.declare_parameter('hole_distance_inches', 10.0)
        self.declare_parameter('max_z_meters', 4.0)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        # Meters per raw depth unit (e.g. RealSense depth_sensor.get_depth_scale() often 0.001).
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        self.declare_parameter('reset_delay_sec', 10.0)
        self.declare_parameter('min_publish_interval_sec', 5.0)
        # Max miss (m): hypot(x_land, depth_land - z_hole); larger → no publish, keyboard still arms.
        self.declare_parameter('max_publish_distance_m', 0.5)
        self.declare_parameter('bag_detection_topic', '/bean_bag_detection')
        self.declare_parameter('min_z_meters', 0.2)
        self.declare_parameter('sync_slop_sec', 0.1)
        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254  # convert to meters
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_publish_interval = float(self.get_parameter('min_publish_interval_sec').value)
        self.max_publish_distance_m = float(self.get_parameter('max_publish_distance_m').value)
        self.min_z_meters = float(self.get_parameter('min_z_meters').value)
        self.max_z_meters = float(self.get_parameter('max_z_meters').value)
        self._sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)

        self._last_publish_mono = float('-inf')
        self._state_lock = threading.Lock()
        self._pending_keyboard_reset = False
        self._stdin_arm_thread: threading.Thread | None = None
        # Camera intrinsics (filled when CameraInfo received)
        self.fx = self.fy = self.cx = self.cy = None

        # State variables
        self.required_points = 3 
        self.state = 'IDLE'          # IDLE, COLLECTING, WAIT
        # (t, x, y, depth_m, u_px, v_px, bbox_area_px) — u,v = NN bbox centroid; depth only at centroid
        self.points = []
        # Previous synced-frame depth (m) at NN centroid; used for far→approaching (rising-edge) gate
        self._prev_nn_centroid_depth: float | None = None
        self._debug_first_frame_bgr: np.ndarray | None = None
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

        self.det_sub = message_filters.Subscriber(
            self,
            Image,
            self.get_parameter('bag_detection_topic').value,
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.det_sub], 10, self._sync_slop_sec
        )
        self.ts.registerCallback(self._image_sync_callback)

        # stdin thread sets _pending_keyboard_reset; this timer runs on the executor thread.
        self._keyboard_poll_timer = self.create_timer(
            0.05, self._keyboard_poll_callback, callback_group=self.callback_group)

        self.get_logger().info('Bean Bag Tracker node started')
        self.get_logger().info(
            f'Depth scale is: {self.depth_scale} m/raw_unit '
            '(ROS param depth_scale; same role as depth_sensor.get_depth_scale())'
        )
        self.get_logger().info(
            f'NN bag sense — samples only when z_centroid strictly decreases; first sample needs '
            f'prev_z > max_z ({self.max_z_meters:g} m) then closer; in-band {self.min_z_meters:g} < z < '
            f'{self.max_z_meters:g} m; bag_detection_topic='
            f'{self.get_parameter("bag_detection_topic").value!r} '
            f'(1×9 float32 Image, sync_slop_sec={self._sync_slop_sec:g})'
        )

    def camera_info_callback(self, msg):
        """Extract camera intrinsics once."""
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def _decode_bag_detection_image(self, det_msg: Image) -> tuple[float, np.ndarray | None]:
        """Decode 1×9 ``32FC1`` row: ``[conf, u0,v0, … u3,v3]``."""
        row = self.bridge.imgmsg_to_cv2(det_msg, desired_encoding='32FC1')
        flat = np.asarray(row, dtype=np.float64).reshape(-1)
        if flat.size < 9:
            return 0.0, None
        flat = flat[:9]
        conf = float(flat[0])
        if conf <= 0.0:
            return conf, None
        corners = flat[1:].reshape(4, 2)
        return conf, corners

    @staticmethod
    def _centroid_uv_from_corners(corners: np.ndarray) -> tuple[int, int]:
        """Pixel (u, v) at the centroid of the four NN box corners (depth is read only here)."""
        u = int(round(float(np.mean(corners[:, 0]))))
        v = int(round(float(np.mean(corners[:, 1]))))
        return u, v

    @staticmethod
    def _bbox_area_from_corners(corners: np.ndarray) -> float:
        umin, vmin = corners.min(axis=0)
        umax, vmax = corners.max(axis=0)
        return float(max(0.0, umax - umin) * max(0.0, vmax - vmin))

    def _image_sync_callback(self, color_msg: Image, depth_msg: Image, det_msg: Image) -> None:
        """Synchronized color, depth, and NN detection; collect up to three samples."""
        if self.state == 'WAIT' or self.fx is None:
            return

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        h, w = color_image.shape[:2]

        _conf, corners = self._decode_bag_detection_image(det_msg)
        if corners is None:
            with self._state_lock:
                self._prev_nn_centroid_depth = None
            return

        u, v = self._centroid_uv_from_corners(corners)
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        bbox_area_px = self._bbox_area_from_corners(corners)

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        depth_raw_u16 = int(depth_image[v, u])
        depth = depth_raw_u16 * self.depth_scale

        # Invalid depth at centroid: ignore frame (do not advance centroid-depth history).
        if depth_raw_u16 == 0:
            return

        t = Time.from_msg(color_msg.header.stamp).nanoseconds / 1e9
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy

        in_band = self.min_z_meters < depth < self.max_z_meters
        skip_prev_update = False

        with self._state_lock:
            prev_d = self._prev_nn_centroid_depth
            # Rising edge toward camera: last frame was beyond max_z, this frame is strictly closer.
            approach_edge = prev_d is not None and prev_d > self.max_z_meters and depth < prev_d

            if self.state == 'IDLE':
                elapsed = time.monotonic() - self._last_publish_mono
                if elapsed >= self.min_publish_interval and in_band and approach_edge:
                    self.points = [(t, x, y, depth, u, v, bbox_area_px)]
                    self.state = 'COLLECTING'
                    self._debug_first_frame_bgr = color_image.copy()

            elif self.state == 'COLLECTING':
                last_depth = self.points[-1][3]
                if not in_band or not (depth < last_depth):
                    # Ignore receding / flat / out-of-band: abort this approach (require new far→closer edge).
                    self.points.clear()
                    self.state = 'IDLE'
                    self._debug_first_frame_bgr = None
                    self._prev_nn_centroid_depth = None
                    skip_prev_update = True
                else:
                    self.points.append((t, x, y, depth, u, v, bbox_area_px))

                    if len(self.points) >= self.required_points:
                        _published, arm_keyboard = self.compute_and_publish()

                        if arm_keyboard:
                            self.state = 'WAIT'
                            self._spawn_stdin_arm_thread()
                        else:
                            self.points.clear()
                            self.state = 'IDLE'
                            self._debug_first_frame_bgr = None
                            self._prev_nn_centroid_depth = None

            if not skip_prev_update:
                self._prev_nn_centroid_depth = depth

    def _spawn_stdin_arm_thread(self) -> None:
        """Wait for a ``y`` line on stdin, then remaining publish cooldown; arm IDLE via poll timer."""
        min_interval = self.min_publish_interval

        def worker() -> None:
            # Requires a TTY when launched with `ros2 run` in a terminal.
            _wait_for_confirm_y_line()
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
        self._debug_first_frame_bgr = None
        self._prev_nn_centroid_depth = None

    def compute_and_publish(self) -> tuple[bool, bool]:
        """Fit trajectory, predict landing. Returns (published, arm_keyboard).

        Delegates to :mod:`bean_bag_tracker.trajectory_common`.
        """
        first_bgr = (
            self._debug_first_frame_bgr.copy()
            if self._debug_first_frame_bgr is not None
            else None
        )
        published, arm_keyboard, new_last = compute_and_publish_points(
            points=self.points,
            depth_scale=self.depth_scale,
            hole_distance_m=self.hole_distance,
            max_publish_distance_m=self.max_publish_distance_m,
            min_publish_interval_sec=self.min_publish_interval,
            last_publish_mono=self._last_publish_mono,
            now_mono=time.monotonic(),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            first_bgr=first_bgr,
            publisher=self.publisher.publish,
            result_topic=self.get_parameter('result_topic').value,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )
        self._last_publish_mono = new_last
        return (published, arm_keyboard)

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