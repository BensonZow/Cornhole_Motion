#!/usr/bin/env python3
"""Depth-only bag capture: probe min-depth in ROI, buffer 3 approach frames, manual bbox labeling, then trajectory publish.

NN is not used. After three gated depth frames are stored, the user presses Enter, then draws a
rectangle on each depth visualization. Centroids + stored depth images feed the same fit/publish
logic as :mod:`bean_bag_tracker.trajectory_common` (shared with ``ros2_bag_sense_fast``).
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray

from bean_bag_tracker.trajectory_common import TrajectoryPoint, compute_and_publish_points


def _wait_for_confirm_y_line() -> None:
    sys.stdout.write('Type y then press Enter when ready for the next bag...\n')
    sys.stdout.flush()
    while True:
        try:
            line = input()
        except EOFError:
            return
        if line.strip().casefold() == 'y':
            return


def _wait_for_enter_after_capture() -> None:
    sys.stdout.write('Captured 3 frames. Press Enter to open labeling windows...\n')
    sys.stdout.flush()
    try:
        input()
    except EOFError:
        pass


def _depth_vis_bgr(depth_u16: np.ndarray) -> np.ndarray:
    """BGR false-color for labeling (not for metrology)."""
    mask = depth_u16 > 0
    if not np.any(mask):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    d = depth_u16.astype(np.float32)
    d[~mask] = np.nan
    lo = float(np.nanpercentile(d, 5.0))
    hi = float(np.nanpercentile(d, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, float(np.nanmax(d))
    norm = np.clip((d - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def _probe_depth_m(
    depth_u16: np.ndarray,
    depth_scale: float,
    x0_n: float,
    x1_n: float,
    y0_n: float,
    y1_n: float,
) -> float | None:
    h, w = depth_u16.shape[:2]
    xa = int(np.clip(min(x0_n, x1_n), 0.0, 1.0) * (w - 1))
    xb = int(np.clip(max(x0_n, x1_n), 0.0, 1.0) * (w - 1)) + 1
    ya = int(np.clip(min(y0_n, y1_n), 0.0, 1.0) * (h - 1))
    yb = int(np.clip(max(y0_n, y1_n), 0.0, 1.0) * (h - 1)) + 1
    roi = depth_u16[ya:yb, xa:xb]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(valid.min()) * depth_scale


class ManualDepthBagTracker(Node):
    def __init__(self) -> None:
        super().__init__('bean_bag_tracker_manual_depth')

        self.declare_parameter('hole_distance_inches', 10.0)
        self.declare_parameter('max_z_meters', 4.0)
        self.declare_parameter('min_z_meters', 0.2)
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('result_topic', '/bean_bag_trajectory')
        self.declare_parameter('min_publish_interval_sec', 5.0)
        self.declare_parameter('max_publish_distance_m', 0.5)
        # Normalized ROI [0..1] for min-depth probe (defaults: full frame).
        self.declare_parameter('probe_roi_x_min', 0.0)
        self.declare_parameter('probe_roi_x_max', 1.0)
        self.declare_parameter('probe_roi_y_min', 0.0)
        self.declare_parameter('probe_roi_y_max', 1.0)

        self.hole_distance = float(self.get_parameter('hole_distance_inches').value) * 0.0254
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_publish_interval = float(self.get_parameter('min_publish_interval_sec').value)
        self.max_publish_distance_m = float(self.get_parameter('max_publish_distance_m').value)
        self.min_z_meters = float(self.get_parameter('min_z_meters').value)
        self.max_z_meters = float(self.get_parameter('max_z_meters').value)
        self._roi = (
            float(self.get_parameter('probe_roi_x_min').value),
            float(self.get_parameter('probe_roi_x_max').value),
            float(self.get_parameter('probe_roi_y_min').value),
            float(self.get_parameter('probe_roi_y_max').value),
        )

        self._last_publish_mono = float('-inf')
        self._state_lock = threading.Lock()
        self._pending_keyboard_reset = False
        self._stdin_arm_thread: threading.Thread | None = None
        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None

        self.required_points = 3
        self.state = 'IDLE'  # IDLE, COLLECTING, LABELING, WAIT
        self._capture_buffer: list[dict[str, Any]] = []
        self._prev_probe_depth: float | None = None

        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('result_topic').value,
            10,
        )

        self._label_event = threading.Event()
        self._label_thread = threading.Thread(target=self._label_worker_loop, daemon=True)
        self._label_thread.start()

        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self._depth_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_callback,
            1,
        )
        self._keyboard_poll_timer = self.create_timer(
            0.05, self._keyboard_poll_callback, callback_group=self.callback_group
        )

        self.get_logger().info('Manual-depth bag tracker started (no NN, depth probe in ROI)')
        self.get_logger().info(
            f'probe_roi_norm=({self._roi[0]:g},{self._roi[1]:g},{self._roi[2]:g},{self._roi[3]:g})  '
            f'depth_scale={self.depth_scale:g}  max_z={self.max_z_meters:g} m  min_z={self.min_z_meters:g} m'
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def _depth_callback(self, msg: Image) -> None:
        if self.state in ('LABELING', 'WAIT') or self.fx is None:
            return

        depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        h, w = depth_u16.shape[:2]
        probe = _probe_depth_m(depth_u16, self.depth_scale, *self._roi)
        if probe is None:
            with self._state_lock:
                self._prev_probe_depth = None
            return

        in_band = self.min_z_meters < probe < self.max_z_meters
        skip_prev_update = False

        with self._state_lock:
            prev_d = self._prev_probe_depth
            approach_edge = prev_d is not None and prev_d > self.max_z_meters and probe < prev_d

            if self.state == 'IDLE':
                elapsed = time.monotonic() - self._last_publish_mono
                if elapsed >= self.min_publish_interval and in_band and approach_edge:
                    vis = _depth_vis_bgr(depth_u16)
                    t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
                    self._capture_buffer = [
                        {
                            't': t,
                            'depth_u16': depth_u16.copy(),
                            'vis_bgr': vis,
                            'probe_m': probe,
                        }
                    ]
                    self.state = 'COLLECTING'
                    self.get_logger().info(
                        f'COLLECTING frame 1/3 probe_z={probe:.3f} m (rising edge from prev_z={prev_d:.3f})'
                    )

            elif self.state == 'COLLECTING':
                last_probe = float(self._capture_buffer[-1]['probe_m'])
                if not in_band or not (probe < last_probe):
                    self._capture_buffer.clear()
                    self.state = 'IDLE'
                    self._prev_probe_depth = None
                    skip_prev_update = True
                    self.get_logger().info('COLLECTING aborted (probe out of band or not decreasing)')
                else:
                    vis = _depth_vis_bgr(depth_u16)
                    t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
                    self._capture_buffer.append(
                        {
                            't': t,
                            'depth_u16': depth_u16.copy(),
                            'vis_bgr': vis,
                            'probe_m': probe,
                        }
                    )
                    self.get_logger().info(
                        f'COLLECTING frame {len(self._capture_buffer)}/3 probe_z={probe:.3f} m'
                    )
                    if len(self._capture_buffer) >= self.required_points:
                        self.state = 'LABELING'
                        self._label_event.set()

            if not skip_prev_update:
                self._prev_probe_depth = probe

    def _label_worker_loop(self) -> None:
        while rclpy.ok():
            if not self._label_event.wait(timeout=0.5):
                continue
            self._label_event.clear()
            try:
                self._process_one_label_cycle()
            except Exception as e:
                self.get_logger().error(f'label/publish cycle failed: {e}', exc_info=True)
                with self._state_lock:
                    self._capture_buffer.clear()
                    self.state = 'IDLE'
                    self._prev_probe_depth = None

    def _process_one_label_cycle(self) -> None:
        with self._state_lock:
            if len(self._capture_buffer) < self.required_points:
                self.state = 'IDLE'
                return
            frames = [
                {
                    't': float(f['t']),
                    'depth_u16': f['depth_u16'].copy(),
                    'vis_bgr': f['vis_bgr'].copy(),
                    'probe_m': float(f['probe_m']),
                }
                for f in self._capture_buffer
            ]
            self._capture_buffer.clear()

        try:
            cv2.startWindowThread()
        except Exception:
            pass

        _wait_for_enter_after_capture()

        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        if fx is None or fy is None or cx is None or cy is None:
            self.get_logger().error('Camera intrinsics missing; aborting label cycle')
            with self._state_lock:
                self.state = 'IDLE'
                self._prev_probe_depth = None
            return

        points: list[TrajectoryPoint] = []
        for i, fr in enumerate(frames):
            vis = fr['vis_bgr']
            win = f'manual_bag_frame_{i + 1}_of_3'
            try:
                r = cv2.selectROI(win, vis, showCrosshair=True, fromCenter=False)
            finally:
                cv2.destroyWindow(win)
            x, y, rw, rh = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            if rw <= 1 or rh <= 1:
                self.get_logger().warn('Empty ROI; aborting cycle')
                with self._state_lock:
                    self.state = 'IDLE'
                    self._prev_probe_depth = None
                return
            u = int(np.clip(round(x + 0.5 * rw), 0, vis.shape[1] - 1))
            v = int(np.clip(round(y + 0.5 * rh), 0, vis.shape[0] - 1))
            depth_img = fr['depth_u16']
            depth_raw = int(depth_img[v, u])
            if depth_raw == 0:
                self.get_logger().warn(f'Invalid depth at ({u},{v}) frame {i}; aborting cycle')
                with self._state_lock:
                    self.state = 'IDLE'
                    self._prev_probe_depth = None
                return
            depth_m = depth_raw * self.depth_scale
            xm = (u - cx) * depth_m / fx
            ym = (v - cy) * depth_m / fy
            area = float(max(0, rw) * max(0, rh))
            points.append((fr['t'], xm, ym, depth_m, u, v, area))
            self.get_logger().info(
                f'  labeled pt[{i}] centroid=({u},{v}) depth={depth_m:.4f} m probe_at_cap={fr["probe_m"]:.4f} m'
            )

        first_bgr = frames[0]['vis_bgr'].copy()
        published, arm_keyboard, new_last = compute_and_publish_points(
            points=points,
            depth_scale=self.depth_scale,
            hole_distance_m=self.hole_distance,
            max_publish_distance_m=self.max_publish_distance_m,
            min_publish_interval_sec=self.min_publish_interval,
            last_publish_mono=self._last_publish_mono,
            now_mono=time.monotonic(),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            first_bgr=first_bgr,
            publisher=self.publisher.publish,
            result_topic=self.get_parameter('result_topic').value,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
            first_frame_panel3_title='Frame 1 depth colormap',
            bbox_area_caption='Manual bbox area (axis-aligned, px^2):',
        )
        self._last_publish_mono = new_last

        with self._state_lock:
            if arm_keyboard:
                self.state = 'WAIT'
                self._spawn_stdin_arm_thread()
            else:
                self.state = 'IDLE'
                self._prev_probe_depth = None

    def _spawn_stdin_arm_thread(self) -> None:
        min_interval = self.min_publish_interval

        def worker() -> None:
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
        self._pending_keyboard_reset = False
        self.state = 'IDLE'
        self._capture_buffer.clear()
        self._prev_probe_depth = None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ManualDepthBagTracker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
