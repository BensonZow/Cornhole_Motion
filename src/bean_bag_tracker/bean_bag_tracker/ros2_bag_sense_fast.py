#!/usr/bin/env python3
"""Bean bag trajectory: depth at NN box + ``trajectory_common`` fit.

Two input modes (see ``use_segment_batch_from_nn``):
- **Batch (recommended with keyboard segment on NN):** three middle samples arrive on
  ``/bean_bag_throw_batch`` from ``bean_bag_nn_detector``; this node only needs depth
  to sample Z at the centroid and intrinsics to build ``(t,x,y,z)``.
- **Legacy:** soft-synced color + ``/bean_bag_detection`` + depth; throw buffer and
  ``throw_silence_timeout_sec`` then middle-3 fit.
"""
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from bean_bag_tracker.trajectory_common import TrajectoryPoint, compute_and_publish_points
from bean_bag_tracker.throw_batch_msg import unpack_throw_batch


def _default_centroid_png_dir() -> str:
    """``<ros2 workspace>/log/bag_fast_centroid`` when import path is under that workspace."""
    p = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        if os.path.isdir(os.path.join(p, 'src')):
            return os.path.join(p, 'log', 'bag_fast_centroid')
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.path.join(os.path.expanduser('~'), 'ros2_jazzy', 'log', 'bag_fast_centroid')


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
        # Max miss (m): hypot(x_land, depth_land - z_hole); larger → no publish.
        self.declare_parameter('max_publish_distance_m', 0.5)
        self.declare_parameter('bag_detection_topic', '/bean_bag_detection')
        self.declare_parameter('min_z_meters', 0.2)
        # Depth frames often lag color on RealSense USB; keep generous so depth_ring can match.
        self.declare_parameter('sync_slop_sec', 1.0)
        # Slow NN can publish detections long after the color frame; keep enough stamps so pairs are not evicted.
        self.declare_parameter('stamp_cache_max', 400)
        self.declare_parameter('depth_ring_max', 400)
        # When true, consume ``Float64MultiArray`` on throw_batch_topic (from segmented NN) only.
        self.declare_parameter('use_segment_batch_from_nn', True)
        self.declare_parameter('throw_batch_topic', '/bean_bag_throw_batch')
        self.declare_parameter('debug_throw_pipeline', False)
        # Save color PNG with box + centroid + depth whenever a synced NN detection has a box.
        self.declare_parameter('centroid_png_enabled', True)
        # Empty string → ``<workspace>/log/bag_fast_centroid`` (see ``_default_centroid_png_dir``).
        self.declare_parameter('centroid_png_output_dir', '')
        # End one throw when no valid in-band sample (box + depth) for this long (monotonic clock).
        self.declare_parameter('throw_silence_timeout_sec', 2)
        # Drop oldest buffer entries if the throw produces more than this (prevents unbounded memory).
        self.declare_parameter('max_throw_buffer_frames', 400)
        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254  # convert to meters
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_publish_interval = float(self.get_parameter('min_publish_interval_sec').value)
        self.max_publish_distance_m = float(self.get_parameter('max_publish_distance_m').value)
        self.min_z_meters = float(self.get_parameter('min_z_meters').value)
        self.max_z_meters = float(self.get_parameter('max_z_meters').value)
        self._throw_silence_timeout = max(0.05, float(self.get_parameter('throw_silence_timeout_sec').value))
        self._max_throw_buffer = max(3, int(self.get_parameter('max_throw_buffer_frames').value))
        self._last_buffer_trim_log_mono = float('-inf')
        self._sync_slop_sec = max(0.05, float(self.get_parameter('sync_slop_sec').value))
        self._sync_slop_ns = int(self._sync_slop_sec * 1e9)
        self._stamp_cache_max = max(32, int(self.get_parameter('stamp_cache_max').value))
        self._depth_ring_max = max(60, int(self.get_parameter('depth_ring_max').value))
        self._use_segment_batch = bool(self.get_parameter('use_segment_batch_from_nn').value)
        self._throw_batch_topic = str(self.get_parameter('throw_batch_topic').value).strip() or '/bean_bag_throw_batch'
        self._debug_throw_pipeline = bool(self.get_parameter('debug_throw_pipeline').value)
        self._centroid_png_enabled = bool(self.get_parameter('centroid_png_enabled').value)
        _raw_png_dir = str(self.get_parameter('centroid_png_output_dir').value).strip()
        self._centroid_png_dir = (
            os.path.expanduser(_raw_png_dir) if _raw_png_dir else _default_centroid_png_dir()
        )
        self._centroid_png_seq = 0
        self._centroid_png_lock = threading.Lock()
        if self._centroid_png_enabled:
            try:
                os.makedirs(self._centroid_png_dir, exist_ok=True)
                self.get_logger().info(
                    f'Centroid PNG snapshots enabled → {self._centroid_png_dir!r} '
                    '(one PNG per synced frame with NN box; set centroid_png_enabled:=false to disable).',
                )
            except OSError as exc:
                self._centroid_png_enabled = False
                self.get_logger().error(
                    f'Could not create centroid_png_output_dir {self._centroid_png_dir!r}: {exc}. '
                    'Disabling centroid PNG writes.',
                )

        self._last_publish_mono = float('-inf')
        self._state_lock = threading.Lock()
        # (t, x, y, depth_m, u_px, v_px, bbox_area_px) per accepted in-band frame (throw buffer)
        self._throw_buffer: list[TrajectoryPoint] = []
        self._last_valid_append_mono: float | None = None
        # Camera intrinsics (filled when CameraInfo received)
        self.fx = self.fy = self.cx = self.cy = None
        self._ingest_lock = threading.Lock()
        self._ingest_sync_total = 0
        self._ingest_with_detection = 0
        self._ingest_no_detection = 0
        self._ingest_invalid_depth = 0
        self._ingest_short_detection_row = 0
        self._last_det_conf = 0.0
        self._last_centroid_uv: tuple[int, int] | None = None
        self._last_depth_m: float | None = None
        self._last_corners_flat: str = ''
        self._last_stamp_skew_color_det_ns: int | None = None
        self._last_det_encoding: str = ''
        self._last_det_shape: str = ''
        self._zero_sync_warned = False
        self._ingest_health_done = False
        self._soft_sync_lock = threading.Lock()
        self._color_by_ns: dict[int, Image] = {}
        self._det_by_ns: dict[int, Image] = {}
        self._depth_ring: deque[tuple[int, Image]] = deque(maxlen=self._depth_ring_max)
        self._soft_emit_ok: set[int] = set()
        self._rx_color = 0
        self._rx_depth = 0
        self._rx_det = 0
        # Use a Reentrant group so the timer and subscriber can run simultaneously
        self.callback_group = ReentrantCallbackGroup()
        # ROS communication
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Float32MultiArray, 
                                               self.get_parameter('result_topic').value, 
                                               10)

        dt = self.get_parameter('depth_topic').value
        self.create_subscription(
            Image,
            dt,
            self._on_depth_soft,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            1,
        )

        if self._use_segment_batch:
            self.create_subscription(
                Float64MultiArray,
                self._throw_batch_topic,
                self._on_throw_batch,
                10,
                callback_group=self.callback_group,
            )
        else:
            ct = self.get_parameter('color_topic').value
            bt = self.get_parameter('bag_detection_topic').value
            self.create_subscription(
                Image,
                ct,
                self._on_color_soft,
                qos_profile=qos_profile_sensor_data,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                Image,
                bt,
                self._on_det_soft,
                qos_profile=qos_profile_sensor_data,
                callback_group=self.callback_group,
            )
            self._throw_silence_timer = self.create_timer(
                0.05, self._throw_silence_timer_callback, callback_group=self.callback_group
            )

        self._ingest_health_timer = self.create_timer(
            6.0,
            self._ingest_health_once,
            callback_group=self.callback_group,
        )

        mode = 'segment_batch' if self._use_segment_batch else 'legacy_sync'
        self.get_logger().info(
            f'bean_bag_tracker started mode={mode} depth_scale={self.depth_scale} sync_slop_sec={self._sync_slop_sec:g}'
        )
        if self._use_segment_batch:
            self.get_logger().info(
                f'Segment batch: {self._throw_batch_topic!r} + depth; in-band z in '
                f'({self.min_z_meters:g}, {self.max_z_meters:g}) m',
            )
        else:
            self.get_logger().info(
                f'Legacy: color+det+depth, throw_silence_timeout_sec={self._throw_silence_timeout:g} s, '
                f'max_throw_buffer_frames={self._max_throw_buffer}',
            )

    def _ingest_health_once(self) -> None:
        """One-shot (~6s): warn if no ingest callbacks yet (slow NN / QoS / topics)."""
        if self._ingest_health_done:
            return
        self._ingest_health_done = True
        try:
            self._ingest_health_timer.cancel()
        except Exception:
            pass
        ct = self.get_parameter('color_topic').value
        dt = self.get_parameter('depth_topic').value
        pd = self.count_publishers(dt)
        with self._ingest_lock:
            sync = self._ingest_sync_total
        with self._soft_sync_lock:
            rxd = self._rx_depth
        if self._use_segment_batch:
            pbatch = self.count_publishers(self._throw_batch_topic)
            if self._rx_depth == 0 and not self._zero_sync_warned:
                self._zero_sync_warned = True
                self.get_logger().warn(
                    f'~6s: no depth for batch mode. depth rx={self._rx_depth} depth_pub={pd} '
                    f'throw_batch_pub={pbatch} topic={self._throw_batch_topic!r}.',
                )
            return
        ct = self.get_parameter('color_topic').value
        bt = self.get_parameter('bag_detection_topic').value
        pc = self.count_publishers(ct)
        pb = self.count_publishers(bt)
        with self._soft_sync_lock:
            rxc = self._rx_color
            rxf = self._rx_det
        if sync == 0 and not self._zero_sync_warned:
            self._zero_sync_warned = True
            self.get_logger().warn(
                'Still no aligned color+depth+det frames by ~6s (ingest_callbacks=0). '
                'Check: (1) start bean_bag_nn_detector before (or with) this node so /bean_bag_detection '
                'has a publisher; (2) realsense2_camera running; (3) topic names; '
                '(4) increase sync_slop_sec / stamp_cache_max if NN is slower than color FPS; '
                f'(5) publisher counts color={pc} depth={pd} bag_detection={pb}; '
                f'(6) messages received color={rxc} depth={rxd} det={rxf}.',
            )

    def camera_info_callback(self, msg):
        """Extract camera intrinsics once."""
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    @staticmethod
    def _trim_stamp_dict(stamp_msg: dict[int, Image], max_keep: int) -> None:
        while len(stamp_msg) > max_keep:
            del stamp_msg[min(stamp_msg)]

    def _nearest_depth_locked(self, t_ns: int) -> Image | None:
        """Best depth ``Image`` whose stamp is within slop of ``t_ns``. Caller holds ``_soft_sync_lock``."""
        best: Image | None = None
        best_d: int | None = None
        for dns, dm in self._depth_ring:
            delta = abs(dns - t_ns)
            if delta <= self._sync_slop_ns and (best_d is None or delta < best_d):
                best = dm
                best_d = delta
        return best

    def _try_soft_emit(self, t_ns: int) -> None:
        """Emit one aligned (color, depth, det) triple for stamp ``t_ns`` (color == det stamp from NN)."""
        bundle: tuple[Image, Image, Image] | None = None
        with self._soft_sync_lock:
            c = self._color_by_ns.get(t_ns)
            n = self._det_by_ns.get(t_ns)
            if c is None or n is None:
                return
            d = self._nearest_depth_locked(t_ns)
            if d is None:
                return
            if t_ns in self._soft_emit_ok:
                return
            self._soft_emit_ok.add(t_ns)
            _emit_trim_hi = max(320, self._stamp_cache_max * 2)
            _emit_trim_lo = max(160, self._stamp_cache_max)
            while len(self._soft_emit_ok) > _emit_trim_hi:
                for k in sorted(self._soft_emit_ok)[:_emit_trim_lo]:
                    self._soft_emit_ok.discard(k)
            bundle = (c, d, n)
        self._image_sync_callback(bundle[0], bundle[1], bundle[2])

    def _on_color_soft(self, msg: Image) -> None:
        ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self._soft_sync_lock:
            self._rx_color += 1
            self._color_by_ns[ns] = msg
            self._trim_stamp_dict(self._color_by_ns, self._stamp_cache_max)
        self._try_soft_emit(ns)

    def _on_depth_soft(self, msg: Image) -> None:
        pending: list[int] = []
        with self._soft_sync_lock:
            self._rx_depth += 1
            ns = Time.from_msg(msg.header.stamp).nanoseconds
            self._depth_ring.append((ns, msg))
            pending = sorted(self._det_by_ns.keys())[-min(160, self._stamp_cache_max):]
        for t in pending:
            self._try_soft_emit(t)

    def _on_det_soft(self, msg: Image) -> None:
        ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self._soft_sync_lock:
            self._rx_det += 1
            self._det_by_ns[ns] = msg
            self._trim_stamp_dict(self._det_by_ns, self._stamp_cache_max)
        self._try_soft_emit(ns)

    def _decode_bag_detection_image(self, det_msg: Image) -> tuple[float, np.ndarray | None]:
        """Decode 1×9 ``32FC1`` row: ``[conf, u0,v0, … u3,v3]``."""
        row = self.bridge.imgmsg_to_cv2(det_msg, desired_encoding='32FC1')
        flat = np.asarray(row, dtype=np.float64).reshape(-1)
        if flat.size < 9:
            with self._ingest_lock:
                self._ingest_short_detection_row += 1
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

    @staticmethod
    def _middle_three(points: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
        """Sorted by time, then the centered window of three samples."""
        if len(points) < 3:
            return []
        s = sorted(points, key=lambda p: p[0])
        n = len(s)
        start = (n - 3) // 2
        return s[start : start + 3]

    @staticmethod
    def _decode_det_row_flat(flat: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Return ``(conf, corners 4x2)`` or no box when ``conf<=0`` or short row."""
        arr = np.asarray(flat, dtype=np.float64).reshape(-1)
        if arr.size < 9:
            return 0.0, None
        conf = float(arr[0])
        if conf <= 0.0:
            return conf, None
        return conf, arr[1:9].reshape(4, 2)

    def _on_throw_batch(self, msg: Float64MultiArray) -> None:
        if self.fx is None:
            return
        up = unpack_throw_batch(msg)
        if up is None:
            self.get_logger().error('throw_batch: unpack failed (wrong version or length)')
            return
        color_w, color_h, three = up
        points: list[TrajectoryPoint] = []
        for i, (stamp_ns, row) in enumerate(three):
            _conf, corners = self._decode_det_row_flat(row)
            if corners is None:
                self.get_logger().error(f'throw_batch: sample {i} has no valid detection')
                return
            if self._debug_throw_pipeline:
                ts, tn = stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000
                self.get_logger().info(
                    f'[bag_fast_throw_debug] rx i={i} stamp={ts}.{tn:09d} vertices_px={corners.tolist()}',
                )
            pt = self._trajectory_point_at_stamp(
                stamp_ns, corners, int(color_w), int(color_h), sample_index=i
            )
            if pt is None:
                return
            points.append(pt)
        if len(points) != 3:
            return
        _, _, new_last = compute_and_publish_points(
            points=points,
            depth_scale=self.depth_scale,
            hole_distance_m=self.hole_distance,
            max_publish_distance_m=self.max_publish_distance_m,
            min_publish_interval_sec=self.min_publish_interval,
            last_publish_mono=self._last_publish_mono,
            now_mono=time.monotonic(),
            publisher=self.publisher.publish,
            result_topic=self.get_parameter('result_topic').value,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )
        self._last_publish_mono = new_last
        if self._debug_throw_pipeline:
            self.get_logger().info(
                '[bag_fast_throw_debug] trajectory fit complete (see `bag distance backtrack` block above for x(t), y(t), depth(t))',
            )

    def _trajectory_point_at_stamp(
        self,
        stamp_ns: int,
        corners: np.ndarray,
        color_w: int,
        color_h: int,
        *,
        sample_index: int,
    ) -> TrajectoryPoint | None:
        w = color_w
        h = color_h
        u = int(np.clip(int(round(float(np.mean(corners[:, 0])))), 0, w - 1))
        v = int(np.clip(int(round(float(np.mean(corners[:, 1])))), 0, h - 1))
        with self._soft_sync_lock:
            dmsg = self._nearest_depth_locked(stamp_ns)
        if dmsg is None:
            self.get_logger().error(
                f'throw_batch: no depth image within sync_slop for stamp_ns={stamp_ns} '
                f'(increase depth_ring_max or reduce throw duration)',
            )
            return None
        dns = Time.from_msg(dmsg.header.stamp).nanoseconds
        depth_image = self.bridge.imgmsg_to_cv2(dmsg, '16UC1')
        dh, dw = depth_image.shape[:2]
        ud = int(np.clip(u, 0, dw - 1))
        vd = int(np.clip(v, 0, dh - 1))
        depth_raw_u16 = int(depth_image[vd, ud])
        depth = depth_raw_u16 * self.depth_scale
        bbox_area_px = self._bbox_area_from_corners(corners)

        if self._debug_throw_pipeline:
            dlt = abs(dns - stamp_ns)
            self.get_logger().info(
                f'[bag_fast_throw_debug] i={sample_index} depth@({ud},{vd}) raw16={depth_raw_u16} '
                f'z_m={depth:.6f} match_delta_ns={dlt} slop_ns={self._sync_slop_ns} '
                f'in_band={self.min_z_meters < depth < self.max_z_meters}'
            )
            self.get_logger().info(
                f'[bag_fast_throw_debug] i={sample_index} t_ros_s={stamp_ns/1e9:.9f} u={u} v={v}  '
                f'x={((u - self.cx) * depth / self.fx):.6f}  y={((v - self.cy) * depth / self.fy):.6f}  z_depth={depth:.6f} m',
            )

        if depth_raw_u16 == 0:
            self.get_logger().error(f'throw_batch: sample {sample_index} invalid depth at centroid')
            return None
        in_band = self.min_z_meters < depth < self.max_z_meters
        if not in_band:
            self.get_logger().error(
                f'throw_batch: sample {sample_index} depth {depth:.4f} m out of in-band z range',
            )
            return None

        t = stamp_ns / 1e9
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return (t, x, y, depth, u, v, bbox_area_px)

    def _write_centroid_snapshot_png(
        self,
        bgr: np.ndarray,
        corners: np.ndarray,
        u: int,
        v: int,
        depth_raw_u16: int,
        depth_m: float | None,
        conf: float,
        stamp_ns: int,
    ) -> None:
        """Match ``bean_bag_nn_detector._render_labeled_snapshot`` box/conf style + centroid dot + depth (m)."""
        ok = False
        fn = ''
        try:
            os.makedirs(self._centroid_png_dir, exist_ok=True)
            cr = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            cr = np.nan_to_num(cr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            vis = np.ascontiguousarray(bgr.copy())
            h, w = vis.shape[:2]
            pts = np.round(cr).astype(np.int32).reshape(1, 4, 2)
            cv2.polylines(vis, pts, isClosed=True, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
            x0 = int(np.clip(int(pts[0, :, 0].min()), 0, w - 1))
            y_conf = int(np.clip(int(pts[0, :, 1].min()) - 8, 22, h - 1))
            cv2.putText(
                vis,
                f'conf={conf:.3f}',
                (x0, y_conf),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if depth_m is None:
                dline = f'Z @ centroid: invalid (raw={depth_raw_u16})'
            else:
                dline = f'Z @ centroid: {depth_m:.3f} m'
            y_z = int(np.clip(y_conf + 26, 24, h - 6))
            cv2.putText(
                vis,
                dline,
                (x0, y_z),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(vis, (u, v), 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (u, v), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.drawMarker(vis, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            with self._centroid_png_lock:
                self._centroid_png_seq += 1
                seq = self._centroid_png_seq
            fn = os.path.join(self._centroid_png_dir, f'{stamp_ns}_{seq:06d}.png')
            vis_out = np.ascontiguousarray(vis)
            ok = bool(cv2.imwrite(fn, vis_out))
        except Exception as exc:
            self.get_logger().error(f'Centroid PNG failed: {exc}')
        if fn and not ok:
            self.get_logger().warn(f'cv2.imwrite failed for {fn!r}')

    def _image_sync_callback(self, color_msg: Image, depth_msg: Image, det_msg: Image) -> None:
        """Synchronized color, depth, and NN detection; buffer valid 3D samples for a throw."""
        with self._ingest_lock:
            self._ingest_sync_total += 1
            t_c = Time.from_msg(color_msg.header.stamp).nanoseconds
            t_n = Time.from_msg(det_msg.header.stamp).nanoseconds
            self._last_stamp_skew_color_det_ns = abs(t_c - t_n)
            self._last_det_encoding = str(det_msg.encoding)
            self._last_det_shape = f'{int(det_msg.width)}x{int(det_msg.height)}'

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        h, w = color_image.shape[:2]

        _conf, corners = self._decode_bag_detection_image(det_msg)

        depth_image: np.ndarray | None = None
        if corners is not None:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            dh, dw = depth_image.shape[:2]
            u0, v0 = self._centroid_uv_from_corners(corners)
            u0 = int(np.clip(u0, 0, w - 1))
            v0 = int(np.clip(v0, 0, h - 1))
            ud = int(np.clip(u0, 0, dw - 1))
            vd = int(np.clip(v0, 0, dh - 1))
            raw16 = int(depth_image[vd, ud])
            zm = float(raw16 * self.depth_scale) if raw16 > 0 else None
            if self._centroid_png_enabled:
                stamp_ns = Time.from_msg(det_msg.header.stamp).nanoseconds
                self._write_centroid_snapshot_png(
                    color_image, corners, u0, v0, raw16, zm, float(_conf), stamp_ns,
                )

        if self.fx is None:
            return

        if corners is None:
            with self._ingest_lock:
                self._ingest_no_detection += 1
                self._last_det_conf = float(_conf)
                self._last_centroid_uv = None
                self._last_depth_m = None
                self._last_corners_flat = ''
            return

        with self._ingest_lock:
            self._ingest_with_detection += 1
            self._last_det_conf = float(_conf)
            self._last_corners_flat = np.array2string(
                corners.reshape(-1),
                precision=2,
                separator=',',
                max_line_width=120,
            )

        u, v = self._centroid_uv_from_corners(corners)
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        bbox_area_px = self._bbox_area_from_corners(corners)

        dh, dw = depth_image.shape[:2]
        ud = int(np.clip(u, 0, dw - 1))
        vd = int(np.clip(v, 0, dh - 1))
        depth_raw_u16 = int(depth_image[vd, ud])
        depth = depth_raw_u16 * self.depth_scale

        # Invalid depth at centroid: ignore frame (do not advance centroid-depth history).
        if depth_raw_u16 == 0:
            with self._ingest_lock:
                self._ingest_invalid_depth += 1
                self._last_centroid_uv = (u, v)
                self._last_depth_m = None
            return

        with self._ingest_lock:
            self._last_centroid_uv = (u, v)
            self._last_depth_m = float(depth)

        t = Time.from_msg(color_msg.header.stamp).nanoseconds / 1e9
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy

        in_band = self.min_z_meters < depth < self.max_z_meters
        if in_band:
            with self._state_lock:
                self._throw_buffer.append((t, x, y, depth, u, v, bbox_area_px))
                self._last_valid_append_mono = time.monotonic()
                # Cap buffer: drop oldest (throw front still ends with silence timeout on latest samples).
                over = len(self._throw_buffer) - self._max_throw_buffer
                if over > 0:
                    del self._throw_buffer[:over]
                    nowm = time.monotonic()
                    if nowm - self._last_buffer_trim_log_mono > 2.0:
                        self._last_buffer_trim_log_mono = nowm
                        self.get_logger().warn(
                            f'throw buffer exceeded max_throw_buffer_frames={self._max_throw_buffer}; '
                            f'dropped {over} oldest sample(s).',
                        )

    def _throw_silence_timer_callback(self) -> None:
        """If the throw buffer has not grown for throw_silence_timeout_sec, finalize the throw."""
        with self._state_lock:
            if not self._throw_buffer:
                return
            t_end = self._last_valid_append_mono
            if t_end is None:
                return
            if time.monotonic() - t_end < self._throw_silence_timeout:
                return
            raw = list(self._throw_buffer)
            self._throw_buffer.clear()
            self._last_valid_append_mono = None
        self._finalize_throw_from_buffer(raw)

    def _finalize_throw_from_buffer(self, raw: list[TrajectoryPoint]) -> None:
        n = len(raw)
        if n < 3:
            self.get_logger().warn(
                f'throw ended with only {n} in-band valid sample(s); need >= 3 for fit — discarding.',
            )
            return
        three = self._middle_three(raw)
        _, _, new_last = compute_and_publish_points(
            points=three,
            depth_scale=self.depth_scale,
            hole_distance_m=self.hole_distance,
            max_publish_distance_m=self.max_publish_distance_m,
            min_publish_interval_sec=self.min_publish_interval,
            last_publish_mono=self._last_publish_mono,
            now_mono=time.monotonic(),
            publisher=self.publisher.publish,
            result_topic=self.get_parameter('result_topic').value,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )
        self._last_publish_mono = new_last

def main(args=None):
    rclpy.init(args=args)
    node: BeanBagTracker | None = None
    try:
        node = BeanBagTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == '__main__':
    main()