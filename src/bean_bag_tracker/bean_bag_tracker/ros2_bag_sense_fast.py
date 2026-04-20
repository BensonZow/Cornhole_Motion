#!/usr/bin/env python3
"""Bean bag trajectory from color + depth + NN detection (``/bean_bag_detection``).

Depth is read only at the NN box centroid. Samples are taken only on a strict
approach toward the camera (depth decreasing), after a frame beyond ``max_z_meters``.
"""
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_best_available, qos_profile_sensor_data
from rclpy.time import Time

from bean_bag_tracker.trajectory_common import compute_and_publish_points


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
        # Depth frames often lag color by >100 ms on RealSense USB; 0.1 s was too tight (sync never fired).
        self.declare_parameter('sync_slop_sec', 0.5)
        self.declare_parameter('debug_ingest', False)
        self.declare_parameter('debug_ingest_period_sec', 0.5)
        # Save color PNG with box + centroid + depth whenever a synced NN detection has a box.
        self.declare_parameter('centroid_png_enabled', True)
        # Empty string → ``<workspace>/log/bag_fast_centroid`` (see ``_default_centroid_png_dir``).
        self.declare_parameter('centroid_png_output_dir', '')
        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254  # convert to meters
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_publish_interval = float(self.get_parameter('min_publish_interval_sec').value)
        self.max_publish_distance_m = float(self.get_parameter('max_publish_distance_m').value)
        self.min_z_meters = float(self.get_parameter('min_z_meters').value)
        self.max_z_meters = float(self.get_parameter('max_z_meters').value)
        self._sync_slop_sec = max(0.05, float(self.get_parameter('sync_slop_sec').value))
        self._sync_slop_ns = int(self._sync_slop_sec * 1e9)
        self._debug_ingest = bool(self.get_parameter('debug_ingest').value)
        self._debug_ingest_period = max(0.05, float(self.get_parameter('debug_ingest_period_sec').value))
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
        self._ingest_lock = threading.Lock()
        self._ingest_sync_total = 0
        self._ingest_with_detection = 0
        self._ingest_no_detection = 0
        self._ingest_invalid_depth = 0
        self._ingest_short_detection_row = 0
        self._ingest_first_detection_logged = False
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
        self._depth_ring: deque[tuple[int, Image]] = deque(maxlen=120)
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

        ct = self.get_parameter('color_topic').value
        dt = self.get_parameter('depth_topic').value
        bt = self.get_parameter('bag_detection_topic').value
        # Stamp-based soft sync (color + det share the same stamp from NN; depth matched within slop).
        self.create_subscription(
            Image,
            ct,
            self._on_color_soft,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Image,
            dt,
            self._on_depth_soft,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.info_sub = self.create_subscription(CameraInfo,
                                                 self.get_parameter('camera_info_topic').value,
                                                 self.camera_info_callback,
                                                 1)

        # ``qos_profile_best_available`` matches RELIABLE NN output and avoids silent non-matching QoS.
        self.create_subscription(
            Image,
            bt,
            self._on_det_soft,
            qos_profile_best_available,
            callback_group=self.callback_group,
        )

        # stdin thread sets _pending_keyboard_reset; this timer runs on the executor thread.
        self._keyboard_poll_timer = self.create_timer(
            0.05, self._keyboard_poll_callback, callback_group=self.callback_group)
        if self._debug_ingest:
            self._ingest_debug_timer = self.create_timer(
                self._debug_ingest_period,
                self._debug_ingest_timer_callback,
                callback_group=self.callback_group,
            )

        self._ingest_health_timer = self.create_timer(
            6.0,
            self._ingest_health_once,
            callback_group=self.callback_group,
        )

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
            f'(1×9 float32 Image, depth match slop sync_slop_sec={self._sync_slop_sec:g})'
        )
        self.get_logger().info(
            'Color + depth: qos_profile_sensor_data. bag_detection: qos_profile_best_available '
            '(matches NN publisher). Triples: color stamp == det stamp + nearest depth within slop.',
        )
        if self._debug_ingest:
            self.get_logger().info(
                f'Debug ingest: periodic INFO every {self._debug_ingest_period:g}s '
                '(set debug_ingest:=false to keep the terminal quiet).',
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
        bt = self.get_parameter('bag_detection_topic').value
        pc = self.count_publishers(ct)
        pd = self.count_publishers(dt)
        pb = self.count_publishers(bt)
        with self._ingest_lock:
            sync = self._ingest_sync_total
        with self._soft_sync_lock:
            rxc = self._rx_color
            rxd = self._rx_depth
            rxf = self._rx_det
        if sync == 0 and not self._zero_sync_warned:
            self._zero_sync_warned = True
            self.get_logger().warn(
                'Still no aligned color+depth+det frames by ~6s (ingest_callbacks=0). '
                'Check: (1) realsense2_camera + bean_bag_nn_detector running; '
                '(2) topic names; (3) increase sync_slop_sec; '
                f'(4) publisher counts color={pc} depth={pd} bag_detection={pb}; '
                f'(5) messages received color={rxc} depth={rxd} det={rxf}.',
            )

    def _debug_ingest_timer_callback(self) -> None:
        if not self._debug_ingest:
            return
        with self._ingest_lock:
            sync = self._ingest_sync_total
            wdet = self._ingest_with_detection
            ndet = self._ingest_no_detection
            bad = self._ingest_invalid_depth
            short = self._ingest_short_detection_row
            conf = self._last_det_conf
            uv = self._last_centroid_uv
            zm = self._last_depth_m
            cf = self._last_corners_flat
            skew = self._last_stamp_skew_color_det_ns
            enc = self._last_det_encoding
            shp = self._last_det_shape
        with self._state_lock:
            st = self.state
            prev_z = self._prev_nn_centroid_depth
        skew_ms = (skew / 1e6) if skew is not None else None
        skew_s = f'{skew_ms:.2f} ms color-vs-det' if skew_ms is not None else 'n/a'
        self.get_logger().info(
            '[ingest] '
            f'sync={sync} nn_box={wdet} no_box={ndet} bad_depth={bad} short_row={short} '
            f'state={st} prev_z={prev_z} | conf={conf:.3f} uv={uv} z_m={zm} | '
            f'det {shp} {enc} skew={skew_s}',
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
            if len(self._soft_emit_ok) > 320:
                for k in sorted(self._soft_emit_ok)[:160]:
                    self._soft_emit_ok.discard(k)
            bundle = (c, d, n)
        self._image_sync_callback(bundle[0], bundle[1], bundle[2])

    def _on_color_soft(self, msg: Image) -> None:
        ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self._soft_sync_lock:
            self._rx_color += 1
            self._color_by_ns[ns] = msg
            self._trim_stamp_dict(self._color_by_ns, 100)
        self._try_soft_emit(ns)

    def _on_depth_soft(self, msg: Image) -> None:
        pending: list[int] = []
        with self._soft_sync_lock:
            self._rx_depth += 1
            ns = Time.from_msg(msg.header.stamp).nanoseconds
            self._depth_ring.append((ns, msg))
            pending = sorted(self._det_by_ns.keys())[-40:]
        for t in pending:
            self._try_soft_emit(t)

    def _on_det_soft(self, msg: Image) -> None:
        ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self._soft_sync_lock:
            self._rx_det += 1
            self._det_by_ns[ns] = msg
            self._trim_stamp_dict(self._det_by_ns, 100)
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
        """Synchronized color, depth, and NN detection; collect up to three samples."""
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

        if self.state == 'WAIT' or self.fx is None:
            return

        if corners is None:
            with self._ingest_lock:
                self._ingest_no_detection += 1
                self._last_det_conf = float(_conf)
                self._last_centroid_uv = None
                self._last_depth_m = None
                self._last_corners_flat = ''
            with self._state_lock:
                self._prev_nn_centroid_depth = None
            if self._debug_ingest:
                self.get_logger().debug(
                    f'[ingest] NN detector message OK but no box: conf={float(_conf):.4f} '
                    f'(conf<=0 or short row) det {self._last_det_shape} enc={self._last_det_encoding!r}',
                )
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
            if not self._ingest_first_detection_logged:
                self._ingest_first_detection_logged = True
                self.get_logger().info(
                    '[bean_bag_sense_fast/ingest] first synchronized NN detection accepted from '
                    f'{self.get_parameter("bag_detection_topic").value!r}: '
                    f'conf={float(_conf):.4f} corners(flat 8)={self._last_corners_flat}',
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
            if self._debug_ingest:
                self.get_logger().debug(
                    f'[ingest] NN box at centroid ({u},{v}) but depth=0 (invalid); conf={float(_conf):.4f}',
                )
            return

        with self._ingest_lock:
            self._last_centroid_uv = (u, v)
            self._last_depth_m = float(depth)

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
                    if self._debug_ingest:
                        self.get_logger().info(
                            '[bean_bag_sense_fast/ingest] COLLECTING 1/3: '
                            f'nn_conf={float(_conf):.4f} centroid=({u},{v}) depth_m={depth:.4f} '
                            f'bbox_area_px={bbox_area_px:.0f} in_band={in_band} approach_edge={approach_edge}',
                        )

            elif self.state == 'COLLECTING':
                last_depth = self.points[-1][3]
                if not in_band or not (depth < last_depth):
                    # Ignore receding / flat / out-of-band: abort this approach (require new far→closer edge).
                    self.points.clear()
                    self.state = 'IDLE'
                    self._debug_first_frame_bgr = None
                    self._prev_nn_centroid_depth = None
                    skip_prev_update = True
                    if self._debug_ingest:
                        self.get_logger().info(
                            '[bean_bag_sense_fast/ingest] COLLECTING aborted: '
                            f'in_band={in_band} depth={depth:.4f} m vs last={last_depth:.4f} m '
                            f'nn_conf={float(_conf):.4f} centroid=({u},{v})',
                        )
                else:
                    self.points.append((t, x, y, depth, u, v, bbox_area_px))
                    if self._debug_ingest:
                        self.get_logger().info(
                            f'[bean_bag_sense_fast/ingest] COLLECTING {len(self.points)}/3: '
                            f'nn_conf={float(_conf):.4f} centroid=({u},{v}) depth_m={depth:.4f}',
                        )

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