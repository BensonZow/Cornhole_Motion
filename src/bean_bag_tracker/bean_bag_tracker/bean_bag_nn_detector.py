#!/usr/bin/env python3
"""YOLO / PyTorch (.pt) bean-bag detector: color image in, stamped ``sensor_msgs/Image`` out.

Loads weights via Ultralytics (e.g. YOLO26n ``yolo26n.pt``). ROS 2 ``std_msgs/Float32MultiArray`` has no
``std_msgs/Header``, so detections are published as a 1×9 row, ``encoding='32FC1'``, with the same
``header`` as the source color frame (for ``message_filters`` sync with depth). The nine floats are::

    [confidence, u0, v0, u1, v1, u2, v2, u3, v3]

Corners are axis-aligned box vertices in **original color pixel coordinates**, order:
top-left, top-right, bottom-right, bottom-left. When nothing exceeds ``confidence_threshold``,
``confidence`` is ``0.0`` and all coordinates are ``0.0``.

When ``stream_snapshots_enabled`` is true, among detections at or above ``stream_snapshot_min_conf`` the
highest-confidence box is chosen for the message and for labeled PNGs under
``stream_snapshot_output_dir`` (rate-limited by ``stream_snapshot_min_interval_sec``).
PNG writes run on a background queue so inference is not blocked by disk I/O.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any


def _mitigate_native_segfaults() -> None:
    """Set before NumPy/OpenCV/Torch import.

    Mixed BLAS/OpenMP threading (OpenCV + PyTorch + NumPy) often segfaults on aarch64 / Raspberry Pi.
    """
    for key, val in (
        ('OMP_NUM_THREADS', '1'),
        ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'),
        ('NUMEXPR_NUM_THREADS', '1'),
        ('VECLIB_MAXIMUM_THREADS', '1'),
    ):
        os.environ.setdefault(key, val)


_mitigate_native_segfaults()

import cv2
import numpy as np
import rclpy

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'bean_bag_nn_detector requires ultralytics (and PyTorch) in the same Python as ``ros2 run``. '
        'Install e.g. ``pip3 install ultralytics``; on Debian/Ubuntu (PEP 668) you may need '
        '``pip3 install ultralytics --break-system-packages`` or a venv whose ``bin`` is on PATH.'
    ) from e

try:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:
    pass

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from bean_bag_tracker.throw_batch_msg import pack_throw_batch

# BEST_EFFORT like RealSense; depth 10 matches prior default publisher queue size.
_NN_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


def _scalar_float(name: str, val: Any) -> float:
    """ROS / YAML sometimes passes non-scalars; Ultralytics needs plain ``float``."""
    if isinstance(val, Mapping):
        raise TypeError(f'Parameter {name!r} must be a number, got a mapping (check launch YAML).')
    return float(val)


def _scalar_int(name: str, val: Any) -> int:
    if isinstance(val, Mapping):
        raise TypeError(f'Parameter {name!r} must be an integer, got a mapping (check launch YAML).')
    return int(val)


def _xyxy_to_corners_tl_tr_br_bl(xyxy: np.ndarray) -> np.ndarray:
    """(4,) xyxy -> (4,2) corners."""
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _render_labeled_snapshot(bgr: np.ndarray, conf: float, corners: np.ndarray | None) -> np.ndarray:
    """BGR copy with axis-aligned box (TL..BL) and confidence text."""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if conf > 0.0 and corners is not None and corners.shape == (4, 2):
        pts = np.round(corners).astype(np.int32).reshape(1, 4, 2)
        cv2.polylines(vis, pts, isClosed=True, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
        x0 = int(np.clip(pts[0, :, 0].min(), 0, w - 1))
        y0 = int(np.clip(pts[0, :, 1].min() - 8, 20, h - 1))
        cv2.putText(
            vis,
            f'conf={conf:.3f}',
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            'no detection (conf=0)',
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def _clip_xyxy(xyxy: np.ndarray, h0: int, w0: int) -> np.ndarray:
    x = xyxy.astype(np.float32).copy()
    x[[0, 2]] = np.clip(x[[0, 2]], 0.0, float(w0 - 1))
    x[[1, 3]] = np.clip(x[[1, 3]], 0.0, float(h0 - 1))
    return x


def _ultralytics_checkpoint_hint(path: str) -> str:
    """Return a non-empty error hint if ``path`` cannot be loaded by ``YOLO(path)`` (wrong format).

    Ultralytics expects ``torch.load`` to yield a dict whose ``model`` is an ``nn.Module``. A bare
    state_dict (``OrderedDict``) — common if the file was renamed or exported from another framework
    (DETR / RT-DETR / etc.) — triggers ``OrderedDict has no attribute float`` inside ultralytics.
    """
    import torch
    import torch.nn as nn

    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:  # pragma: no cover
        return f'Cannot read weights {path!r}: {e}'
    if not isinstance(ckpt, dict):
        return f'{path!r} is not a dict checkpoint; expected an Ultralytics YOLO .pt file.'
    m = ckpt.get('model')
    if m is None:
        return f'{path!r} has no ``model`` key; expected an Ultralytics YOLO .pt file.'
    if isinstance(m, nn.Module):
        return ''
    if isinstance(m, Mapping):
        return (
            f'{path!r} is not an Ultralytics YOLO checkpoint: ``model`` is a state_dict, not a full '
            'PyTorch module. The file may be mislabeled (e.g. not real yolo26n.pt) or from another '
            'training stack (ViT/DETR-style ``args``). Replace it with official YOLO26n weights from '
            'https://docs.ultralytics.com/models/yolo26/ or download via Ultralytics in a temp dir, '
            'then copy ``yolo26n.pt`` into ``share/bean_bag_tracker/models/`` and rebuild.'
        )
    return f'{path!r}: unexpected ``model`` type {type(m)!r}; expected Ultralytics YOLO .pt.'


class BeanBagNnDetector(Node):
    def __init__(self) -> None:
        super().__init__('bean_bag_nn_detector')

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/bean_bag_detection')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('inference_size', 640)
        self.declare_parameter('max_fps', 0.0)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('class_ids', [])
        # Labeled PNGs on ``y``+Enter (see ``_stdin_y_snapshot_loop``).
        self.declare_parameter('snapshot_output_dir', '/home/cornholio/ros2_jazzy/log/NN')
        self.declare_parameter('enable_y_snapshot', True)
        self.declare_parameter('stream_snapshots_enabled', True)
        self.declare_parameter('stream_snapshot_min_conf', 0.4)
        self.declare_parameter('stream_snapshot_min_interval_sec', 0.0)
        self.declare_parameter('stream_snapshot_output_dir', '')
        self.declare_parameter('stream_write_queue_size', 8)
        # <=0 disables periodic stdout timing (ms per pipeline stage + bag-detection publish Hz).
        self.declare_parameter('timing_stats_interval_sec', 1.0)
        # Keyboard ``b``/``e``+Enter: buffer color frames without inference, then YOLO on middle-3 only.
        self.declare_parameter('segmented_throw_mode', False)
        self.declare_parameter('throw_segment_max_frames', 2000)
        self.declare_parameter('throw_batch_topic', '/bean_bag_throw_batch')
        self.declare_parameter('debug_throw_pipeline', False)

        self._bridge = CvBridge()
        self._conf_thr = _scalar_float('confidence_threshold', self.get_parameter('confidence_threshold').value)
        self._inf_size = _scalar_int('inference_size', self.get_parameter('inference_size').value)
        self._max_fps = _scalar_float('max_fps', self.get_parameter('max_fps').value)
        self._device = str(self.get_parameter('device').value).strip() or 'cpu'
        raw_classes = self.get_parameter('class_ids').value
        if raw_classes is None:
            self._class_ids: list[int] = []
        elif isinstance(raw_classes, (list, tuple)):
            self._class_ids = [int(x) for x in raw_classes]
        else:
            self._class_ids = [int(raw_classes)]
        self._last_infer_mono = 0.0

        self._timing_interval = _scalar_float(
            'timing_stats_interval_sec',
            self.get_parameter('timing_stats_interval_sec').value,
        )
        self._timing_last_print_mono = time.monotonic()
        self._timing_frame_count = 0
        self._timing_pub_count = 0
        self._timing_sum_decode = 0.0
        self._timing_sum_infer = 0.0
        self._timing_sum_post = 0.0
        self._timing_sum_pub = 0.0

        self._snapshot_dir = os.path.expanduser(str(self.get_parameter('snapshot_output_dir').value).strip())
        self._enable_y_snapshot = bool(self.get_parameter('enable_y_snapshot').value)
        self._stream_snapshots_enabled = bool(self.get_parameter('stream_snapshots_enabled').value)
        self._stream_min_conf = _scalar_float(
            'stream_snapshot_min_conf',
            self.get_parameter('stream_snapshot_min_conf').value,
        )
        self._stream_interval = max(
            0.0,
            _scalar_float(
                'stream_snapshot_min_interval_sec',
                self.get_parameter('stream_snapshot_min_interval_sec').value,
            ),
        )
        raw_stream_out = str(self.get_parameter('stream_snapshot_output_dir').value).strip()
        if raw_stream_out:
            self._stream_out_dir = os.path.expanduser(raw_stream_out)
        else:
            self._stream_out_dir = os.path.join(self._snapshot_dir, 'stream')
        self._last_stream_save_mono = float('-inf')
        self._stream_queue: queue.Queue[tuple[str, np.ndarray]] | None = None
        self._stream_writer_thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._snap_bgr: np.ndarray | None = None
        self._snap_conf: float = 0.0
        self._snap_corners: np.ndarray | None = None

        self._segmented_throw_mode = bool(self.get_parameter('segmented_throw_mode').value)
        self._throw_segment_max_frames = max(3, _scalar_int('throw_segment_max_frames', self.get_parameter('throw_segment_max_frames').value))
        self._debug_throw_pipeline = bool(self.get_parameter('debug_throw_pipeline').value)
        self._throw_batch_topic = str(self.get_parameter('throw_batch_topic').value).strip() or '/bean_bag_throw_batch'
        self._seg_lock = threading.Lock()
        self._segment_recording = False
        # (stamp_ns, bgr) while recording a throw
        self._segment_buffer: list[tuple[int, np.ndarray]] = []

        model_path = self.get_parameter('model_path').value
        if not model_path or not str(model_path).strip():
            share = self._package_share_dir()
            model_path = os.path.join(share, 'models', 'yolo26n.pt')
        model_path = os.path.expanduser(str(model_path))

        if not os.path.isfile(model_path):
            self.get_logger().error(
                f'Model weights not found at {model_path!r}. Set param model_path or place '
                'yolo26n.pt (or your .pt) under share/bean_bag_tracker/models/.'
            )
            raise FileNotFoundError(model_path)

        bad = _ultralytics_checkpoint_hint(model_path)
        if bad:
            self.get_logger().error(bad)
            raise ValueError(bad)

        try:
            self._model = YOLO(model_path)
        except AttributeError as e:
            err = str(e).lower()
            if 'ordereddict' in err and 'float' in err:
                self.get_logger().error(
                    f'{model_path!r}: Ultralytics failed to load (often wrong checkpoint format). '
                    'See previous log line if checkpoint preflight ran; otherwise verify the file is '
                    'official ``yolo26n.pt`` or an Ultralytics-trained ``best.pt``.'
                )
            raise
        self.get_logger().info(f'Loaded YOLO weights from {model_path!r} (device={self._device!r})')
        if self._timing_interval > 0.0:
            self.get_logger().info(
                f'Stdout timing every {self._timing_interval:g}s: DECODE/INFER/POSTPROCESS/PUBLISH (ms) '
                f'and publish_hz for {self.get_parameter("detection_topic").value!r}. '
                'Set timing_stats_interval_sec:=0 to disable.',
            )

        self._pub = self.create_publisher(
            Image,
            self.get_parameter('detection_topic').value,
            qos_profile=_NN_IMAGE_QOS,
        )
        self._pub_throw_batch = self.create_publisher(
            Float64MultiArray,
            self._throw_batch_topic,
            10,
        )
        # BEST_EFFORT matches RealSense ``image_raw``; avoids RELIABLE backlog vs a fast camera.
        self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self._color_cb,
            qos_profile=_NN_IMAGE_QOS,
        )

        if self._segmented_throw_mode or self._enable_y_snapshot:
            self._stdin_cmd_thread = threading.Thread(target=self._stdin_command_loop, daemon=True)
            self._stdin_cmd_thread.start()
            if self._segmented_throw_mode:
                self.get_logger().info(
                    f'Segmented throw: type b + Enter to start recording, e + Enter to stop; '
                    f'middle-3 YOLO -> {self._throw_batch_topic!r}'
                )
            if self._enable_y_snapshot:
                self.get_logger().info(
                    f'Y-snapshot: type y + Enter to save a labeled PNG under {self._snapshot_dir!r}'
                )
        if self._stream_snapshots_enabled:
            qsz = max(
                1,
                _scalar_int(
                    'stream_write_queue_size',
                    self.get_parameter('stream_write_queue_size').value,
                ),
            )
            self._stream_queue = queue.Queue(maxsize=qsz)
            self._stream_writer_thread = threading.Thread(target=self._stream_writer_loop, daemon=True)
            self._stream_writer_thread.start()
            self.get_logger().info(
                f'Stream snapshots: conf>={self._stream_min_conf:g} -> {self._stream_out_dir!r} '
                f'(min_interval_sec={self._stream_interval:g}, async write queue={qsz})',
            )

    def _stream_writer_loop(self) -> None:
        q = self._stream_queue
        if q is None:
            return
        while rclpy.ok():
            try:
                out_path, vis = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if not cv2.imwrite(out_path, vis):
                    self.get_logger().error(f'cv2.imwrite failed for stream snapshot {out_path!r}')
            except cv2.error as e:
                self.get_logger().error(f'OpenCV imwrite error (stream worker): {e}')

    def _enqueue_stream_png(self, out_path: str, vis: np.ndarray) -> None:
        q = self._stream_queue
        if q is None:
            try:
                if not cv2.imwrite(out_path, vis):
                    self.get_logger().error(f'cv2.imwrite failed for stream snapshot {out_path!r}')
            except cv2.error as e:
                self.get_logger().error(f'OpenCV imwrite error (stream): {e}')
            return
        item = (out_path, vis)
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _maybe_save_stream_snapshot(
        self,
        bgr: np.ndarray,
        conf: float,
        corners: np.ndarray | None,
    ) -> None:
        if not self._stream_snapshots_enabled or conf + 1e-9 < self._stream_min_conf:
            return
        now = time.monotonic()
        if self._stream_interval > 0.0 and (now - self._last_stream_save_mono) < self._stream_interval:
            return
        self._last_stream_save_mono = now
        try:
            os.makedirs(self._stream_out_dir, exist_ok=True)
        except OSError as e:
            self.get_logger().error(f'Cannot create stream snapshot directory {self._stream_out_dir!r}: {e}')
            return
        vis = _render_labeled_snapshot(bgr, conf, corners)
        out_path = os.path.join(
            self._stream_out_dir,
            f'nn_stream_{time.time_ns()}_{conf:.4f}.png',
        )
        self._enqueue_stream_png(out_path, vis)

    def _stdin_command_loop(self) -> None:
        sys.stdout.write('Commands: b+Enter=start throw segment, e+Enter=end+infer middle-3, y+Enter=PNG snapshot\n')
        sys.stdout.flush()
        while rclpy.ok():
            try:
                line = input()
            except EOFError:
                return
            s = line.strip().casefold()
            if s == 'b' and self._segmented_throw_mode:
                with self._seg_lock:
                    self._segment_buffer.clear()
                    self._segment_recording = True
                self.get_logger().info('segment: recording started (e+Enter to stop, middle-3 YOLO on stop)')
            elif s == 'e' and self._segmented_throw_mode:
                self._finalize_throw_segment()
            elif s == 'y' and self._enable_y_snapshot:
                self._save_y_snapshot()

    def _finalize_throw_segment(self) -> None:
        with self._seg_lock:
            if not self._segment_recording:
                self.get_logger().warn('segment: not recording; ignored')
                return
            buf = list(self._segment_buffer)
            self._segment_buffer.clear()
            self._segment_recording = False
        n = len(buf)
        if n < 3:
            self.get_logger().error(
                f'segment: need at least 3 frames, got {n}; not publishing batch',
            )
            return
        start = (n - 3) // 2
        h0, w0 = buf[0][1].shape[:2]
        triple: list[tuple[int, np.ndarray]] = []
        for k in range(3):
            idx = start + k
            stamp_ns, bgr = buf[idx]
            t_decode0 = time.perf_counter()
            pkwargs: dict[str, Any] = {
                'source': bgr,
                'imgsz': int(self._inf_size),
                'conf': float(self._conf_thr),
                'device': self._device,
                'verbose': False,
                'half': False,
                'max_det': 1,
            }
            if self._class_ids:
                pkwargs['classes'] = self._class_ids
            results = self._model.predict(**pkwargs)
            t_post0 = time.perf_counter()
            infer_ms = (t_post0 - t_decode0) * 1000.0
            row, snap_conf, snap_corners = self._detection_row_from_results(results, h0, w0)
            t_end = time.perf_counter()
            post_ms = (t_end - t_post0) * 1000.0
            if self._debug_throw_pipeline:
                tsec, tnsec = stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000
                self.get_logger().info(
                    f'[nn_throw_debug] middle k={k} buffer_idx={idx} N={n} '
                    f'stamp={tsec}.{tnsec:09d} infer_ms={infer_ms:.2f} post_ms={post_ms:.2f}',
                )
                self.get_logger().info(
                    f'[nn_throw_debug]   det_row[9]={np.asarray(row, dtype=np.float32).ravel()[:9].tolist()}',
                )
            if self._enable_y_snapshot:
                with self._frame_lock:
                    self._snap_bgr = bgr.copy()
                    self._snap_conf = float(snap_conf)
                    self._snap_corners = None if snap_corners is None else snap_corners.copy()
            if self._stream_snapshots_enabled:
                self._maybe_save_stream_snapshot(bgr, snap_conf, snap_corners)
            triple.append((stamp_ns, row))
        if self._debug_throw_pipeline:
            self.get_logger().info(
                f'[nn_throw_debug] selected middle-3: start_idx={start} of N={n} '
                f'stamps_ns={[t[0] for t in triple]}',
            )
        try:
            batch = pack_throw_batch(w0, h0, triple)
        except Exception as exc:
            self.get_logger().error(f'segment: pack batch failed: {exc}')
            return
        if self._debug_throw_pipeline:
            for i, (st_ns, r) in enumerate(triple):
                tsec, tn = st_ns // 1_000_000_000, st_ns % 1_000_000_000
                self.get_logger().info(
                    f'[nn_throw_debug] to_bag_sense i={i} stamp={tsec}.{tn:09d} row={r.ravel()[:9].tolist()}',
                )
        self._pub_throw_batch.publish(batch)
        self.get_logger().info(
            f'segment: published {self._throw_batch_topic!r} (middle-3, N={n} buffered)',
        )

    def _detection_row_from_results(
        self,
        results: Any,
        h0: int,
        w0: int,
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        row = np.zeros((1, 9), dtype=np.float32)
        snap_conf = 0.0
        snap_corners: np.ndarray | None = None
        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = r.boxes.conf.cpu().numpy().astype(np.float32)
                if self._stream_snapshots_enabled:
                    sel = confs >= (self._stream_min_conf - 1e-9)
                else:
                    sel = np.ones(len(confs), dtype=bool)
                if np.any(sel):
                    masked = np.where(sel, confs, -1.0)
                    idx = int(np.argmax(masked))
                    best = _clip_xyxy(xyxy[idx], h0, w0)
                    corners = _xyxy_to_corners_tl_tr_br_bl(best)
                    flat = corners.reshape(8).astype(np.float32)
                    row = np.concatenate([[float(confs[idx])], flat]).reshape(1, 9).astype(np.float32)
                    snap_conf = float(confs[idx])
                    snap_corners = corners.copy()
        return row, snap_conf, snap_corners

    def _save_y_snapshot(self) -> None:
        out_dir = self._snapshot_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            self.get_logger().error(f'Cannot create snapshot directory {out_dir!r}: {e}')
            return

        with self._frame_lock:
            if self._snap_bgr is None:
                self.get_logger().warn('No camera frame received yet; skipping PNG save.')
                return
            bgr = self._snap_bgr.copy()
            conf = float(self._snap_conf)
            corners = None if self._snap_corners is None else self._snap_corners.copy()

        vis = _render_labeled_snapshot(bgr, conf, corners)
        out_path = os.path.join(out_dir, f'nn_label_{time.time_ns()}.png')
        try:
            if not cv2.imwrite(out_path, vis):
                self.get_logger().error(f'cv2.imwrite failed for {out_path!r}')
                return
        except cv2.error as e:
            self.get_logger().error(f'OpenCV imwrite error: {e}')
            return
        self.get_logger().info(f'Saved labeled snapshot: {out_path}')

    def _package_share_dir(self) -> str:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory('bean_bag_tracker')

    def _timing_accum_and_maybe_print(
        self,
        decode_ms: float,
        infer_ms: float,
        post_ms: float,
        pub_ms: float,
    ) -> None:
        """Stdout: rolling averages and publish rate for ``detection_topic``."""
        if self._timing_interval <= 0.0:
            return
        self._timing_frame_count += 1
        self._timing_pub_count += 1
        self._timing_sum_decode += decode_ms
        self._timing_sum_infer += infer_ms
        self._timing_sum_post += post_ms
        self._timing_sum_pub += pub_ms
        now = time.monotonic()
        wall_s = now - self._timing_last_print_mono
        if wall_s < self._timing_interval:
            return
        n = self._timing_frame_count
        topic = self.get_parameter('detection_topic').value
        hz = self._timing_pub_count / wall_s if wall_s > 0.0 else 0.0
        if n > 0:
            ad = self._timing_sum_decode / n
            ai = self._timing_sum_infer / n
            ap = self._timing_sum_post / n
            ab = self._timing_sum_pub / n
        else:
            ad = ai = ap = ab = 0.0
        sys.stdout.write(
            '[bean_bag_nn_detector timing] '
            f'window={wall_s * 1000.0:.0f}ms frames={n} publishes={self._timing_pub_count} '
            f'→ {topic!r} publish_hz={hz:.2f}\n'
            '  state avg_ms (this window): '
            f'DECODE={ad:.2f} INFER={ai:.2f} POSTPROCESS={ap:.2f} PUBLISH={ab:.2f}\n'
            '  state last_ms (final frame): '
            f'DECODE={decode_ms:.2f} INFER={infer_ms:.2f} POSTPROCESS={post_ms:.2f} PUBLISH={pub_ms:.2f}\n'
        )
        sys.stdout.flush()
        self._timing_last_print_mono = now
        self._timing_frame_count = 0
        self._timing_pub_count = 0
        self._timing_sum_decode = 0.0
        self._timing_sum_infer = 0.0
        self._timing_sum_post = 0.0
        self._timing_sum_pub = 0.0

    def _color_cb(self, msg: Image) -> None:
        if not rclpy.ok():
            return
        t_decode0 = time.perf_counter()
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h0, w0 = bgr.shape[:2]
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds

        if self._segmented_throw_mode:
            with self._seg_lock:
                if self._segment_recording:
                    self._segment_buffer.append((stamp_ns, bgr.copy()))
                    while len(self._segment_buffer) > self._throw_segment_max_frames:
                        self._segment_buffer.pop(0)
                    return

        if self._max_fps > 0.0:
            now = time.monotonic()
            min_dt = 1.0 / self._max_fps
            if now - self._last_infer_mono < min_dt:
                return
            self._last_infer_mono = now

        yolo_conf = float(self._conf_thr)
        if self._stream_snapshots_enabled:
            yolo_conf = float(min(self._conf_thr, self._stream_min_conf))
        kwargs: dict[str, Any] = {
            'source': bgr,
            'imgsz': int(self._inf_size),
            'conf': yolo_conf,
            'device': self._device,
            'verbose': False,
            'half': False,
            'max_det': 1,
        }
        if self._class_ids:
            kwargs['classes'] = self._class_ids
        t_infer0 = time.perf_counter()

        try:
            results = self._model.predict(**kwargs)
        except AttributeError as e:
            err = str(e).lower()
            if 'ordereddict' in err and 'float' in err:
                self.get_logger().error(
                    'YOLO inference failed with an OrderedDict/float error. Typical causes: (1) '
                    '``ultralytics`` is too old for this checkpoint — run ``pip3 install -U ultralytics``; '
                    '(2) the ``.pt`` file is not an Ultralytics YOLO checkpoint (e.g. raw state_dict). '
                    'Train or export with Ultralytics and use that ``.pt``.'
                )
            raise
        t_post0 = time.perf_counter()
        row, snap_conf, snap_corners = self._detection_row_from_results(results, h0, w0)

        if self._stream_snapshots_enabled:
            self._maybe_save_stream_snapshot(bgr, snap_conf, snap_corners)

        if self._enable_y_snapshot:
            with self._frame_lock:
                self._snap_bgr = bgr.copy()
                self._snap_conf = snap_conf
                self._snap_corners = snap_corners

        t_pub0 = time.perf_counter()
        row = np.ascontiguousarray(row, dtype=np.float32)
        out_msg = self._bridge.cv2_to_imgmsg(row, encoding='32FC1')
        out_msg.header = msg.header
        try:
            self._pub.publish(out_msg)
        except Exception:
            if rclpy.ok():
                raise
        t_end = time.perf_counter()
        decode_ms = (t_infer0 - t_decode0) * 1000.0
        infer_ms = (t_post0 - t_infer0) * 1000.0
        post_ms = (t_pub0 - t_post0) * 1000.0
        pub_ms = (t_end - t_pub0) * 1000.0
        self._timing_accum_and_maybe_print(decode_ms, infer_ms, post_ms, pub_ms)


def _sanitize_ros_args_for_rcl(argv: list[str] | None) -> list[str] | None:
    """Drop ``-p`` / ``--param`` overrides with an empty value (``name:=``). rcl rejects those."""
    if argv is None:
        return None
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        # Glued form, e.g. ``-pmodel_path:=`` (invalid) or ``-pmodel_path:=/path`` (ok).
        if tok.startswith('-p') and tok not in ('-p', '--param') and ':=' in tok:
            body = tok[2:]
            _, _, rhs = body.partition(':=')
            if rhs == '':
                i += 1
                continue
            out.append(tok)
            i += 1
            continue
        if tok in ('-p', '--param') and i + 1 < n:
            spec = argv[i + 1]
            if ':=' in spec:
                _, _, rhs = spec.partition(':=')
                if rhs == '':
                    i += 2
                    continue
            out.append(tok)
            out.append(argv[i + 1])
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(args: Any = None) -> None:
    init_argv = list(sys.argv) if args is None else list(args)
    rclpy.init(args=_sanitize_ros_args_for_rcl(init_argv))
    node: BeanBagNnDetector | None = None
    try:
        node = BeanBagNnDetector()
    except (FileNotFoundError, ValueError):
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
