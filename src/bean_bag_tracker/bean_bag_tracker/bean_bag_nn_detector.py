#!/usr/bin/env python3
"""YOLO / PyTorch (.pt) bean-bag detector: color image in, stamped ``sensor_msgs/Image`` out.

Loads weights via Ultralytics (e.g. YOLO26n ``yolo26n.pt``). ROS 2 ``std_msgs/Float32MultiArray`` has no
``std_msgs/Header``, so detections are published as a 1×9 row, ``encoding='32FC1'``, with the same
``header`` as the source color frame (for ``message_filters`` sync with depth). The nine floats are::

    [confidence, u0, v0, u1, v1, u2, v2, u3, v3]

Corners are axis-aligned box vertices in **original color pixel coordinates**, order:
top-left, top-right, bottom-right, bottom-left. When nothing exceeds ``confidence_threshold``,
``confidence`` is ``0.0`` and all coordinates are ``0.0``.
"""
from __future__ import annotations

import os
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
from sensor_msgs.msg import Image


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

        self._pub = self.create_publisher(
            Image, self.get_parameter('detection_topic').value, 10
        )
        self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self._color_cb,
            10,
        )

    def _package_share_dir(self) -> str:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory('bean_bag_tracker')

    def _color_cb(self, msg: Image) -> None:
        if self._max_fps > 0.0:
            now = time.monotonic()
            min_dt = 1.0 / self._max_fps
            if now - self._last_infer_mono < min_dt:
                return
            self._last_infer_mono = now

        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h0, w0 = bgr.shape[:2]

        kwargs: dict[str, Any] = {
            'source': bgr,
            'imgsz': int(self._inf_size),
            'conf': float(self._conf_thr),
            'device': self._device,
            'verbose': False,
            'half': False,
        }
        if self._class_ids:
            kwargs['classes'] = self._class_ids

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
        row = np.zeros((1, 9), dtype=np.float32)

        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = r.boxes.conf.cpu().numpy().astype(np.float32)
                idx = int(np.argmax(confs))
                best = _clip_xyxy(xyxy[idx], h0, w0)
                corners = _xyxy_to_corners_tl_tr_br_bl(best)
                flat = corners.reshape(8).astype(np.float32)
                row = np.concatenate([[float(confs[idx])], flat]).reshape(1, 9).astype(np.float32)

        row = np.ascontiguousarray(row, dtype=np.float32)
        out_msg = self._bridge.cv2_to_imgmsg(row, encoding='32FC1')
        out_msg.header = msg.header
        self._pub.publish(out_msg)


def main(args: Any = None) -> None:
    rclpy.init(args=args)
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
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
