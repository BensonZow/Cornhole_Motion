#!/usr/bin/env python3
"""Launch RealSense (optional), capture every Nth color frame to training/unlabelled."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# #region agent log
_DEBUG_LOG = '/home/cornholio/ros2_jazzy/.cursor/debug-063e7b.log'


def _agent_dbg(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    run_id: str = 'pre',
) -> None:
    try:
        rec = {
            'sessionId': '063e7b',
            'runId': run_id,
            'hypothesisId': hypothesis_id,
            'location': location,
            'message': message,
            'data': data or {},
            'timestamp': int(time.time() * 1000),
        }
        with open(_DEBUG_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, default=str) + '\n')
    except Exception:
        pass


# #endregion


def _workspace_root_from_colcon() -> Path | None:
    cpp = os.environ.get('COLCON_PREFIX_PATH', '').split(os.pathsep)[0].strip()
    if not cpp:
        return None
    return Path(cpp).resolve().parent


class TrainUnlabelledCapture(Node):
    """Subscribe to color image; while capturing, save every ``frame_stride``-th frame."""

    def __init__(self) -> None:
        super().__init__('train_unlabelled_capture')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('reliability', 'reliable')
        self.declare_parameter('frame_stride', 10)
        self.declare_parameter('output_dir', '')
        self.declare_parameter('jpeg_quality', 88)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._capturing = False
        self._frame_index = 0
        self._saved_total = 0
        self._got_image = threading.Event()
        self._dbg_cb_count = 0

        out_param = str(self.get_parameter('output_dir').value).strip()
        if out_param:
            self._out_dir = Path(out_param).expanduser().resolve()
        else:
            root = _workspace_root_from_colcon()
            if root is None:
                self.get_logger().warn(
                    'COLCON_PREFIX_PATH unset; using ./training/unlabelled '
                    'relative to current working directory.',
                )
                root = Path.cwd()
            self._out_dir = root / 'training' / 'unlabelled'

        self._stride = max(1, int(self.get_parameter('frame_stride').value))
        self._jpeg_q = int(self.get_parameter('jpeg_quality').value)

        rel = str(self.get_parameter('reliability').value).lower()
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
            if rel == 'best_effort'
            else ReliabilityPolicy.RELIABLE,
        )
        topic = str(self.get_parameter('color_topic').value)
        self.create_subscription(Image, topic, self._cb, qos)

        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f'Saving to {self._out_dir} (every {self._stride}th frame while capturing; topic {topic})',
        )
        # #region agent log
        _agent_dbg(
            'H2',
            'train_unlabelled_capture:__init__',
            'resolved_paths',
            {
                'out_dir': str(self._out_dir),
                'colcon_prefix_path': os.environ.get('COLCON_PREFIX_PATH', ''),
                'cwd': str(Path.cwd()),
                'topic': topic,
                'reliability': rel,
            },
        )
        # #endregion

    def begin_capture(self) -> None:
        # #region agent log
        _agent_dbg('H3', 'train_unlabelled_capture:begin_capture', 'enter', {})
        # #endregion
        with self._lock:
            self._capturing = True
            self._frame_index = 0

    def end_capture(self) -> int:
        with self._lock:
            self._capturing = False
            return self._frame_index

    def _cb(self, msg: Image) -> None:
        try:
            self._got_image.set()
            self._dbg_cb_count += 1
            cap = False
            with self._lock:
                cap = self._capturing
            # #region agent log
            if self._dbg_cb_count <= 15:
                _agent_dbg(
                    'H1',
                    'train_unlabelled_capture:_cb',
                    'callback',
                    {
                        'n': self._dbg_cb_count,
                        'encoding': msg.encoding,
                        'capturing': cap,
                        'w': msg.width,
                        'h': msg.height,
                    },
                )
            # #endregion
            with self._lock:
                if not self._capturing:
                    return
                idx = self._frame_index
                self._frame_index += 1
                if idx % self._stride != 0:
                    return
                bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
            self._saved_total += 1
            seq = self._saved_total
            name = f'{stamp}_{seq:06d}.jpg'
            path = self._out_dir / name
            tmp = path.with_suffix('.partial.jpg')
            # #region agent log
            _agent_dbg(
                'H4',
                'train_unlabelled_capture:_cb',
                'save_attempt',
                {'path': str(path), 'idx': idx, 'stride': self._stride},
            )
            # #endregion
            try:
                ok = cv2.imwrite(
                    str(tmp),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q],
                )
                # #region agent log
                _agent_dbg(
                    'H4',
                    'train_unlabelled_capture:_cb',
                    'imwrite_result',
                    {
                        'ok': bool(ok),
                        'tmp_exists': tmp.exists(),
                        'tmp_size': tmp.stat().st_size if tmp.exists() else -1,
                    },
                )
                # #endregion
                if not ok:
                    self.get_logger().error(f'cv2.imwrite returned False for {path}')
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                    return
                os.replace(tmp, path)
                # #region agent log
                _agent_dbg(
                    'H4',
                    'train_unlabelled_capture:_cb',
                    'saved_ok',
                    {'path': str(path)},
                )
                # #endregion
            except OSError as e:
                self.get_logger().error(f'Failed to write {path}: {e}')
                # #region agent log
                _agent_dbg(
                    'H4',
                    'train_unlabelled_capture:_cb',
                    'save_oserror',
                    {'err': str(e), 'path': str(path)},
                )
                # #endregion
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
        except Exception as e:
            # #region agent log
            _agent_dbg(
                'H5',
                'train_unlabelled_capture:_cb',
                'exception',
                {'type': type(e).__name__, 'err': str(e)},
            )
            # #endregion
            self.get_logger().error(f'Image callback error: {e}')

    def wait_for_first_image(self, timeout_s: float = 60.0) -> bool:
        return self._got_image.wait(timeout=timeout_s)


def _spin_node(node: TrainUnlabelledCapture) -> None:
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def _stop_realsense(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--no-launch-camera',
        action='store_true',
        help='Do not spawn realsense2_camera; use an already-running driver.',
    )
    parsed, ros_argv = parser.parse_known_args(sys.argv[1:])
    rclpy.init(args=[sys.argv[0]] + ros_argv)

    ros2 = shutil.which('ros2')
    cam_proc: Optional[subprocess.Popen] = None
    if not parsed.no_launch_camera:
        if not ros2:
            sys.stderr.write('ros2 not found in PATH; use --no-launch-camera if the driver is already up.\n')
            rclpy.shutdown()
            sys.exit(1)
        cam_proc = subprocess.Popen(
            [ros2, 'launch', 'realsense2_camera', 'rs_launch.py', 'initial_reset:=true'],
        )

    node: TrainUnlabelledCapture | None = None
    spin_thr: threading.Thread | None = None
    try:
        node = TrainUnlabelledCapture()
        spin_thr = threading.Thread(target=_spin_node, args=(node,), daemon=True)
        spin_thr.start()

        if not node.wait_for_first_image(timeout_s=90.0):
            # #region agent log
            _agent_dbg(
                'H1',
                'train_unlabelled_capture:main',
                'wait_first_image_timeout',
                {'no_launch_camera': parsed.no_launch_camera},
            )
            # #endregion
            sys.stderr.write('Timed out waiting for first image; check the camera and topic.\n')
            return

        # #region agent log
        _agent_dbg(
            'H1',
            'train_unlabelled_capture:main',
            'first_image_ok',
            {'no_launch_camera': parsed.no_launch_camera},
        )
        # #endregion
        node.get_logger().info('Images received. Interactive capture controls on stdin.')
        stride = max(1, int(node.get_parameter('frame_stride').value))

        while rclpy.ok():
            print(f'Type y and Enter to start capturing every {stride}th frame.', flush=True)
            while True:
                line = sys.stdin.readline()
                if not line:
                    return
                if line.strip().lower() == 'y':
                    break

            node.begin_capture()
            print('Capturing. Press Enter to stop.', flush=True)
            if not sys.stdin.readline():
                node.end_capture()
                return

            n = node.end_capture()
            node.get_logger().info(
                f'Stopped after {n} frames in this burst (only every {stride}th is saved).',
            )

    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thr is not None and spin_thr.is_alive():
            spin_thr.join(timeout=3.0)
        if node is not None:
            node.destroy_node()
        _stop_realsense(cam_proc)
        time.sleep(0.2)


if __name__ == '__main__':
    main()
