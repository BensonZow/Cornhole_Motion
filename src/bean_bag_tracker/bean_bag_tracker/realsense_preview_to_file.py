#!/usr/bin/env python3
"""Write latest RealSense color frame to a JPEG (no GUI — works over SSH)."""

from __future__ import annotations

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class RealsensePreviewToFile(Node):
    def __init__(self) -> None:
        super().__init__('realsense_preview_to_file')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter(
            'output_path',
            '/home/cornholio/ros2_jazzy/log/realsense_preview.jpg',
        )
        self.declare_parameter('reliability', 'reliable')
        self._out = str(self.get_parameter('output_path').value)
        self._bridge = CvBridge()
        rel = str(self.get_parameter('reliability').value).lower()
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
            if rel == 'best_effort'
            else ReliabilityPolicy.RELIABLE,
        )
        topic = str(self.get_parameter('color_topic').value)
        self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(f'Writing latest frame to {self._out} (topic {topic})')

    def _cb(self, msg: Image) -> None:
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Extension must be .jpg so OpenCV picks the JPEG writer (not ``.jpg.tmp``).
        base, ext = os.path.splitext(self._out)
        tmp = f'{base}.partial{ext or ".jpg"}'
        cv2.imwrite(tmp, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        os.replace(tmp, self._out)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RealsensePreviewToFile()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
