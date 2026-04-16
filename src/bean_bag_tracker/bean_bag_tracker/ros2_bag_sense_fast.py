#!/usr/bin/env python3
import math
import os
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

# 3D trajectory debug PNGs (matplotlib Agg).
TRAJECTORY_DEBUG_PLOT_DIR = '/home/cornholio/ros2_jazzy/log'

# Purple HSV tuning (OpenCV BGR→HSV: H 0–179, S and V 0–255). Lenient band for varied lighting / saturation.
# Wider hue (violet–blue-magenta edge), lower S floor (dusty purples), wider V (shadow + slightly brighter).
PURPLE_HSV_LOWER = (122, 45, 15)
PURPLE_HSV_UPPER = (168, 255, 210)
# Square kernel edge length for open/close (e.g. 3 = gentler than 5).
PURPLE_MASK_MORPH_KERNEL_SIZE = 3


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
        self.hole_distance = self.get_parameter('hole_distance_inches').value * 0.0254  # convert to meters
        self.depth_scale = float(self.get_parameter('depth_scale').value)
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
        self.points = []              # list of (timestamp, x, y, depth) in meters
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

        # Approximate time synchronizer (slop 0.1s)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)

        # stdin thread sets _pending_keyboard_reset; this timer runs on the executor thread.
        self._keyboard_poll_timer = self.create_timer(
            0.05, self._keyboard_poll_callback, callback_group=self.callback_group)

        self.get_logger().info('Bean Bag Tracker node started')
        self.get_logger().info(
            f'Depth scale is: {self.depth_scale} m/raw_unit '
            '(ROS param depth_scale; same role as depth_sensor.get_depth_scale())'
        )

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
        centroid = self.find_purple_centroid(color_image)

        if centroid is None:
            return

        # 2. Extract 3D data
        u, v = centroid
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        depth_raw_u16 = int(depth_image[v, u])
        depth = depth_raw_u16 * self.depth_scale

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
                self._debug_first_frame_bgr = color_image.copy()

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
                        self._debug_first_frame_bgr = None

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

    def _purple_binary_mask_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """HSV dark-purple mask (8-bit) after morphology; same semantics as find_purple_centroid pre-contours."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(PURPLE_HSV_LOWER, dtype=np.uint8)
        upper = np.array(PURPLE_HSV_UPPER, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        k = max(1, int(PURPLE_MASK_MORPH_KERNEL_SIZE))
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _purple_masked_bgr(self, bgr: np.ndarray) -> np.ndarray:
        mask = self._purple_binary_mask_bgr(bgr)
        return cv2.bitwise_and(bgr, bgr, mask=mask)

    def find_purple_centroid(self, bgr_image):
        """Detect the largest purple blob and return its centroid (u, v) or None."""
        mask = self._purple_binary_mask_bgr(bgr_image)

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

    def _backtrack_depth_scale_line(self) -> str:
        return (
            f'  Depth scale is: {self.depth_scale} m/raw_unit  '
            f'(depth_m = uint16_depth * depth_scale; cf. depth_sensor.get_depth_scale())'
        )

    def _backtrack_sample_point_lines(self, t0: float) -> list[str]:
        lines: list[str] = [
            self._backtrack_depth_scale_line(),
            '  samples (t, x, y_cam, depth_m from uint16 * depth_scale):',
        ]
        for i, (t_abs, xi, yi, di) in enumerate(self.points):
            tr = t_abs - t0
            lines.append(
                f'    pt[{i}] t_ros={t_abs:.6f}s  t_rel={tr:.6f}s  '
                f'x={xi:.6f}m  y_cam={yi:.6f}m  depth={di:.6f}m'
            )
        return lines

    def _emit_bag_distance_backtrack(self, lines: list[str]) -> None:
        """Single INFO log per completed 3-point bag cycle (distance math only)."""
        self.get_logger().info('\n'.join(lines))

    def _save_trajectory_debug_plot_3d(
        self,
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        depths: np.ndarray,
        vx: float,
        x0: float,
        a_half: float,
        vy: float,
        y0: float,
        vz: float,
        z0: float,
        t_land: float | None,
        x_land: float | None,
        y_land: float | None,
        depth_land: float | None,
        zh: float,
        first_bgr: np.ndarray | None,
    ) -> None:
        """Save dual-tile PNG: 3D trajectory (mpl X=x, Y=depth, Z=y_cam) + optional purple-mask frame."""
        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.get_logger().warn('matplotlib not installed; skipping 3d debug plot PNG')
            return

        out_dir = TRAJECTORY_DEBUG_PLOT_DIR
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            self.get_logger().warn(f'Could not create trajectory plot dir {out_dir!r}: {e}')
            return

        xs_c = np.asarray(xs, dtype=float).copy()
        ys_c = np.asarray(ys, dtype=float).copy()
        depths_c = np.asarray(depths, dtype=float).copy()

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.scatter([0.0], [0.0], [0.0], c='k', s=80, marker='^', label='camera')
        ax.plot(xs_c, depths_c, ys_c, 'b-', alpha=0.5, linewidth=1)
        ax.scatter(xs_c, depths_c, ys_c, c='blue', s=35, label='3 samples')
        ax.scatter(
            [xs_c[0]], [depths_c[0]], [ys_c[0]],
            c='orange', s=120, marker='o', label='bag frame 1 (earliest)',
        )

        arc_x = arc_y_cam = arc_depth = None
        if t_land is not None and x_land is not None and y_land is not None and depth_land is not None:
            if math.isfinite(t_land):
                n = max(25, int(40 * (1.0 + min(abs(t_land), 5.0))))
                tt = np.linspace(0.0, float(t_land), n)
                arc_x = np.polyval(np.array([vx, x0]), tt)
                arc_y_cam = np.polyval(np.array([a_half, vy, y0]), tt)
                arc_depth = np.polyval(np.array([vz, z0]), tt)
                ax.plot(arc_x, arc_depth, arc_y_cam, 'g--', alpha=0.7, linewidth=1.2, label='fit arc to t_land')
            ax.scatter(
                [x_land], [depth_land], [y_land],
                c='red', s=140, marker='*', label='predicted landing',
            )

        x_parts = [xs_c, np.array([0.0])]
        depth_parts = [depths_c, np.array([0.0])]
        y_parts = [ys_c, np.array([0.0])]
        if arc_x is not None:
            x_parts.append(arc_x)
            depth_parts.append(arc_depth)
            y_parts.append(arc_y_cam)
        if x_land is not None:
            x_parts.append(np.array([float(x_land)]))
            depth_parts.append(np.array([float(depth_land)]))
            y_parts.append(np.array([float(y_land)]))

        x_all = np.concatenate(x_parts)
        depth_all = np.concatenate(depth_parts)
        y_all = np.concatenate(y_parts)

        x_abs = float(np.max(np.abs(x_all)))
        if x_abs < 1e-6:
            x_abs = 0.1
        x_pad = max(0.02, 0.1 * x_abs)
        ax.set_xlim(-x_abs - x_pad, x_abs + x_pad)

        dmin = float(np.min(depth_all))
        dmax = float(np.max(depth_all))
        if dmax - dmin < 1e-6:
            dmin -= 0.05
            dmax += 0.05
        else:
            dm = 0.1 * (dmax - dmin)
            dmin -= dm
            dmax += dm
        ax.set_ylim(dmin, dmax)

        y_hi = float(np.max(y_all))
        if y_hi < 1e-6:
            y_hi = 0.05
        y_pad = max(0.02, 0.05 * y_hi)
        ax.set_zlim(0.0, y_hi + y_pad)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('depth (m)')
        ax.set_zlabel('y_cam (m)')
        ax.set_title('Trajectory (camera frame: floor x–depth, up y_cam)')
        ax.legend(loc='upper left', fontsize=8)

        ax_img = fig.add_subplot(1, 2, 2)
        if first_bgr is not None:
            rgb = cv2.cvtColor(self._purple_masked_bgr(first_bgr), cv2.COLOR_BGR2RGB)
            ax_img.imshow(rgb)
            ax_img.set_title('Frame 1 (purple mask)')
            ax_img.axis('off')
        else:
            ax_img.text(0.5, 0.5, 'no frame captured', ha='center', va='center', transform=ax_img.transAxes)
            ax_img.axis('off')

        fig.text(0.02, 0.02, f'z_hole (range ref) = {zh:.4f} m', fontsize=8)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f'trajectory_debug_{time.time_ns()}.png')
        try:
            fig.savefig(out_path, dpi=120)
        except OSError as e:
            self.get_logger().warn(f'Could not write trajectory plot {out_path!r}: {e}')
        finally:
            plt.close(fig)

        self.get_logger().info(f'Wrote 3d trajectory debug plot: {out_path}')

    def compute_and_publish(self) -> tuple[bool, bool]:
        """Fit trajectory, predict landing. Returns (published, arm_keyboard).

        Linear ``x(t)``, linear ``depth(t)`` for ``t_land`` where ``depth(t_land)=z_hole``;
        quadratic ``y(t)`` (gravity along image vertical); ``y_land`` evaluated at ``t_land``.
        Miss: ``hypot(x_land, depth_land - z_hole)``; angle ``atan2(d_depth, x_land)``.

        ``published`` is True only if a message was sent. ``arm_keyboard`` is True
        to enter WAIT + stdin (all finished cycles except compute aborted on cooldown).
        """
        now = time.monotonic()
        self.points.sort(key=lambda p: p[0])
        t0 = self.points[0][0]

        if now - self._last_publish_mono < self.min_publish_interval:
            lines = [
                '--- bag distance backtrack ---',
                self._backtrack_depth_scale_line(),
                'outcome: publish_cooldown (no trajectory message; landing math not run)',
                f'  elapsed_since_last_publish_s={now - self._last_publish_mono:.6f}  '
                f'min_publish_interval_s={self.min_publish_interval:.6f}',
            ]
            lines.extend(self._backtrack_sample_point_lines(t0)[1:])
            self._emit_bag_distance_backtrack(lines)
            return (False, False)

        first_bgr = (
            self._debug_first_frame_bgr.copy()
            if self._debug_first_frame_bgr is not None
            else None
        )

        times = np.array([p[0] - t0 for p in self.points])
        xs = np.array([p[1] for p in self.points])
        ys = np.array([p[2] for p in self.points])
        depths = np.array([p[3] for p in self.points])

        coeffs_x = np.polyfit(times, xs, 1)
        coeffs_y = np.polyfit(times, ys, 2)
        coeffs_depth = np.polyfit(times, depths, 1)

        vx, x0 = coeffs_x
        a_half, vy, y0 = coeffs_y
        vz, z0 = coeffs_depth

        zh = float(self.hole_distance)
        lines = [
            '--- bag distance backtrack ---',
        ]
        lines.extend(self._backtrack_sample_point_lines(t0))
        lines.append(
            f'  polyfit (t_rel from first sample): '
            f'x(t)={vx:.6f}*t+{x0:.6f}  '
            f'y_cam(t)={a_half:.6f}*t^2+{vy:.6f}*t+{y0:.6f}  '
            f'depth(t)={vz:.6f}*t+{z0:.6f}'
        )
        lines.append(
            f'  t_land from depth to z_hole: depth(t_land)=z_hole={zh:.6f}m  '
            f'=> vz*t_land+z0=zh  => t_land=(zh-z0)/vz'
        )

        if abs(vz) < 1e-6:
            lines.append(f'  vz={vz:.6e}  => |vz|<1e-6, t_land undefined (division by zero)')
            lines.append('  outcome: vz_near_zero (no /bean_bag_trajectory publish)')
            self._emit_bag_distance_backtrack(lines)
            self._save_trajectory_debug_plot_3d(
                xs=xs,
                ys=ys,
                depths=depths,
                vx=vx,
                x0=x0,
                a_half=a_half,
                vy=vy,
                y0=y0,
                vz=vz,
                z0=z0,
                t_land=None,
                x_land=None,
                y_land=None,
                depth_land=None,
                zh=zh,
                first_bgr=first_bgr,
            )
            return (False, True)

        t_land = (zh - z0) / vz
        x_land = vx * t_land + x0
        y_land = float(np.polyval(np.array([a_half, vy, y0]), t_land))
        depth_land = vz * t_land + z0
        dx = x_land
        d_depth = depth_land - zh
        distance_m = math.hypot(dx, d_depth)
        angle = math.atan2(d_depth, dx)

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
            f'  depth_land = vz*t_land+z0 = {vz:.6f}*{t_land:.6f}+{z0:.6f} = {depth_land:.6f} m'
        )
        lines.append(f'  z_hole (range ref, m) = {zh:.6f}  => d_depth = depth_land - z_hole = {d_depth:.6f} m')
        lines.append(
            f'  miss: dx=x_land={dx:.6f} m, d_depth={d_depth:.6f} m  '
            f'(angle_rad=atan2(d_depth,dx))'
        )
        lines.append(
            f'  distance_m = hypot(dx,d_depth) = sqrt({dx:.6f}^2+{d_depth:.6f}^2) = {distance_m:.6f} m'
        )
        lines.append(f'  angle_rad = atan2(d_depth,dx) = {angle:.6f}')
        lines.append(f'  max_publish_distance_m = {self.max_publish_distance_m:.6f}')

        topic = self.get_parameter('result_topic').value
        if distance_m > self.max_publish_distance_m:
            lines.append(
                '  outcome: withheld_over_distance_cap '
                f'(distance_m > max_publish_distance_m; no publish on {topic!r})'
            )
            self._emit_bag_distance_backtrack(lines)
            self._save_trajectory_debug_plot_3d(
                xs=xs,
                ys=ys,
                depths=depths,
                vx=vx,
                x0=x0,
                a_half=a_half,
                vy=vy,
                y0=y0,
                vz=vz,
                z0=z0,
                t_land=t_land,
                x_land=x_land,
                y_land=y_land,
                depth_land=depth_land,
                zh=zh,
                first_bgr=first_bgr,
            )
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
        self._save_trajectory_debug_plot_3d(
            xs=xs,
            ys=ys,
            depths=depths,
            vx=vx,
            x0=x0,
            a_half=a_half,
            vy=vy,
            y0=y0,
            vz=vz,
            z0=z0,
            t_land=t_land,
            x_land=x_land,
            y_land=y_land,
            depth_land=depth_land,
            zh=zh,
            first_bgr=first_bgr,
        )

        return (True, True)

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