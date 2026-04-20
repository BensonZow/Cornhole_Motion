"""Shared trajectory fit, landing prediction, debug plot for bag sense nodes.

Used by ``ros2_bag_sense_fast`` and ``ros2_bag_sense_manual_depth``. Keep landing
math, publish format, and plot logic in sync across callers.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray

# 3D trajectory debug PNGs (matplotlib Agg).
TRAJECTORY_DEBUG_PLOT_DIR = '/home/cornholio/ros2_jazzy/log'

# (t_ros_s, x_m, y_cam_m, depth_m, u_px, v_px, bbox_area_px)
TrajectoryPoint = tuple[float, float, float, float, int, int, float]


def xy_depth_to_uv(
    x: float, y: float, depth: float, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int] | None:
    """Pinhole inverse of ``x=(u-cx)*d/fx`` → pixel (u, v) for overlay."""
    if depth <= 1e-9:
        return None
    u = int(round(float(x) * float(fx) / float(depth) + float(cx)))
    v = int(round(float(y) * float(fy) / float(depth) + float(cy)))
    return (u, v)


def backtrack_depth_scale_line(depth_scale: float) -> str:
    return (
        f'  Depth scale is: {depth_scale} m/raw_unit  '
        f'(depth_m = uint16_depth * depth_scale; cf. depth_sensor.get_depth_scale())'
    )


def backtrack_sample_point_lines(points: list[TrajectoryPoint], t0: float, depth_scale: float) -> list[str]:
    lines: list[str] = [
        backtrack_depth_scale_line(depth_scale),
        '  samples (t, x, y_cam, depth_m from uint16 * depth_scale):',
    ]
    for i, (t_abs, xi, yi, di, ui, vi, a_px) in enumerate(points):
        tr = t_abs - t0
        lines.append(
            f'    pt[{i}] t_ros={t_abs:.6f}s  t_rel={tr:.6f}s  '
            f'x={xi:.6f}m  y_cam={yi:.6f}m  depth={di:.6f}m  depth_sample_uv=({ui},{vi})  '
            f'bbox_area_px={a_px:.1f}'
        )
    return lines


def save_trajectory_debug_plot_3d(
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
    sample_us: np.ndarray,
    sample_vs: np.ndarray,
    bbox_areas_px: np.ndarray,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    out_dir: str,
    log_warn: Callable[[str], None],
    log_info: Callable[[str], None],
    first_frame_panel3_title: str = 'Frame 1 raw color',
    bbox_area_caption: str = 'NN bbox area (axis-aligned, px^2):',
) -> None:
    """Save 2×3 PNG: 3D trajectory, frame-1 overlays, reference image; y_cam vs x; y_cam vs depth (z)."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log_warn('matplotlib not installed; skipping 3d debug plot PNG')
        return

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        log_warn(f'Could not create trajectory plot dir {out_dir!r}: {e}')
        return

    xs_c = np.asarray(xs, dtype=float).copy()
    ys_c = np.asarray(ys, dtype=float).copy()
    depths_c = np.asarray(depths, dtype=float).copy()
    bu = np.asarray(sample_us, dtype=int).ravel()
    bv = np.asarray(sample_vs, dtype=int).ravel()
    b_area = np.asarray(bbox_areas_px, dtype=float).ravel()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 3, 1, projection='3d')

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

    ax_img = fig.add_subplot(2, 3, 2)
    if first_bgr is not None:
        vis_bgr = first_bgr.copy()
        h_img, w_img = vis_bgr.shape[:2]
        rad = max(6, min(w_img, h_img) // 80)
        if fx is not None and fy is not None and cx is not None and cy is not None and len(xs_c) > 0:
            uv = xy_depth_to_uv(
                float(xs_c[0]),
                float(ys_c[0]),
                float(depths_c[0]),
                float(fx),
                float(fy),
                float(cx),
                float(cy),
            )
            if uv is not None:
                u, v = uv
                if 0 <= u < w_img and 0 <= v < h_img:
                    ring_bgr = (0, 165, 255)
                    cv2.circle(vis_bgr, (u, v), rad, ring_bgr, 2)
                    cv2.circle(vis_bgr, (u, v), 2, (255, 255, 255), -1)
                    cv2.putText(
                        vis_bgr,
                        'pt1 3D',
                        (min(u + 6, w_img - 56), max(v - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        if len(bu) > 0 and len(bv) > 0:
            blob_r = max(3, min(w_img, h_img) // 120)
            cyan_bgr = (255, 255, 0)
            ub, vb = int(bu[0]), int(bv[0])
            if 0 <= ub < w_img and 0 <= vb < h_img:
                cv2.circle(vis_bgr, (ub, vb), blob_r, cyan_bgr, -1, lineType=cv2.LINE_AA)
                cv2.circle(vis_bgr, (ub, vb), 1, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        if len(b_area) > 0:
            a0 = float(b_area[0])
            cv2.putText(
                vis_bgr,
                f'bbox area={a0:.0f} px^2',
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        ax_img.imshow(rgb)
        img_title = 'Frame 1 (earliest): depth sample + 3D reproject'
        if len(b_area) > 0:
            img_title += f' | bbox A={float(b_area[0]):.0f} px^2'
        ax_img.set_title(img_title)
        ax_img.axis('off')
    else:
        ax_img.text(0.5, 0.5, 'no frame captured', ha='center', va='center', transform=ax_img.transAxes)
        ax_img.axis('off')

    ax_grad = fig.add_subplot(2, 3, 3)
    if first_bgr is not None:
        ax_grad.imshow(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB))
        ax_grad.set_title(first_frame_panel3_title)
    else:
        ax_grad.text(0.5, 0.5, 'no frame', ha='center', va='center', transform=ax_grad.transAxes)
        ax_grad.set_title(first_frame_panel3_title)
    ax_grad.axis('off')

    ax_yx = fig.add_subplot(2, 3, 4)
    ax_yx.scatter([0.0], [0.0], c='k', s=50, marker='^', label='camera', zorder=5)
    ax_yx.plot(xs_c, ys_c, 'b-', alpha=0.5, linewidth=1, label='samples')
    ax_yx.scatter(xs_c, ys_c, c='blue', s=25, zorder=4)
    ax_yx.scatter([xs_c[0]], [ys_c[0]], c='orange', s=80, marker='o', zorder=6)
    for i in range(len(xs_c)):
        if i < len(b_area):
            ax_yx.annotate(
                f'{float(b_area[i]):.0f}px^2',
                (float(xs_c[i]), float(ys_c[i])),
                textcoords='offset points',
                xytext=(6, 6),
                fontsize=7,
                color='navy',
            )
    if arc_x is not None:
        ax_yx.plot(arc_x, arc_y_cam, 'g--', alpha=0.7, linewidth=1.2, label='fit arc')
    if x_land is not None and y_land is not None:
        ax_yx.scatter([float(x_land)], [float(y_land)], c='red', s=100, marker='*', zorder=6, label='landing')
    ax_yx.set_xlabel('x (m)')
    ax_yx.set_ylabel('y_cam (m)')
    ax_yx.set_title('y_cam vs x')
    ax_yx.set_xlim(-x_abs - x_pad, x_abs + x_pad)
    ax_yx.set_ylim(0.0, y_hi + y_pad)
    ax_yx.grid(True, alpha=0.3)
    ax_yx.legend(loc='upper left', fontsize=7)

    ax_ydepth = fig.add_subplot(2, 3, 5)
    ax_ydepth.plot(depths_c, ys_c, 'b-', alpha=0.5, linewidth=1, label='samples')
    ax_ydepth.scatter(depths_c, ys_c, c='blue', s=25, zorder=4)
    ax_ydepth.scatter([depths_c[0]], [ys_c[0]], c='orange', s=80, marker='o', zorder=6)
    for i in range(len(depths_c)):
        if i < len(b_area):
            ax_ydepth.annotate(
                f'{float(b_area[i]):.0f}px^2',
                (float(depths_c[i]), float(ys_c[i])),
                textcoords='offset points',
                xytext=(6, 6),
                fontsize=7,
                color='navy',
            )
    if arc_depth is not None:
        ax_ydepth.plot(arc_depth, arc_y_cam, 'g--', alpha=0.7, linewidth=1.2, label='fit arc')
    if depth_land is not None and y_land is not None:
        ax_ydepth.scatter(
            [float(depth_land)], [float(y_land)], c='red', s=100, marker='*', zorder=6, label='landing'
        )
    ax_ydepth.axvline(float(zh), color='gray', linestyle=':', linewidth=1.2, label='z_hole')
    ax_ydepth.set_xlabel('depth z (m)')
    ax_ydepth.set_ylabel('y_cam (m)')
    ax_ydepth.set_title('y_cam vs depth (z)')
    ax_ydepth.set_xlim(dmin, dmax)
    ax_ydepth.set_ylim(0.0, y_hi + y_pad)
    ax_ydepth.grid(True, alpha=0.3)
    ax_ydepth.legend(loc='upper left', fontsize=7)

    ax_spacer = fig.add_subplot(2, 3, 6)
    ax_spacer.axis('off')
    if len(b_area) > 0:
        area_lines = [bbox_area_caption]
        for i in range(len(b_area)):
            area_lines.append(f'  sample {i}: {float(b_area[i]):.0f}')
        ax_spacer.text(
            0.05,
            0.95,
            '\n'.join(area_lines),
            transform=ax_spacer.transAxes,
            va='top',
            ha='left',
            fontsize=10,
            family='monospace',
        )

    footer = f'z_hole (range ref) = {zh:.4f} m'
    if len(b_area) > 0:
        footer += '  |  bbox A (px^2): ' + ', '.join(f'{float(a):.0f}' for a in b_area)
    fig.text(0.02, 0.01, footer, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

    out_path = os.path.join(out_dir, f'trajectory_debug_{time.time_ns()}.png')
    try:
        fig.savefig(out_path, dpi=120)
    except OSError as e:
        log_warn(f'Could not write trajectory plot {out_path!r}: {e}')
    finally:
        plt.close(fig)

    log_info(f'Wrote 3d trajectory debug plot: {out_path}')


def compute_and_publish_points(
    *,
    points: list[TrajectoryPoint],
    depth_scale: float,
    hole_distance_m: float,
    max_publish_distance_m: float,
    min_publish_interval_sec: float,
    last_publish_mono: float,
    now_mono: float,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    first_bgr: np.ndarray | None,
    publisher: Callable[[Float32MultiArray], None],
    result_topic: str,
    log_info: Callable[[str], None],
    log_warn: Callable[[str], None],
    plot_out_dir: str = TRAJECTORY_DEBUG_PLOT_DIR,
    first_frame_panel3_title: str = 'Frame 1 raw color',
    bbox_area_caption: str = 'NN bbox area (axis-aligned, px^2):',
) -> tuple[bool, bool, float]:
    """Fit trajectory, predict landing, optionally publish. Returns (published, arm_keyboard, last_publish_mono).

    ``last_publish_mono`` in the return value is updated only when a message is published;
    otherwise pass through the input ``last_publish_mono``.
    """
    pts = sorted(points, key=lambda p: p[0])
    t0 = pts[0][0]

    if now_mono - last_publish_mono < min_publish_interval_sec:
        lines = [
            '--- bag distance backtrack ---',
            backtrack_depth_scale_line(depth_scale),
            'outcome: publish_cooldown (no trajectory message; landing math not run)',
            f'  elapsed_since_last_publish_s={now_mono - last_publish_mono:.6f}  '
            f'min_publish_interval_s={min_publish_interval_sec:.6f}',
        ]
        lines.extend(backtrack_sample_point_lines(pts, t0, depth_scale)[1:])
        log_info('\n'.join(lines))
        return (False, False, last_publish_mono)

    times = np.array([p[0] - t0 for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    depths = np.array([p[3] for p in pts])
    sample_us = np.array([p[4] for p in pts], dtype=int)
    sample_vs = np.array([p[5] for p in pts], dtype=int)
    bbox_areas = np.array([p[6] for p in pts], dtype=float)

    coeffs_x = np.polyfit(times, xs, 1)
    coeffs_y = np.polyfit(times, ys, 2)
    coeffs_depth = np.polyfit(times, depths, 1)

    vx, x0 = coeffs_x
    a_half, vy, y0 = coeffs_y
    vz, z0 = coeffs_depth

    zh = float(hole_distance_m)
    lines = [
        '--- bag distance backtrack ---',
    ]
    lines.extend(backtrack_sample_point_lines(pts, t0, depth_scale))
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

    def _plot(
        t_land: float | None,
        x_land: float | None,
        y_land: float | None,
        depth_land: float | None,
    ) -> None:
        save_trajectory_debug_plot_3d(
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
            sample_us=sample_us,
            sample_vs=sample_vs,
            bbox_areas_px=bbox_areas,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            out_dir=plot_out_dir,
            log_warn=log_warn,
            log_info=log_info,
            first_frame_panel3_title=first_frame_panel3_title,
            bbox_area_caption=bbox_area_caption,
        )

    if abs(vz) < 1e-6:
        lines.append(f'  vz={vz:.6e}  => |vz|<1e-6, t_land undefined (division by zero)')
        lines.append('  outcome: vz_near_zero (no /bean_bag_trajectory publish)')
        log_info('\n'.join(lines))
        _plot(None, None, None, None)
        return (False, True, last_publish_mono)

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
    lines.append(f'  max_publish_distance_m = {max_publish_distance_m:.6f}')

    if distance_m > max_publish_distance_m:
        lines.append(
            '  outcome: withheld_over_distance_cap '
            f'(distance_m > max_publish_distance_m; no publish on {result_topic!r})'
        )
        log_info('\n'.join(lines))
        _plot(t_land, x_land, y_land, depth_land)
        return (False, True, last_publish_mono)

    msg = Float32MultiArray()
    msg.data = [float(distance_m), float(angle)]
    publisher(msg)
    new_last = time.monotonic()
    lines.append(
        f'  outcome: published std_msgs/Float32MultiArray on {result_topic!r} '
        f'data=[distance_m, angle_rad]=[{distance_m:.6f}, {angle:.6f}]'
    )
    log_info('\n'.join(lines))
    _plot(t_land, x_land, y_land, depth_land)

    return (True, True, new_last)
