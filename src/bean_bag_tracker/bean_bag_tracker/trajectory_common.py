"""Shared trajectory fit, landing prediction, debug plot for ``ros2_bag_sense_fast``.

Keep landing math, publish format, and plot logic aligned with the tracker node.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
import numpy as np
from std_msgs.msg import Float32MultiArray

# Trajectory debug PNGs (matplotlib Agg, 1×2 y–x and y–z).
TRAJECTORY_DEBUG_PLOT_DIR = '/home/cornholio/ros2_jazzy/log'

# (t_ros_s, x_m, y_cam_m, depth_m, u_px, v_px, bbox_area_px)
TrajectoryPoint = tuple[float, float, float, float, int, int, float]


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


def save_trajectory_2d_plots(
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
    bbox_areas_px: np.ndarray,
    out_dir: str,
    log_warn: Callable[[str], None],
    log_info: Callable[[str], None],
) -> None:
    """Save one PNG: 1×2 y_cam vs x, y_cam vs depth (z) with fit curves when landing is defined."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log_warn('matplotlib not installed; skipping trajectory debug plot PNG')
        return

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        log_warn(f'Could not create trajectory plot dir {out_dir!r}: {e}')
        return

    xs_c = np.asarray(xs, dtype=float).copy()
    ys_c = np.asarray(ys, dtype=float).copy()
    depths_c = np.asarray(depths, dtype=float).copy()
    b_area = np.asarray(bbox_areas_px, dtype=float).ravel()

    arc_x = arc_y_cam = arc_depth = None
    if t_land is not None and x_land is not None and y_land is not None and depth_land is not None:
        if math.isfinite(float(t_land)):
            n = max(25, int(40 * (1.0 + min(abs(float(t_land)), 5.0))))
            tt = np.linspace(0.0, float(t_land), n)
            arc_x = np.polyval(np.array([vx, x0]), tt)
            arc_y_cam = np.polyval(np.array([a_half, vy, y0]), tt)
            arc_depth = np.polyval(np.array([vz, z0]), tt)

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

    dmin = float(np.min(depth_all))
    dmax = float(np.max(depth_all))
    if dmax - dmin < 1e-6:
        dmin -= 0.05
        dmax += 0.05
    else:
        dm = 0.1 * (dmax - dmin)
        dmin -= dm
        dmax += dm

    y_hi = float(np.max(y_all))
    if y_hi < 1e-6:
        y_hi = 0.05
    y_pad = max(0.02, 0.05 * y_hi)

    fig, (ax_yx, ax_ydepth) = plt.subplots(1, 2, figsize=(12, 5))

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
    if arc_x is not None and arc_y_cam is not None:
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
    if arc_depth is not None and arc_y_cam is not None:
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

    log_info(f'Wrote 2d trajectory debug plot: {out_path}')


def compute_and_publish_points(
    *,
    points: list[TrajectoryPoint],
    depth_scale: float,
    hole_distance_m: float,
    max_publish_distance_m: float,
    min_publish_interval_sec: float,
    last_publish_mono: float,
    now_mono: float,
    publisher: Callable[[Float32MultiArray], None],
    result_topic: str,
    log_info: Callable[[str], None],
    log_warn: Callable[[str], None],
    plot_out_dir: str = TRAJECTORY_DEBUG_PLOT_DIR,
) -> tuple[bool, bool, float]:
    """Fit trajectory, predict landing, optionally publish. Returns (published, unused_false, last_publish_mono).

    The second return value is kept for API compatibility; callers should ignore it.
    ``last_publish_mono`` is updated only when a message is published; otherwise unchanged.
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
        save_trajectory_2d_plots(
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
            bbox_areas_px=bbox_areas,
            out_dir=plot_out_dir,
            log_warn=log_warn,
            log_info=log_info,
        )

    if abs(vz) < 1e-6:
        lines.append(f'  vz={vz:.6e}  => |vz|<1e-6, t_land undefined (division by zero)')
        lines.append('  outcome: vz_near_zero (no /bean_bag_trajectory publish)')
        log_info('\n'.join(lines))
        _plot(None, None, None, None)
        return (False, False, last_publish_mono)

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
        return (False, False, last_publish_mono)

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

    return (True, False, new_last)
