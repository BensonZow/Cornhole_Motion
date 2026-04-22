"""Wire format: ``std_msgs/Float64MultiArray`` for middle-3 throw handoff (NN -> bag_sense_fast).

Layout version 1 — length 36:
  [0]   = version (1.0)
  [1]   = color_w (px)
  [2]   = color_h (px)
  For k in 0,1,2 (base = 3 + k * 11):
    [base+0]  = header stamp sec (float, integer-valued)
    [base+1]  = header stamp nanosec (float, integer-valued 0..1e9)
    [base+2:base+11] = det row: conf, u0,v0, u1,v1, u2,v2, u3,v3
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from std_msgs.msg import Float64MultiArray

THROW_BATCH_VERSION = 1.0
THROW_BATCH_LEN = 36


def pack_throw_batch(
    color_w: int,
    color_h: int,
    samples: list[tuple[int, np.ndarray]],
) -> Float64MultiArray:
    """``samples`` is three ``(stamp_ns, row9)`` with ``row9`` shape ``(1,9)`` conf+8 corners."""
    if len(samples) != 3:
        raise ValueError(f'need 3 samples, got {len(samples)}')
    msg = Float64MultiArray()
    data: list[float] = [THROW_BATCH_VERSION, float(color_w), float(color_h)]
    for stamp_ns, row in samples:
        sec = stamp_ns // 1_000_000_000
        nsec = int(stamp_ns - sec * 1_000_000_000)
        data.append(float(sec))
        data.append(float(nsec))
        flat = np.asarray(row, dtype=np.float64).reshape(-1)
        if flat.size < 9:
            raise ValueError('row must have 9 values')
        data.extend(float(x) for x in flat[:9])
    if len(data) != THROW_BATCH_LEN:
        raise ValueError(f'packed len {len(data)} expected {THROW_BATCH_LEN}')
    msg.data = [float(x) for x in data]
    return msg


def unpack_throw_batch(
    msg: Any,
) -> tuple[int, int, list[tuple[int, np.ndarray]]] | None:
    """Returns ``(color_w, color_h, list of (stamp_ns, row9 as shape (1,9)))`` or None if invalid."""
    d = list(getattr(msg, 'data', []) or [])
    if len(d) < THROW_BATCH_LEN:
        return None
    if not math.isclose(d[0], THROW_BATCH_VERSION, rel_tol=0.0, abs_tol=0.01):
        return None
    cw = int(round(d[1]))
    ch = int(round(d[2]))
    out: list[tuple[int, np.ndarray]] = []
    o = 3
    for _k in range(3):
        sec = int(round(d[o]))
        nsec = int(round(d[o + 1]))
        o += 2
        stamp_ns = sec * 1_000_000_000 + max(0, min(nsec, 999_999_999))
        row = np.array(d[o : o + 9], dtype=np.float64).reshape(1, 9)
        o += 9
        out.append((stamp_ns, row))
    if o != THROW_BATCH_LEN:
        return None
    return (cw, ch, out)
