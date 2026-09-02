"""2D Bézier and interpolating B-spline helpers for doc videos."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Cubic Bézier in R^d; u ∈ [0, 1] → shape (len(u), d)."""
    s = np.asarray(u, dtype=float)
    omu = 1.0 - s
    omu2 = omu * omu
    omu3 = omu2 * omu
    s2 = s * s
    s3 = s2 * s
    return (
        omu3[:, None] * p0
        + 3 * omu2[:, None] * s[:, None] * p1
        + 3 * omu[:, None] * s2[:, None] * p2
        + s3[:, None] * p3
    )


def interpolating_spline_2d(
    waypoints: np.ndarray,
    u: np.ndarray,
    *,
    degree: int = 3,
) -> np.ndarray:
    """Cubic interpolating spline through 2D waypoints; u ∈ [0, 1]."""
    pts = np.asarray(waypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"waypoints must be (n, 2), got {pts.shape}")
    if len(pts) < degree + 1:
        raise ValueError(f"need at least {degree + 1} waypoints")
    u_knot = np.linspace(0.0, 1.0, len(pts))
    u = np.asarray(u, dtype=float)
    sx = make_interp_spline(u_knot, pts[:, 0], k=degree)
    sy = make_interp_spline(u_knot, pts[:, 1], k=degree)
    return np.column_stack([sx(u), sy(u)])


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, shape (m, 2)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()
    order = np.lexsort((pts[:, 0], pts[:, 1]))
    pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull
