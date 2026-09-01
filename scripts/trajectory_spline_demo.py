"""Joint-space splines and trapezoidal time parameterization (NumPy + SciPy).

Demonstrates: piecewise cubic Hermite, cubic Bézier, interpolating B-spline,
and u(t) trapezoid → q(t).

Run::

    uv sync --extra sim
    uv run python scripts/trajectory_spline_demo.py
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import make_interp_spline
except ImportError as exc:
    raise SystemExit(
        "SciPy required for B-splines.\n  uv sync --extra sim"
    ) from exc


def cubic_hermite_segment(
    q0: float, q1: float, v0: float, v1: float, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic on t ∈ [0, 1] with position/velocity at endpoints."""
    t = np.asarray(t, dtype=float)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    q = h00 * q0 + h10 * v0 + h01 * q1 + h11 * v1
    dq = (6 * t**2 - 6 * t) * q0 + (3 * t**2 - 4 * t + 1) * v0
    dq += (-6 * t**2 + 6 * t) * q1 + (3 * t**2 - 2 * t) * v1
    return q, dq


def cubic_bezier(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Cubic Bézier in R^d; u ∈ [0, 1] → shape (len(u), d)."""
    s = np.asarray(u, dtype=float)
    omu = 1.0 - s
    omu2 = omu * omu
    omu3 = omu2 * omu
    s2 = s * s
    s3 = s2 * s
    return omu3[:, None] * p0 + 3 * omu2[:, None] * s[:, None] * p1 + 3 * omu[:, None] * s2[:, None] * p2 + s3[:, None] * p3


def trapezoid_u(t: np.ndarray, *, T: float, v_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoidal velocity profile: u(0)=0, u(T)=1, |u̇| ≤ v_max."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, T)
    u = np.zeros_like(t)
    ud = np.zeros_like(t)

    # Triangular velocity when v_max * T < 2 (no cruise segment).
    if v_max * T < 2.0:
        v_peak = 2.0 / T
        half = T / 2.0
        for i, ti in enumerate(t):
            if ti <= half:
                ud[i] = v_peak * (ti / half)
                u[i] = 0.5 * v_peak * ti**2 / half
            else:
                dt = T - ti
                ud[i] = v_peak * (dt / half)
                u[i] = 1.0 - 0.5 * v_peak * dt**2 / half
        return u, ud

    t_acc = 1.0 / v_max
    t_cruise = T - 2.0 * t_acc
    u_after_accel = 0.5 * v_max * t_acc
    for i, ti in enumerate(t):
        if ti < t_acc:
            ud[i] = v_max * (ti / t_acc)
            u[i] = 0.5 * v_max * ti**2 / t_acc
        elif ti < t_acc + t_cruise:
            ud[i] = v_max
            u[i] = u_after_accel + v_max * (ti - t_acc)
        else:
            dt_dec = ti - (t_acc + t_cruise)
            ud[i] = v_max * (1.0 - dt_dec / t_acc)
            u[i] = u_after_accel + v_max * t_cruise + v_max * dt_dec - 0.5 * v_max * dt_dec**2 / t_acc
    return u, ud


def main() -> None:
    print("=== 1. Piecewise cubic Hermite (one joint, t ∈ [0,1]) ===")
    t = np.linspace(0, 1, 5)
    q, dq = cubic_hermite_segment(0.0, 1.2, 0.0, 0.0, t)
    for ti, qi, dqi in zip(t, q, dq, strict=True):
        print(f"  t={ti:.2f}  q={qi:.4f}  q̇={dqi:.4f}")

    print("\n=== 2. Cubic Bézier (EE point in xy, u samples) ===")
    p0, p1, p2, p3 = (
        np.array([0.0, 0.0]),
        np.array([0.15, 0.25]),
        np.array([0.35, 0.15]),
        np.array([0.5, 0.0]),
    )
    u = np.linspace(0, 1, 5)
    xy = cubic_bezier(p0, p1, p2, p3, u)
    for ui, pt in zip(u, xy, strict=True):
        print(f"  u={ui:.2f}  xy=({pt[0]:.3f}, {pt[1]:.3f})")

    print("\n=== 3. Cubic B-spline through joint waypoints ===")
    q_way = np.array([0.0, 0.4, 0.9, 0.6, 1.0])
    u_way = np.linspace(0, 1, len(q_way))
    spline = make_interp_spline(u_way, q_way, k=3)
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  u={u:.2f}  q={float(spline(u)):.4f}  dq/du={float(spline(u, 1)):.4f}")

    print("\n=== 4. Trapezoidal u(t) → q(t) = spline(u(t)) ===")
    T = 2.0
    v_max = 0.8
    t = np.linspace(0, T, 6)
    u_t, udot_t = trapezoid_u(t, T=T, v_max=v_max)
    q_t = spline(u_t)
    qdot_t = spline(u_t, 1) * udot_t
    for ti, ui, qi, qdi in zip(t, u_t, q_t, qdot_t, strict=True):
        print(f"  t={ti:.2f}  u={ui:.3f}  q={qi:.4f}  q̇={qdi:.4f}")


if __name__ == "__main__":
    main()
