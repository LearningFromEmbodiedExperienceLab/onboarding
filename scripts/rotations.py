"""Rotation conversions. All quaternions in this package are ``(w, x, y, z)``.

Run::

    uv run python scripts/rotations.py
"""

from __future__ import annotations

import numpy as np

from robotics.math_utils import (
    axis_angle_from_quat,
    euler_from_quat,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_rotate,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def main() -> None:
    rpy = np.array([0.0, 0.0, np.pi / 2])
    q = quat_from_euler_xyz(rpy)
    print("90° yaw  wxyz", q)
    print("          xyzw", wxyz_to_xyzw(q))
    print("R\n", matrix_from_quat(q))
    print("axis-angle", axis_angle_from_quat(q))
    print("euler back", euler_from_quat(q))
    x = np.array([1.0, 0.0, 0.0])
    print("R applied to +X (expect +Y)", quat_rotate(q, x))

    q_xyzw_misused = wxyz_to_xyzw(q)
    print("WRONG: quat_rotate(xyzw, x) as if wxyz", quat_rotate(q_xyzw_misused, x))
    print("fixed: quat_rotate(xyzw_to_wxyz(xyzw), x)", quat_rotate(xyzw_to_wxyz(q_xyzw_misused), x))

    aa = axis_angle_from_quat(q)
    q2 = quat_from_angle_axis(np.linalg.norm(aa), aa / np.linalg.norm(aa))
    print("q and -q same rotation", np.allclose(quat_rotate(q, x), quat_rotate(-q, x)))
    print("roundtrip axis-angle", np.allclose(quat_rotate(q2, x), quat_rotate(q, x)))


if __name__ == "__main__":
    main()
