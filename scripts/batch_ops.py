"""Same geometry, three ranks: (), (n_env,), (n_env, n_body,).

Run::

    uv run python scripts/batch_ops.py
"""

from __future__ import annotations

import numpy as np

from robotics.ik.differential import damped_lstsq
from robotics.math_utils import quat_mul, quat_rotate, yaw_rotate


def show(label: str, x: np.ndarray) -> None:
    print(f"{label:40s}  shape={tuple(x.shape)}")


def main() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # 90° about +Z
    v = np.array([1.0, 0.0, 0.0])

    show("quat_mul(q, identity)", quat_mul(q, identity))
    show("quat_rotate(q, v)  # → +Y", quat_rotate(q, v))

    n_env = 4
    q_env = np.broadcast_to(q, (n_env, 4)).copy()
    v_env = np.broadcast_to(v, (n_env, 3)).copy()
    show("quat_rotate, (n_env,)", quat_rotate(q_env, v_env))

    n_body = 3
    q_body = np.broadcast_to(q, (n_env, n_body, 4)).copy()
    v_body = np.broadcast_to(v, (n_env, n_body, 3)).copy()
    show("quat_rotate, (n_env, n_body)", quat_rotate(q_body, v_body))

    # One yaw per env, many bodies: yaw (n_env,) broadcasts onto (n_env, n_body, 3).
    yaw = np.linspace(0.0, np.pi / 2, n_env)
    rotated = yaw_rotate(yaw[:, None], v_body)
    show("yaw_rotate(yaw[:, None], vec)", rotated)

    # Batched IK: one Jacobian per env.
    J = np.broadcast_to(np.eye(3), (n_env, 3, 3)).copy()
    dx = np.zeros((n_env, 3))
    dx[:, 0] = 0.1
    show("damped_lstsq, (n_env, 3, 3)", damped_lstsq(J, dx))


if __name__ == "__main__":
    main()
