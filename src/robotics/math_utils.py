"""Batch-agnostic geometry on the last axis.

Feature axes are last: quaternions ``(..., 4)``, vectors ``(..., 3)``.
Leading axes (envs, bodies, time, …) are unnamed and may be absent.
"""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Unit vectors along the last axis. Shape ``(..., d)`` → ``(..., d)``."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap radians to ``[-π, π]``. Any shape."""
    wrapped = (angles + np.pi) % (2 * np.pi)
    return np.where((wrapped == 0) & (angles > 0), np.pi, wrapped - np.pi)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate ``(w, x, y, z)``. Shape ``(..., 4)``."""
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1)


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of ``(w, x, y, z)`` quaternions. Shapes broadcast."""
    q, r = np.broadcast_arrays(q, r)
    w1, x1, y1, z1 = np.moveaxis(q, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(r, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by quaternions ``(w, x, y, z)``.

    ``quat`` is ``(..., 4)``, ``vec`` is ``(..., 3)``; leading axes broadcast.
    """
    xyz = quat[..., 1:]
    t = np.cross(xyz, vec) * 2
    return vec + quat[..., 0:1] * t + np.cross(xyz, t)


def yaw_rotate(yaw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` in XY by ``yaw`` (radians); Z is unchanged.

    ``yaw`` broadcasts with ``vec[..., 0]``. A per-env yaw over bodies is
    ``yaw[:, None]`` against ``vec`` of shape ``(n_env, n_body, 3)``.
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    x = c * vec[..., 0] - s * vec[..., 1]
    y = s * vec[..., 0] + c * vec[..., 1]
    z = vec[..., 2] + np.zeros_like(x)
    return np.stack([x, y, z], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` → ``(x, y, z, w)``."""
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """``(x, y, z, w)`` → ``(w, x, y, z)``."""
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` from ``(w, x, y, z)``."""
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_from_euler_xyz(rpy: np.ndarray) -> np.ndarray:
    """XYZ intrinsic Euler (roll, pitch, yaw) → ``(w, x, y, z)``."""
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    return np.stack(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        axis=-1,
    )


def euler_from_quat(quat: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` → XYZ Euler ``(..., 3)`` (roll, pitch, yaw)."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.where(
        np.abs(sin_pitch) >= 1,
        np.copysign(np.pi / 2.0, sin_pitch),
        np.arcsin(sin_pitch),
    )
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return np.stack([roll, pitch, yaw], axis=-1)


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """``(w, x, y, z)`` → axis-angle vector ``(..., 3)`` (radians × unit axis)."""
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    scale = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / angle,
        0.5 - angle * angle / 48,
    )
    return quat[..., 1:4] / scale[..., None]


def quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Axis-angle (radians, ``(...,)`` + unit ``(..., 3)``) → ``(w, x, y, z)``."""
    theta = (angle / 2)[..., None]
    xyz = normalize(axis) * np.sin(theta)
    w = np.cos(theta)
    return normalize(np.concatenate([w, xyz], axis=-1))
