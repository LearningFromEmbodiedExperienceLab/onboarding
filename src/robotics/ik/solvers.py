"""Concrete IK solvers. Import this module to populate the registry."""

from __future__ import annotations

import numpy as np

from robotics.ik.base import IKController
from robotics.ik.differential import damped_lstsq
from robotics.ik.registry import register


@register("trans")
class TransposeIK(IKController):
    """``dq = Jᵀ dx`` — cheap, stable near singularities, slow to converge."""

    def compute(self, J: np.ndarray, dx: np.ndarray) -> np.ndarray:
        j_t = np.swapaxes(J, -1, -2)
        return (j_t @ dx[..., None])[..., 0]


@register("pinv")
class PinvIK(IKController):
    """Moore–Penrose pseudoinverse. Accurate off singularities."""

    def compute(self, J: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return (np.linalg.pinv(J) @ dx[..., None])[..., 0]


@register("dls")
class DampedIK(IKController):
    """Damped least squares (Levenberg–Marquardt)."""

    def __init__(self, damping: float = 1e-3) -> None:
        super().__init__()
        self.damping = damping

    def compute(self, J: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return damped_lstsq(J, dx, damping=self.damping)

    def __repr__(self) -> str:
        return f"DampedIK(damping={self.damping})"
