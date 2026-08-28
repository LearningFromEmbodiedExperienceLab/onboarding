"""Small records: a pose tuple and an IK config dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import NamedTuple

from robotics.ik.base import IKController
from robotics.ik.registry import make_ik


class Pose(NamedTuple):
    """Immutable pair ``(pos, quat)``. Unpack as ``pos, quat = pose``.

    The *tuple* cannot be reassigned (``pose.pos = …`` fails) but the arrays
    it holds are still mutable: ``pose.pos[:] = 0`` writes through.
    """

    pos: np.ndarray
    quat: np.ndarray


@dataclass
class IKConfig:
    """Mutable config with defaults and validation. Not a tuple."""

    method: str = "dls"
    damping: float = 1e-3

    def __post_init__(self) -> None:
        if self.method == "dls" and self.damping < 0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")

    def build(self) -> IKController:
        kwargs = {}
        if self.method == "dls":
            kwargs["damping"] = self.damping
        return make_ik(self.method, **kwargs)
