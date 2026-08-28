"""Shared IK controller surface: ``ctrl(J, dx)`` plus per-env indexing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IKController(ABC):
    """Map a Jacobian and task error to a joint step.

    ``J`` is ``(..., m, n)``, ``dx`` is ``(..., m)`` → ``dq`` is ``(..., n)``.
    """

    def __init__(self) -> None:
        self._dq: np.ndarray | None = None

    @abstractmethod
    def compute(self, J: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Return ``dq``; do not store it — :meth:`__call__` does that."""

    def __call__(self, J: np.ndarray, dx: np.ndarray) -> np.ndarray:
        self._dq = np.asarray(self.compute(J, dx))
        return self._dq

    def __len__(self) -> int:
        if self._dq is None:
            return 0
        return 1 if self._dq.ndim == 1 else int(self._dq.shape[0])

    def __getitem__(self, env_id: int | slice) -> np.ndarray:
        if self._dq is None:
            raise RuntimeError("call the controller before indexing")
        if self._dq.ndim == 1:
            if env_id != 0:
                raise IndexError(env_id)
            return self._dq
        return self._dq[env_id]

    def __setitem__(self, env_id: int | slice, value: np.ndarray) -> None:
        if self._dq is None:
            raise RuntimeError("call the controller before indexing")
        if self._dq.ndim == 1:
            if env_id != 0:
                raise IndexError(env_id)
            self._dq[:] = value
            return
        self._dq[env_id] = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
