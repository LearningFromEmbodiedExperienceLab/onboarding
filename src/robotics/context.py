"""Reusable context managers: wall-clock timer and a scoped NumPy seed."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import TypeVar

import numpy as np

F = TypeVar("F", bound=Callable)


class Timer:
    """``with Timer("ik"): ...`` stores wall time in ``.elapsed`` (seconds).

    CUDA kernels may still be running when the block exits — see the
    async section. This timer is for CPU / host-side work, or after an
    explicit ``torch.cuda.synchronize()``.
    """

    def __init__(self, name: str = "", *, print_on_exit: bool = True):
        self.name = name
        self.elapsed = 0.0
        self.print_on_exit = print_on_exit
        self._t0 = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.elapsed = time.perf_counter() - self._t0
        if self.print_on_exit:
            label = f"{self.name}: " if self.name else ""
            print(f"{label}{self.elapsed * 1e3:.3f} ms")
        return False


def timed(fn: F) -> F:
    """Decorator: wrap a function in :class:`Timer` using the function name."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        with Timer(fn.__qualname__):
            return fn(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


@contextmanager
def numpy_seed(seed: int) -> Iterator[None]:
    """Pin the *legacy* global NumPy RNG for the block, then restore it.

    Isolated new code should use ``np.random.default_rng(seed)`` instead of
    touching global state. This context is for a short reproducible snippet
    or a library that still calls ``np.random.randn``.
    """
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)
