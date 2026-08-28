"""Shape manipulation: reshape, squeeze, expand/repeat, einops.

Run::

    uv run python scripts/reshape_tensors.py
"""

from __future__ import annotations

import numpy as np
from einops import rearrange, reduce, repeat


def show(label: str, x: np.ndarray) -> None:
    print(f"{label}\n  shape={tuple(x.shape)}\n")


def main() -> None:
    n_env, n_body = 4, 3
    x = np.arange(n_env * n_body * 3, dtype=np.float64).reshape(n_env, n_body, 3)
    show("x", x)

    show("reshape (n, b, 3) → (n, b*3)", x.reshape(n_env, -1))
    show("transpose then reshape (always ok)", np.transpose(x, (0, 2, 1)).reshape(n_env, 3, n_body))

    mass = np.array([1.0, 10.0, 100.0])
    show("mass (b,)", mass)
    show("mass[:, None] unsqueeze → (b, 1)", mass[:, None])
    show("squeeze last axis of that", mass[:, None].squeeze(-1))

    ones = np.ones((n_env, 1, 3))
    show("broadcast_to (expand-like, read-only)", np.broadcast_to(ones, (n_env, n_body, 3)))
    show("np.repeat along body axis (copy)", np.repeat(ones, n_body, axis=1))

    show('rearrange  "n b xyz -> n xyz b"', rearrange(x, "n b xyz -> n xyz b"))
    show('rearrange  "n b xyz -> n (b xyz)"', rearrange(x, "n b xyz -> n (b xyz)"))
    feat = rearrange(x, "n b xyz -> n (b xyz)")
    show("split back  n (b xyz) -> n b xyz", rearrange(feat, "n (b xyz) -> n b xyz", b=n_body, xyz=3))

    hx = np.arange(n_env * 2).reshape(n_env, 2)
    show('repeat hx  "n h -> n t h" t=3', repeat(hx, "n h -> n t h", t=3))
    show('reduce x  "n b xyz -> n xyz" mean', reduce(x, "n b xyz -> n xyz", "mean"))


if __name__ == "__main__":
    main()
