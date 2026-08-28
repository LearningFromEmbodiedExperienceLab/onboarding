"""Broadcasting, then heterogeneous batched indexing.

Run::

    uv run python scripts/index_tensors.py
"""

from __future__ import annotations

import numpy as np


def show(label: str, x: np.ndarray) -> None:
    print(f"{label}\n  shape={tuple(x.shape)}\n{x}\n")


def main() -> None:
    # ------------------------------------------------------------------
    # Broadcasting (arithmetic)
    # ------------------------------------------------------------------
    n_env, n_body = 4, 3
    x = np.arange(n_env * n_body * 3, dtype=np.float64).reshape(n_env, n_body, 3)
    show("x (n_env, n_body, 3)", x)

    mass = np.array([1.0, 10.0, 100.0])  # (n_body,)
    scaled = x * mass[:, None]
    show("x * mass[:, None]   # (4, 3, 3) * (3, 1) → (4, 3, 3)", scaled)

    roots = x[:, 0, :]  # (n_env, 3)
    delta = roots[:, None, :] - roots[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    show("pairwise root distances (4, 4)  via (4,1,3)-(1,4,3)", dist)

    row = np.arange(4.0)
    silent = np.zeros((4, 4)) + row
    show("(4, 4) + (4,)  adds along the last axis (each ROW gets `row`)", silent)
    print(
        "wanted per-column? use row[:, None] → shape (4, 1)\n"
        f"  (4, 4) + (4,)  last dim 4 vs 4 → OK, maybe not what you meant\n"
        f"  (4, 3) + (4,)  last dim 3 vs 4 → ValueError\n"
    )

    # ------------------------------------------------------------------
    # Indexing: homogeneous zip (same-length 1-D) vs per-env lists
    # ------------------------------------------------------------------
    env_ids = np.array([0, 2])
    show("x[env_ids]  # (2, 3, 3) whole envs", x[env_ids])

    zip_body = np.array([1, 0])
    zip_dim = np.array([2, 1])
    show(
        "homogeneous zip  x[env_ids, zip_body, zip_dim]  → (2,)",
        x[env_ids, zip_body, zip_dim],
    )

    # Per env: different bodies AND different dims.
    # env 0 → bodies 0,2  and dims x,z
    # env 2 → bodies 1,0  and dims y,x
    body_ids = np.array([[0, 2], [1, 0]])  # (K, B)
    dim_ids = np.array([[0, 2], [1, 0]])  # (K, D)
    e = env_ids[:, None, None]  # (K, 1, 1)
    b = body_ids[:, :, None]  # (K, B, 1)
    d = dim_ids[:, None, :]  # (K, 1, D)
    gathered = x[e, b, d]
    show(
        "heterogeneous  x[e, b, d]  with e (K,1,1), body (K,B,1), dim (K,1,D)\n"
        "  → (K, B, D);  out[k, i, j] = x[env_ids[k], body_ids[k,i], dim_ids[k,j]]",
        gathered,
    )
    assert gathered[0, 0, 1] == x[0, 0, 2]
    assert gathered[1, 1, 0] == x[2, 0, 1]

    copy = x[e, b, d]
    copy[:] = 0.0
    print(f"copy = x[e,b,d]; copy[:] = 0  →  x[0, 0, 2] still {x[0, 0, 2]}  (copy)\n")
    x[e, b, d] = 0.0
    show("x[e, b, d] = 0  indexed set", x)


if __name__ == "__main__":
    main()
