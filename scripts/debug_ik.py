"""A script meant to be *paused*, not just run.

Run with ``python scripts/debug_ik.py`` and you drop into pdb (or pdbpp)
at ``breakpoint()``. In the editor, use the gutter + F5 instead (see
``.vscode/launch.json``) and you can leave this line in place — debugpy
honours it too.
"""

import numpy as np

from robotics.ik.differential import damped_lstsq


def residual(J: np.ndarray, dx: np.ndarray, dq: np.ndarray) -> np.ndarray:
    return J @ dq - dx


def main() -> None:
    # Rank-1 Jacobian: columns are copies of each other, so JᵀJ is singular
    # without damping. Step into ``damped_lstsq`` and inspect ``jtj``.
    J = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )
    dx = np.array([1.0, 0.0, 0.0])
    damping = 1e-3

    breakpoint()

    dq = damped_lstsq(J, dx, damping=damping)
    r = residual(J, dx, dq)
    print("dq", dq)
    print("residual", r, "norm", float(np.linalg.norm(r)))


if __name__ == "__main__":
    main()
