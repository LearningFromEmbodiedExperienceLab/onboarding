import numpy as np


def damped_lstsq(J: np.ndarray, dx: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Solve ``J @ dq = dx`` with damped least squares.

    ``J`` is ``(..., m, n)``, ``dx`` is ``(..., m)`` → ``dq`` is ``(..., n)``.
    ``dq = (JᵀJ + λI)⁻¹ Jᵀ dx``. Larger ``damping`` makes the step smaller and
    more stable when ``J`` is ill-conditioned.
    """
    j_t = np.swapaxes(J, -1, -2)
    jtj = j_t @ J
    n = jtj.shape[-1]
    rhs = j_t @ dx[..., None]
    dq = np.linalg.solve(jtj + damping * np.eye(n), rhs)
    return dq[..., 0]
