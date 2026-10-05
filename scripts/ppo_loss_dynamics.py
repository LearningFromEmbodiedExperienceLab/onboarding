"""Why PPO policy and value losses are not supervised-learning curves.

Numpy only. The shapes match the diagnostics in the lab trainer
(``PPOPolicy.train_op``): advantage normalization, a clipped surrogate that is
~0 when the importance ratio is 1, and a value MSE whose magnitude tracks
return variance. See ``docs/reinforcement-learning/ppo-diagnostics.qmd``.
"""

from __future__ import annotations

import numpy as np


def clipped_policy_loss(log_ratio: np.ndarray, adv: np.ndarray, eps: float = 0.2) -> float:
    """Minimized PPO surrogate: ``-mean(min(r A, clip(r) A))``."""
    ratio = np.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    return float(-np.minimum(surr1, surr2).mean())


def supervised_regression(rng: np.random.Generator) -> list[float]:
    """Fixed inputs and labels. Full-batch gradient descent on MSE."""
    x = rng.normal(size=(256, 4))
    y = x @ np.array([1.0, -0.5, 0.3, 0.2])
    w = np.zeros(4)
    losses: list[float] = []
    for _ in range(12):
        err = x @ w - y
        w -= 0.15 * (x.T @ err) / len(x)
        losses.append(float(np.mean(err**2)))
    return losses


def policy_updates(rng: np.random.Generator, n_updates: int = 6) -> list[tuple[float, float]]:
    """Each update: fresh rollout, standardized advantages, one trust-region step.

    Returns ``(loss at ratio=1, loss after the step)`` per update. The step
    moves log-prob in the advantage direction and then stops — the next
    rollout's behavior policy is the new policy, so the ratio resets to 1.
    """
    rows: list[tuple[float, float]] = []
    for _ in range(n_updates):
        adv = rng.normal(size=8192)
        adv = (adv - adv.mean()) / adv.std()
        at_start = clipped_policy_loss(np.zeros_like(adv), adv)
        log_ratio = np.clip(0.15 * adv, -0.18, 0.18)
        after = clipped_policy_loss(log_ratio, adv)
        rows.append((at_start, after))
    return rows


def value_across_rollouts(rng: np.random.Generator, n_rollouts: int = 8) -> list[tuple[float, float, float, float]]:
    """Growing return scale, fixed irreducible noise fraction, a few SGD steps.

    Returns ``(return std, mse at start of fit, mse after fit, explained variance)``.
    Targets are frozen inside a rollout (so the inner MSE drops) and redrawn
    every rollout with a larger scale (so the logged MSE need not drop).
    """
    w_true = np.array([1.0, 0.4, -0.3, 0.2])
    w = np.zeros(4)
    rows: list[tuple[float, float, float, float]] = []
    for k in range(n_rollouts):
        scale = 1.0 + 0.45 * k
        x = rng.normal(size=(1024, 4))
        # Noise std is a fixed fraction of the signal, so a perfect fit still
        # has MSE proportional to scale^2.
        signal = scale * (x @ w_true)
        y = signal + rng.normal(scale=0.55 * scale, size=1024)
        start_mse = float(np.mean((x @ w - y) ** 2))
        for _ in range(6):
            err = x @ w - y
            w -= 0.25 * (x.T @ err) / len(x)
        end_mse = float(np.mean((x @ w - y) ** 2))
        explained = 1.0 - end_mse / float(np.var(y))
        rows.append((float(np.std(y)), start_mse, end_mse, explained))
    return rows


def main() -> None:
    rng = np.random.default_rng(0)

    print("supervised MSE on a fixed dataset (should fall and stay low)")
    sl = supervised_regression(rng)
    print("  first 3:", " ".join(f"{v:.4f}" for v in sl[:3]))
    print("  last  3:", " ".join(f"{v:.4f}" for v in sl[-3:]))

    print()
    print("policy surrogate on successive rollouts (standardized advantages)")
    print(f"  {'update':>6}  {'at ratio=1':>12}  {'after step':>12}")
    for i, (start, after) in enumerate(policy_updates(rng)):
        print(f"  {i:6d}  {start:12.4f}  {after:12.4f}")

    print()
    print("value MSE on successive rollouts (return scale grows, targets frozen within a rollout)")
    print(f"  {'rollout':>7}  {'ret std':>8}  {'mse start':>10}  {'mse end':>8}  {'expl var':>8}")
    for i, (ret_std, mse0, mse1, ev) in enumerate(value_across_rollouts(rng)):
        print(f"  {i:7d}  {ret_std:8.3f}  {mse0:10.3f}  {mse1:8.3f}  {ev:8.3f}")


if __name__ == "__main__":
    main()
