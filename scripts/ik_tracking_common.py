"""Shared differential IK tracking helpers for MuJoCo and Motrix demos."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from robotics.ik import make_ik

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
ARM_DOF = len(ARM_JOINT_NAMES)

# Nominal arm posture used to seed circle center and random targets.
NOMINAL_Q = np.array([0.5, 0.3, -0.4, 0.2, 0.1, -0.2], dtype=float)

POS_TOL_M = 5e-3
MAX_STEPS_REACH = 400
MAX_STEPS_CIRCLE = 600
INTEGRATION_GAIN = 0.35
DLS_DAMPING = 1e-2


class TrackingTest(str, Enum):
    REACH = "reach"
    CIRCLE = "circle"


@dataclass(frozen=True)
class TrackingResult:
    test: TrackingTest
    backend: str
    steps: int
    final_pos_err_m: float
    success: bool
    mean_pos_err_m: float


class RobotKinematics(Protocol):
    """Minimal FK + Jacobian interface for differential IK."""

    backend: str

    def reset(self) -> None: ...

    def get_arm_qpos(self) -> np.ndarray: ...

    def set_arm_qpos(self, q: np.ndarray) -> None: ...

    def get_ee_pos(self) -> np.ndarray: ...

    def position_jacobian(self) -> np.ndarray: ...

    def forward(self) -> None: ...


def clip_arm_qpos(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    return np.clip(q, q_min, q_max)


def numerical_position_jacobian(
    get_ee_pos,
    get_arm_qpos,
    set_arm_qpos,
    forward,
    n_dof: int = ARM_DOF,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference position Jacobian via FK (Motrix has no analytic J API)."""
    q0 = get_arm_qpos().copy()
    p0 = get_ee_pos()
    jac = np.zeros((3, n_dof), dtype=float)
    for i in range(n_dof):
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[i] += eps
        q_minus[i] -= eps
        set_arm_qpos(q_plus)
        forward()
        p_plus = get_ee_pos()
        set_arm_qpos(q_minus)
        forward()
        p_minus = get_ee_pos()
        jac[:, i] = (p_plus - p_minus) / (2.0 * eps)
    set_arm_qpos(q0)
    forward()
    return jac


def random_proximal_target(
    ee_pos: np.ndarray,
    rng: np.random.Generator,
    box_m: float = 0.04,
) -> np.ndarray:
    """Sample a reachable-ish target near the current EE position."""
    offset = rng.uniform(-box_m, box_m, size=3)
    return ee_pos + offset


def circle_target(
    center: np.ndarray,
    radius: float,
    phase: float,
    plane: str = "xy",
) -> np.ndarray:
    c, s = np.cos(phase), np.sin(phase)
    if plane == "xy":
        delta = np.array([radius * c, radius * s, 0.0])
    else:
        delta = np.array([radius * c, 0.0, radius * s])
    return center + delta


def run_differential_ik(
    robot: RobotKinematics,
    *,
    test: TrackingTest,
    q_min: np.ndarray,
    q_max: np.ndarray,
    rng: np.random.Generator,
    circle_center: np.ndarray | None = None,
    circle_radius: float = 0.035,
    circle_omega: float = 1.5,
) -> TrackingResult:
    """Position-only differential IK loop (headless, kinematic integration)."""
    ctrl = make_ik("dls", damping=DLS_DAMPING)
    robot.reset()
    robot.set_arm_qpos(NOMINAL_Q.copy())
    robot.forward()
    ee_home = robot.get_ee_pos()

    if test is TrackingTest.REACH:
        target = random_proximal_target(ee_home, rng)
        max_steps = MAX_STEPS_REACH
    else:
        center = circle_center if circle_center is not None else ee_home.copy()
        target = circle_target(center, circle_radius, phase=0.0)
        max_steps = MAX_STEPS_CIRCLE

    pos_errors: list[float] = []

    for step in range(max_steps):
        if test is TrackingTest.CIRCLE:
            phase = circle_omega * step * 0.02
            target = circle_target(center, circle_radius, phase)

        ee_pos = robot.get_ee_pos()
        dx = target - ee_pos
        err = float(np.linalg.norm(dx))
        pos_errors.append(err)

        if test is TrackingTest.REACH and err < POS_TOL_M:
            break

        jac = robot.position_jacobian()
        dq = ctrl(jac, dx)
        q = robot.get_arm_qpos() + INTEGRATION_GAIN * dq
        robot.set_arm_qpos(clip_arm_qpos(q, q_min, q_max))
        robot.forward()

    final_err = pos_errors[-1] if pos_errors else float("inf")
    success = final_err < POS_TOL_M if test is TrackingTest.REACH else final_err < circle_radius * 0.75
    return TrackingResult(
        test=test,
        backend=robot.backend,
        steps=len(pos_errors),
        final_pos_err_m=final_err,
        success=success,
        mean_pos_err_m=float(np.mean(pos_errors)),
    )


def format_result(result: TrackingResult) -> str:
    status = "ok" if result.success else "FAIL"
    return (
        f"[{result.backend}] {result.test.value:6s}  "
        f"steps={result.steps:4d}  "
        f"final_err={result.final_pos_err_m * 1e3:6.2f} mm  "
        f"mean_err={result.mean_pos_err_m * 1e3:6.2f} mm  "
        f"{status}"
    )
