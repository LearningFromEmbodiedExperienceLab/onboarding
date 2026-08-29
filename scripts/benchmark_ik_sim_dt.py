"""Find maximum sim dt for stable dynamic IK on Piper (MuJoCo + Motrix).

Uses position actuators: each tick sets ``ctrl`` from differential IK, then
``mj_step`` / ``step`` (not kinematic ``qpos`` writes). Sweeps ``timestep`` and,
on MuJoCo, all four integrators.

Run::

    uv sync --extra sim
    uv run python scripts/benchmark_ik_sim_dt.py
    uv run python scripts/benchmark_ik_sim_dt.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

from ik_tracking_common import (
    ARM_DOF,
    ARM_JOINT_NAMES,
    MAX_STEPS_CIRCLE,
    MAX_STEPS_REACH,
    NOMINAL_Q,
    POS_TOL_M,
    circle_target,
    clip_arm_qpos,
    random_proximal_target,
)
from robotics.ik import make_ik
from sim_common import EE_LINK, require_piper_scene, section

# Kinematic demo advances circle phase as if each tick were 20 ms.
NOMINAL_CTRL_DT = 0.02

MAX_QVEL_RAD_S = 20.0

MUJOCO_INTEGRATORS: dict[str, int] = {
    "euler": 0,
    "rk4": 1,
    "implicit": 2,
    "implicitfast": 3,
}

# Sweep coarse → fine near Menagerie default (2 ms).
DT_CANDIDATES = [
    0.0005,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
]

QUICK_DT_CANDIDATES = [0.002, 0.005, 0.01, 0.02, 0.05]


@dataclass(frozen=True)
class TrialResult:
    backend: str
    integrator: str
    dt: float
    test: str
    success: bool
    stable: bool
    final_err_mm: float
    max_qvel: float
    steps: int


def _ensure_headless() -> None:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def _scaled_steps(base_steps: int, dt: float) -> int:
    return max(base_steps, int(base_steps * NOMINAL_CTRL_DT / dt))


def _sync_all_actuator_ctrl_mujoco(model, data) -> None:
    """Hold every actuator at its current joint target (avoids stale ctrl on gripper)."""
    for i in range(model.nu):
        trntype = model.actuator_trntype[i]
        if trntype != 0:  # mjTRN_JOINT
            continue
        joint_id = model.actuator_trnid[i, 0]
        qadr = model.jnt_qposadr[joint_id]
        data.ctrl[i] = data.qpos[qadr]


class MujocoDynamicIk:
    backend = "mujoco"

    def __init__(self, model, data, *, dt: float, integrator: int) -> None:
        import mujoco as mj

        self._mj = mj
        self.model = model
        self.data = data
        self.model.opt.timestep = dt
        self.model.opt.integrator = integrator
        self.dt = dt
        self._ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
        self._act_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_JOINT_NAMES
        ]
        self._q_min = np.array([model.jnt_range[i, 0] for i in range(ARM_DOF)])
        self._q_max = np.array([model.jnt_range[i, 1] for i in range(ARM_DOF)])
        self._ctrl = make_ik("dls", damping=1e-2)
        self.integrator_name = next(
            k for k, v in MUJOCO_INTEGRATORS.items() if v == integrator
        )

    def reset(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[:ARM_DOF] = q
        _sync_all_actuator_ctrl_mujoco(self.model, self.data)

    def get_arm_qpos(self) -> np.ndarray:
        return np.array(self.data.qpos[:ARM_DOF], dtype=float)

    def get_arm_qvel(self) -> np.ndarray:
        return np.array(self.data.qvel[:ARM_DOF], dtype=float)

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self._ee_id], dtype=float)

    def forward(self) -> None:
        self._mj.mj_forward(self.model, self.data)

    def position_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mj.mj_jacBody(self.model, self.data, jacp, jacr, self._ee_id)
        return jacp[:, :ARM_DOF].copy()

    def apply_ik_step(self, target: np.ndarray, gain: float = 0.35) -> float:
        ee_pos = self.get_ee_pos()
        dx = target - ee_pos
        err = float(np.linalg.norm(dx))
        dq = self._ctrl(self.position_jacobian(), dx)
        q_des = clip_arm_qpos(self.get_arm_qpos() + gain * dq, self._q_min, self._q_max)
        for act_id, q_i in zip(self._act_ids, q_des, strict=True):
            self.data.ctrl[act_id] = q_i
        self._mj.mj_step(self.model, self.data)
        return err

    def is_stable(self) -> bool:
        q = self.get_arm_qpos()
        qv = self.get_arm_qvel()
        return bool(np.isfinite(q).all() and np.isfinite(qv).all() and np.max(np.abs(qv)) < MAX_QVEL_RAD_S)


def _sync_all_actuator_ctrl_motrix(model, data) -> None:
    for name in model.actuator_names:
        act = model.get_actuator(name)
        joint = model.get_joint(act.target_name)
        q = float(np.asarray(joint.get_dof_pos(data)).reshape(-1)[0])
        act.set_ctrl(data, q)


class MotrixDynamicIk:
    backend = "motrix"
    integrator_name = "default"

    def __init__(self, model, data, *, dt: float) -> None:
        from motrixsim import forward_kinematic, step

        self._forward = forward_kinematic
        self._step = step
        self.model = model
        self.data = data
        self.model.options.timestep = dt
        self.dt = dt
        self._joints = [model.get_joint(name) for name in ARM_JOINT_NAMES]
        self._acts = [model.get_actuator(name) for name in ARM_JOINT_NAMES]
        self._ee = model.get_link(EE_LINK)
        self._q_min = np.array([float(j.range[0][0]) for j in self._joints])
        self._q_max = np.array([float(j.range[0][1]) for j in self._joints])
        self._ctrl = make_ik("dls", damping=1e-2)

    def reset(self) -> None:
        self.data.reset(self.model)

    def set_arm_qpos(self, q: np.ndarray) -> None:
        for joint, qi in zip(self._joints, q, strict=True):
            joint.set_dof_pos(self.data, float(qi))
        _sync_all_actuator_ctrl_motrix(self.model, self.data)

    def get_arm_qpos(self) -> np.ndarray:
        return np.array(
            [
                float(np.asarray(j.get_dof_pos(self.data)).reshape(-1)[0])
                for j in self._joints
            ],
            dtype=float,
        )

    def get_arm_qvel(self) -> np.ndarray:
        return np.asarray(self.data.dof_vel[:ARM_DOF], dtype=float).reshape(-1)

    def get_ee_pos(self) -> np.ndarray:
        return np.asarray(self._ee.get_position(self.data), dtype=float).reshape(3)

    def forward(self) -> None:
        self._forward(self.model, self.data)

    def position_jacobian(self, eps: float = 1e-6) -> np.ndarray:
        q0 = self.get_arm_qpos()
        jac = np.zeros((3, ARM_DOF), dtype=float)
        for i in range(ARM_DOF):
            q_plus = q0.copy()
            q_minus = q0.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            self.set_arm_qpos(q_plus)
            self.forward()
            p_plus = self.get_ee_pos()
            self.set_arm_qpos(q_minus)
            self.forward()
            p_minus = self.get_ee_pos()
            jac[:, i] = (p_plus - p_minus) / (2.0 * eps)
        self.set_arm_qpos(q0)
        self.forward()
        return jac

    def apply_ik_step(self, target: np.ndarray, gain: float = 0.35) -> float:
        ee_pos = self.get_ee_pos()
        dx = target - ee_pos
        err = float(np.linalg.norm(dx))
        dq = self._ctrl(self.position_jacobian(), dx)
        q_des = clip_arm_qpos(self.get_arm_qpos() + gain * dq, self._q_min, self._q_max)
        for act, q_i in zip(self._acts, q_des, strict=True):
            act.set_ctrl(self.data, float(q_i))
        self._step(self.model, self.data)
        return err

    def is_stable(self) -> bool:
        q = self.get_arm_qpos()
        qv = self.get_arm_qvel()
        return bool(np.isfinite(q).all() and np.isfinite(qv).all() and np.max(np.abs(qv)) < MAX_QVEL_RAD_S)


def _run_reach(robot, *, dt: float, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    robot.reset()
    robot.set_arm_qpos(NOMINAL_Q.copy())
    robot.forward()
    target = random_proximal_target(robot.get_ee_pos(), rng)
    max_steps = _scaled_steps(MAX_STEPS_REACH, dt)
    max_qvel = 0.0
    final_err = float("inf")

    for step in range(max_steps):
        if not robot.is_stable():
            return TrialResult(
                robot.backend,
                getattr(robot, "integrator_name", "default"),
                dt,
                "reach",
                False,
                False,
                final_err * 1e3,
                max_qvel,
                step,
            )
        final_err = robot.apply_ik_step(target)
        max_qvel = max(max_qvel, float(np.max(np.abs(robot.get_arm_qvel()))))
        if final_err < POS_TOL_M:
            break

    stable = robot.is_stable()
    success = stable and final_err < POS_TOL_M
    return TrialResult(
        robot.backend,
        getattr(robot, "integrator_name", "default"),
        dt,
        "reach",
        success,
        stable,
        final_err * 1e3,
        max_qvel,
        step + 1,
    )


def _run_circle(robot, *, dt: float, seed: int) -> TrialResult:
    robot.reset()
    robot.set_arm_qpos(NOMINAL_Q.copy())
    robot.forward()
    center = robot.get_ee_pos().copy()
    radius = 0.035
    max_steps = _scaled_steps(MAX_STEPS_CIRCLE, dt)
    max_qvel = 0.0
    final_err = float("inf")

    for step in range(max_steps):
        if not robot.is_stable():
            return TrialResult(
                robot.backend,
                getattr(robot, "integrator_name", "default"),
                dt,
                "circle",
                False,
                False,
                final_err * 1e3,
                max_qvel,
                step,
            )
        sim_time = (step + 1) * dt
        phase = 1.5 * sim_time
        target = circle_target(center, radius, phase)
        final_err = robot.apply_ik_step(target)
        max_qvel = max(max_qvel, float(np.max(np.abs(robot.get_arm_qvel()))))

    stable = robot.is_stable()
    success = stable and final_err < radius * 0.75
    return TrialResult(
        robot.backend,
        getattr(robot, "integrator_name", "default"),
        dt,
        "circle",
        success,
        stable,
        final_err * 1e3,
        max_qvel,
        max_steps,
    )


def _max_stable_dt(results: list[TrialResult]) -> float | None:
    by_dt: dict[float, list[TrialResult]] = {}
    for r in results:
        by_dt.setdefault(r.dt, []).append(r)
    ok_dts = [
        dt
        for dt, trials in sorted(by_dt.items())
        if all(t.success and t.stable for t in trials)
    ]
    return max(ok_dts) if ok_dts else None


def _run_mujoco(dt: float, integrator: str, seed: int) -> list[TrialResult]:
    import mujoco as mj

    scene = require_piper_scene()
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    robot = MujocoDynamicIk(
        model, data, dt=dt, integrator=MUJOCO_INTEGRATORS[integrator]
    )
    return [
        _run_reach(robot, dt=dt, seed=seed),
        _run_circle(robot, dt=dt, seed=seed),
    ]


def _run_motrix(dt: float, seed: int) -> list[TrialResult]:
    from motrixsim import SceneData, load_model

    scene = require_piper_scene()
    model = load_model(str(scene))
    data = SceneData(model)
    robot = MotrixDynamicIk(model, data, dt=dt)
    try:
        return [
            _run_reach(robot, dt=dt, seed=seed),
            _run_circle(robot, dt=dt, seed=seed),
        ]
    except Exception as exc:  # Motrix may panic on extreme dt
        print(f"  motrix dt={dt:.4f} error: {exc}", file=sys.stderr)
        return [
            TrialResult("motrix", "default", dt, "reach", False, False, float("inf"), 0.0, 0),
            TrialResult("motrix", "default", dt, "circle", False, False, float("inf"), 0.0, 0),
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fewer dt samples (faster smoke run)",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_headless()
    args = parse_args()
    dts = QUICK_DT_CANDIDATES if args.quick else DT_CANDIDATES

    section("dynamic IK sim-dt benchmark")
    print(f"seed          {args.seed}")
    print(f"ctrl model    position actuators + DLS IK per step")
    print(f"nominal tick  {NOMINAL_CTRL_DT * 1e3:.0f} ms (kinematic reference)")
    print(f"success       reach < {POS_TOL_M * 1e3:.0f} mm, circle < 26 mm, |qvel| < {MAX_QVEL_RAD_S}")

    all_results: list[TrialResult] = []

    section("MuJoCo")
    print(f"{'integrator':<12} {'dt (ms)':>8}  {'reach':>6}  {'circle':>6}  {'reach err':>10}  {'circle err':>10}  {'max qvel':>8}")
    for integrator in MUJOCO_INTEGRATORS:
        integrator_results: list[TrialResult] = []
        for dt in dts:
            trials = _run_mujoco(dt, integrator, args.seed)
            integrator_results.extend(trials)
            all_results.extend(trials)
            reach, circle = trials
            print(
                f"{integrator:<12} {dt * 1e3:8.2f}  "
                f"{'ok' if reach.success else 'FAIL':>6}  "
                f"{'ok' if circle.success else 'FAIL':>6}  "
                f"{reach.final_err_mm:10.2f}  {circle.final_err_mm:10.2f}  "
                f"{max(reach.max_qvel, circle.max_qvel):8.2f}"
            )
        max_dt = _max_stable_dt(integrator_results)
        label = f"{max_dt * 1e3:.2f} ms" if max_dt is not None else "none"
        print(f"  → max stable dt ({integrator}): {label}\n")

    section("Motrix")
    print("(single integrator — MJCF <option integrator> is ignored by Motrix)")
    print(f"{'integrator':<12} {'dt (ms)':>8}  {'reach':>6}  {'circle':>6}  {'reach err':>10}  {'circle err':>10}  {'max qvel':>8}")
    motrix_results: list[TrialResult] = []
    for dt in dts:
        trials = _run_motrix(dt, args.seed)
        motrix_results.extend(trials)
        all_results.extend(trials)
        reach, circle = trials
        print(
            f"{'default':<12} {dt * 1e3:8.2f}  "
            f"{'ok' if reach.success else 'FAIL':>6}  "
            f"{'ok' if circle.success else 'FAIL':>6}  "
            f"{reach.final_err_mm:10.2f}  {circle.final_err_mm:10.2f}  "
            f"{max(reach.max_qvel, circle.max_qvel):8.2f}"
        )
    max_dt = _max_stable_dt(motrix_results)
    label = f"{max_dt * 1e3:.2f} ms" if max_dt is not None else "none"
    print(f"  → max stable dt (motrix): {label}")

    section("summary")
    for backend in ("mujoco", "motrix"):
        if backend == "motrix":
            subset = motrix_results
            integrators = ["default"]
        else:
            integrators = list(MUJOCO_INTEGRATORS)
        for integrator in integrators:
            if backend == "mujoco":
                subset = [r for r in all_results if r.backend == "mujoco" and r.integrator == integrator]
            max_stable = _max_stable_dt(subset)
            if max_stable is None:
                print(f"{backend}/{integrator}: no dt passed both tests")
            else:
                print(f"{backend}/{integrator}: max dt = {max_stable * 1e3:.2f} ms ({max_stable:.4f} s)")


if __name__ == "__main__":
    main()
