"""Differential IK tracking on the Piper arm — MuJoCo and Motrix, headless.

Two test cases:
  reach  — move to a random proximal (nearby) target position
  circle — track a horizontal circle in workspace

Run (after Menagerie fetch + sim extra)::

    bash scripts/fetch_menagerie_assets.sh
    uv sync --extra sim
    uv run python scripts/ik_tracking_demo.py
    uv run python scripts/ik_tracking_demo.py --backend motrix --test circle

No viewer is opened; this script never blocks on a live GUI.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ik_tracking_common import (
    ARM_DOF,
    ARM_JOINT_NAMES,
    TrackingTest,
    format_result,
    numerical_position_jacobian,
    run_differential_ik,
)
from sim_common import EE_LINK, require_piper_scene, section


def _ensure_headless() -> None:
    # Kinematic IK only — no viewer — but MuJoCo still loads a GL backend on import.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


class MujocoRobot:
    backend = "mujoco"

    def __init__(self, model, data) -> None:
        import mujoco as mj

        self._mj = mj
        self.model = model
        self.data = data
        self._ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
        self._q_min = np.array([model.jnt_range[i, 0] for i in range(ARM_DOF)])
        self._q_max = np.array([model.jnt_range[i, 1] for i in range(ARM_DOF)])

    @property
    def q_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_min, self._q_max

    def reset(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

    def get_arm_qpos(self) -> np.ndarray:
        return np.array(self.data.qpos[:ARM_DOF], dtype=float)

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[:ARM_DOF] = q

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self._ee_id], dtype=float)

    def forward(self) -> None:
        self._mj.mj_forward(self.model, self.data)

    def position_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mj.mj_jacBody(self.model, self.data, jacp, jacr, self._ee_id)
        return jacp[:, :ARM_DOF].copy()


class MotrixRobot:
    backend = "motrix"

    def __init__(self, model, data) -> None:
        from motrixsim import forward_kinematic

        self._forward_kinematic = forward_kinematic
        self.model = model
        self.data = data
        self._joints = [model.get_joint(name) for name in ARM_JOINT_NAMES]
        self._ee = model.get_link(EE_LINK)
        self._q_min = np.array([float(j.range[0][0]) for j in self._joints])
        self._q_max = np.array([float(j.range[0][1]) for j in self._joints])

    @property
    def q_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_min, self._q_max

    def reset(self) -> None:
        self.data.reset(self.model)

    def get_arm_qpos(self) -> np.ndarray:
        return np.array(
            [
                float(np.asarray(j.get_dof_pos(self.data)).reshape(-1)[0])
                for j in self._joints
            ],
            dtype=float,
        )

    def set_arm_qpos(self, q: np.ndarray) -> None:
        for joint, qi in zip(self._joints, q, strict=True):
            joint.set_dof_pos(self.data, float(qi))

    def get_ee_pos(self) -> np.ndarray:
        return np.asarray(self._ee.get_position(self.data), dtype=float).reshape(3)

    def forward(self) -> None:
        self._forward_kinematic(self.model, self.data)

    def position_jacobian(self) -> np.ndarray:
        return numerical_position_jacobian(
            self.get_ee_pos,
            self.get_arm_qpos,
            self.set_arm_qpos,
            self.forward,
            n_dof=ARM_DOF,
        )


def run_backend(name: str, tests: list[TrackingTest], seed: int) -> list[str]:
    scene = require_piper_scene()
    rng = np.random.default_rng(seed)
    lines: list[str] = []

    if name == "mujoco":
        import mujoco as mj

        model = mj.MjModel.from_xml_path(str(scene))
        data = mj.MjData(model)
        robot = MujocoRobot(model, data)
    elif name == "motrix":
        from motrixsim import SceneData, load_model

        model = load_model(str(scene))
        data = SceneData(model)
        robot = MotrixRobot(model, data)
    else:
        raise ValueError(name)

    q_min, q_max = robot.q_limits
    robot.reset()
    robot.set_arm_qpos(np.zeros(ARM_DOF))
    robot.forward()
    circle_center = robot.get_ee_pos().copy()

    for test in tests:
        result = run_differential_ik(
            robot,
            test=test,
            q_min=q_min,
            q_max=q_max,
            rng=rng,
            circle_center=circle_center,
        )
        lines.append(format_result(result))

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("mujoco", "motrix", "both"),
        default="both",
        help="simulator backend (default: both)",
    )
    parser.add_argument(
        "--test",
        choices=("reach", "circle", "all"),
        default="all",
        help="tracking test case (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    return parser.parse_args()


def main() -> None:
    _ensure_headless()
    args = parse_args()

    if args.test == "all":
        tests = [TrackingTest.REACH, TrackingTest.CIRCLE]
    else:
        tests = [TrackingTest(args.test)]

    backends = ["mujoco", "motrix"] if args.backend == "both" else [args.backend]

    section("Differential IK tracking (headless)")
    print(f"EE link   {EE_LINK}")
    print(f"tests     {[t.value for t in tests]}")
    print(f"backends  {backends}")
    print(f"seed      {args.seed}")

    all_ok = True
    for backend in backends:
        section(backend)
        for line in run_backend(backend, tests, seed=args.seed):
            print(line)
            if line.endswith("FAIL"):
                all_ok = False

    section("done")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
