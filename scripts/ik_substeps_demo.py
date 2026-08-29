"""Compare unstable lockstep vs stable substepped dynamic IK (MuJoCo, headless).

Demonstrates the fix motivated by large-``dt`` instability: keep ``dt_sim`` small,
run multiple physics substeps per IK tick (hold ``ctrl``), decimate control rate.

Run::

    uv sync --extra sim
    uv run python scripts/ik_substeps_demo.py
"""

from __future__ import annotations

import os

import numpy as np

from ik_tracking_common import (
    ARM_DOF,
    ARM_JOINT_NAMES,
    MAX_STEPS_REACH,
    NOMINAL_Q,
    POS_TOL_M,
    clip_arm_qpos,
    random_proximal_target,
)
from robotics.ik import make_ik
from sim_common import EE_LINK, require_piper_scene, section

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

IK_GAIN = 0.35
MAX_QVEL = 20.0


def _run_reach(
    *,
    dt_sim: float,
    integrator: int,
    n_substeps: int,
    seed: int,
) -> tuple[bool, float, float, int]:
    import mujoco as mj

    scene = require_piper_scene()
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    model.opt.timestep = dt_sim
    model.opt.integrator = integrator

    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
    act_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINT_NAMES
    ]
    grip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "gripper")
    j7_adr = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "joint7")]

    q_min = np.array([model.jnt_range[i, 0] for i in range(ARM_DOF)])
    q_max = np.array([model.jnt_range[i, 1] for i in range(ARM_DOF)])
    ik = make_ik("dls", damping=1e-2)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:ARM_DOF] = NOMINAL_Q
    for i in range(model.nu):
        if model.actuator_trntype[i] == 0:
            joint_id = model.actuator_trnid[i, 0]
            data.ctrl[i] = data.qpos[model.jnt_qposadr[joint_id]]
    mj.mj_forward(model, data)

    rng = np.random.default_rng(seed)
    target = random_proximal_target(data.xpos[ee_id].copy(), rng)
    dt_ctrl = dt_sim * n_substeps
    max_steps_ctrl = max(1, int(MAX_STEPS_REACH * 0.02 / dt_ctrl))

    final_err = float("inf")
    peak_qvel = 0.0
    ctrl_steps = 0

    for _ in range(max_steps_ctrl):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, ee_id)
        ee_pos = data.xpos[ee_id].copy()
        dx = target - ee_pos
        final_err = float(np.linalg.norm(dx))
        dq = ik(jacp[:, :ARM_DOF], dx)
        q_des = clip_arm_qpos(data.qpos[:ARM_DOF] + IK_GAIN * dq, q_min, q_max)
        for act_id, q_i in zip(act_ids, q_des, strict=True):
            data.ctrl[act_id] = q_i
        data.ctrl[grip_id] = data.qpos[j7_adr]

        for _ in range(n_substeps):
            mj.mj_step(model, data)
            peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))))

        ctrl_steps += 1
        if final_err < POS_TOL_M:
            break
        if peak_qvel > MAX_QVEL or not np.isfinite(data.qpos[:ARM_DOF]).all():
            break

    stable = peak_qvel < MAX_QVEL and np.isfinite(data.qpos[:ARM_DOF]).all()
    ok = stable and final_err < POS_TOL_M
    return ok, final_err * 1e3, peak_qvel, ctrl_steps


def main() -> None:
    section("dynamic IK — lockstep vs substeps (Euler, seed 0)")
    seed = 0

    # Unstable: one 20 ms physics step per IK tick (same as instability video).
    bad_ok, bad_err, bad_qv, bad_steps = _run_reach(
        dt_sim=0.02, integrator=0, n_substeps=1, seed=seed
    )
    print(
        f"lockstep   dt_sim=20 ms, n_sub=1  →  "
        f"{'ok' if bad_ok else 'FAIL'}  err={bad_err:.1f} mm  "
        f"peak |qvel|={bad_qv:.1f}  ctrl_steps={bad_steps}"
    )

    # Stable: 2 ms physics, 10 substeps → 20 ms control period (50 Hz IK).
    good_ok, good_err, good_qv, good_steps = _run_reach(
        dt_sim=0.002, integrator=0, n_substeps=10, seed=seed
    )
    print(
        f"substeps   dt_sim=2 ms, n_sub=10 (dt_ctrl=20 ms)  →  "
        f"{'ok' if good_ok else 'FAIL'}  err={good_err:.1f} mm  "
        f"peak |qvel|={good_qv:.1f}  ctrl_steps={good_steps}"
    )

    section("done")
    if not good_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
