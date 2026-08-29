"""Compare position vs motor actuators with control-rate vs substep commands.

Menagerie Piper ships with ``<position>`` actuators (``ctrl`` = target joint
angle). This demo swaps in ``<motor>`` actuators (``ctrl`` = joint torque) for
the arm and runs the same seed-0 differential-IK reach at 50 Hz with ten 2 ms
physics substeps.

Shows:
  - **Position:** ``q_des`` updated at controller rate, held across substeps — OK
    (inner PD runs inside ``step`` every substep).
  - **Motor:** ``tau = PD(q_des, q)`` computed once per control tick and held —
    FAIL (zero-order hold on torque goes unstable as ``q`` moves).
  - **Motor:** ``tau`` recomputed every substep — OK (torque loop matches physics).

Run::

    uv sync --extra sim
    uv run python scripts/actuator_control_demo.py
"""

from __future__ import annotations

import os
import re
import sys

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

# Match Menagerie Piper position actuator gains (used for explicit motor PD).
ARM_KP = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0])
ARM_KV = np.array([5.0, 5.0, 5.0, 5.0, 1.5, 1.5])
TAU_LIMIT = 100.0

DT_SIM = 0.002
N_SUBSTEPS = 10
DT_CTRL = DT_SIM * N_SUBSTEPS
IK_GAIN = 0.35
MAX_QVEL = 20.0


def _load_piper_motor_model():
    """Piper arm with ``<motor>`` actuators on joint1–joint6 (gripper stays position)."""
    import mujoco as mj

    piper = require_piper_scene().parent / "piper.xml"
    xml = piper.read_text()
    xml = re.sub(
        r'<position name="(joint[1-6])" joint="\1" class="piper" kp="(\d+)" kv="([\d.]+)"/>',
        r'<motor name="\1" joint="\1" class="piper" ctrlrange="-100 100"/>',
        xml,
    )
    tmp = piper.parent / "_piper_motor_tutorial.xml"
    tmp.write_text(xml)
    try:
        return mj.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _motor_pd(q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    tau = ARM_KP * (q_des - q) - ARM_KV * qd
    return np.clip(tau, -TAU_LIMIT, TAU_LIMIT)


def _run_reach(
    *,
    actuator: str,
    torque_rate: str,
    seed: int,
) -> tuple[bool, float, float, int]:
    """Return (success, final_err_mm, peak_qvel, control_steps)."""
    import mujoco as mj

    if actuator == "motor":
        model = _load_piper_motor_model()
    elif actuator == "position":
        model = mj.MjModel.from_xml_path(str(require_piper_scene()))
    else:
        raise ValueError(actuator)

    data = mj.MjData(model)
    model.opt.timestep = DT_SIM
    model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
    act_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINT_NAMES
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
    max_ctrl_steps = max(1, int(MAX_STEPS_REACH * 0.02 / DT_CTRL))

    final_err = float("inf")
    peak_qvel = 0.0
    ctrl_steps = 0

    for _ in range(max_ctrl_steps):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, ee_id)
        ee_pos = data.xpos[ee_id].copy()
        dx = target - ee_pos
        final_err = float(np.linalg.norm(dx))
        dq = ik(jacp[:, :ARM_DOF], dx)
        q_des = clip_arm_qpos(data.qpos[:ARM_DOF] + IK_GAIN * dq, q_min, q_max)

        if actuator == "position":
            for act_id, q_i in zip(act_ids, q_des, strict=True):
                data.ctrl[act_id] = q_i
            data.ctrl[grip_id] = data.qpos[j7_adr]
            for _ in range(N_SUBSTEPS):
                mj.mj_step(model, data)
                peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))))
        else:
            if torque_rate == "control":
                tau = _motor_pd(q_des, data.qpos[:ARM_DOF], data.qvel[:ARM_DOF])
                for act_id, t_i in zip(act_ids, tau, strict=True):
                    data.ctrl[act_id] = t_i
                data.ctrl[grip_id] = data.qpos[j7_adr]
                for _ in range(N_SUBSTEPS):
                    mj.mj_step(model, data)
                    peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))))
            elif torque_rate == "substep":
                data.ctrl[grip_id] = data.qpos[j7_adr]
                for _ in range(N_SUBSTEPS):
                    tau = _motor_pd(q_des, data.qpos[:ARM_DOF], data.qvel[:ARM_DOF])
                    for act_id, t_i in zip(act_ids, tau, strict=True):
                        data.ctrl[act_id] = t_i
                    mj.mj_step(model, data)
                    peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))))
            else:
                raise ValueError(torque_rate)

        ctrl_steps += 1
        if final_err < POS_TOL_M:
            break
        if peak_qvel > MAX_QVEL or not np.isfinite(data.qpos[:ARM_DOF]).all():
            break

    stable = peak_qvel < MAX_QVEL and np.isfinite(data.qpos[:ARM_DOF]).all()
    ok = stable and final_err < POS_TOL_M
    return ok, final_err * 1e3, peak_qvel, ctrl_steps


def main() -> None:
    seed = 0
    section("actuator comparison — dynamic IK reach (seed 0)")
    print(f"physics     Δt_sim = {DT_SIM * 1e3:.0f} ms, implicitfast")
    print(f"control     Δt_ctrl = {DT_CTRL * 1e3:.0f} ms (50 Hz), n_sub = {N_SUBSTEPS}")
    print(f"pass        reach err < {POS_TOL_M * 1e3:.0f} mm, peak |qvel| < {MAX_QVEL}")

    cases = [
        ("position", "position", "control", "ctrl = q_des @ 50 Hz, held 10 substeps"),
        ("motor", "motor", "control", "ctrl = tau @ 50 Hz (PD once), held 10 substeps"),
        ("motor", "motor", "substep", "ctrl = tau recomputed every 2 ms substep"),
    ]

    results: dict[str, bool] = {}
    for label, act, rate, desc in cases:
        ok, err, qv, steps = _run_reach(actuator=act, torque_rate=rate, seed=seed)
        key = f"{act}/{rate}"
        results[key] = ok
        status = "ok" if ok else "FAIL"
        print(
            f"{label:8s} {rate:7s}  {status:4s}  err={err:6.1f} mm  "
            f"peak |qvel|={qv:6.1f}  steps={steps:3d}  — {desc}"
        )

    section("done")
    if not (results["position/control"] and results["motor/substep"]):
        raise SystemExit(1)
    if results["motor/control"]:
        print("warning: expected motor/control-rate hold to fail", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
