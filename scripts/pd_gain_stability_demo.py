"""Show joint PD gain too large *or* too small on dynamic IK reach.

Uses dynamic IK + position actuators on Piper (seed-0 reach). At ``Δt_sim = 10 ms``
with Euler integration (lockstep):

  - Nominal Menagerie ``kp`` / ``kv`` → stable reach
  - ``5× kp`` → FAIL (numerically unstable — large |qvel|)
  - ``kv = 0`` → FAIL (under-damped oscillation)
  - ``0.2× kp`` → FAIL (stable but **sluggish** — cannot track ``q*``, large EE error)

Run::

    uv sync --extra sim
    uv run python scripts/pd_gain_stability_demo.py
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

DT_SIM = 0.01
INTEGRATOR = 0  # Euler
IK_GAIN = 0.35
MAX_QVEL = 20.0


def _load_piper_position_model(*, kp_scale: float, zero_kv: bool):
    import mujoco as mj

    piper = require_piper_scene().parent / "piper.xml"
    xml = piper.read_text()
    xml = re.sub(
        r'kp="(\d+)"',
        lambda m: f'kp="{max(1, int(float(m.group(1)) * kp_scale))}"',
        xml,
    )
    if zero_kv:
        xml = re.sub(r'kv="[\d.]+"', 'kv="0"', xml)
    tmp = piper.parent / "_piper_pd_gain_tutorial.xml"
    tmp.write_text(xml)
    try:
        return mj.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _run_reach(*, kp_scale: float, zero_kv: bool, seed: int) -> tuple[bool, float, float]:
    import mujoco as mj

    model = _load_piper_position_model(kp_scale=kp_scale, zero_kv=zero_kv)
    data = mj.MjData(model)
    model.opt.timestep = DT_SIM
    model.opt.integrator = INTEGRATOR

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

    target = random_proximal_target(
        data.xpos[ee_id].copy(), np.random.default_rng(seed)
    )
    max_steps = max(1, int(MAX_STEPS_REACH * 0.02 / DT_SIM))

    final_err = float("inf")
    peak_qvel = 0.0

    for _ in range(max_steps):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, ee_id)
        dx = target - data.xpos[ee_id]
        final_err = float(np.linalg.norm(dx))
        dq = ik(jacp[:, :ARM_DOF], dx)
        q_des = clip_arm_qpos(data.qpos[:ARM_DOF] + IK_GAIN * dq, q_min, q_max)
        for act_id, q_i in zip(act_ids, q_des, strict=True):
            data.ctrl[act_id] = q_i
        data.ctrl[grip_id] = data.qpos[j7_adr]
        mj.mj_step(model, data)
        peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))))
        if final_err < POS_TOL_M:
            break
        if peak_qvel > MAX_QVEL or not np.isfinite(data.qpos[:ARM_DOF]).all():
            break

    stable = peak_qvel < MAX_QVEL and np.isfinite(data.qpos[:ARM_DOF]).all()
    ok = stable and final_err < POS_TOL_M
    return ok, final_err * 1e3, peak_qvel


def main() -> None:
    seed = 0
    section("PD gain stability — dynamic IK reach (seed 0)")
    print(f"physics     Δt_sim = {DT_SIM * 1e3:.0f} ms, Euler, lockstep (1 step / IK tick)")
    print(f"pass        reach err < {POS_TOL_M * 1e3:.0f} mm, peak |qvel| < {MAX_QVEL}")

    cases = [
        (1.0, False, "nominal Menagerie kp/kv"),
        (0.2, False, "0.2× kp (sluggish servos — stable but won't track)"),
        (5.0, False, "5× kp (stiffer servos — unstable)"),
        (1.0, True, "kv = 0 (no joint damping)"),
    ]

    results: dict[str, bool] = {}
    sluggish: tuple[bool, float, float] | None = None
    for kp_scale, zero_kv, desc in cases:
        ok, err, qv = _run_reach(kp_scale=kp_scale, zero_kv=zero_kv, seed=seed)
        key = f"kp{kp_scale}_kv0={zero_kv}"
        results[key] = ok
        if kp_scale == 0.2 and not zero_kv:
            sluggish = (ok, err, qv)
        print(
            f"{'ok' if ok else 'FAIL':4s}  err={err:6.1f} mm  peak |qvel|={qv:6.1f}  — {desc}"
        )

    section("done")
    if not results["kp1.0_kv0=False"]:
        raise SystemExit(1)
    if results["kp5.0_kv0=False"]:
        print("warning: expected 5× kp case to fail", file=sys.stderr)
        raise SystemExit(1)
    if results["kp1.0_kv0=True"]:
        print("warning: expected kv=0 case to fail", file=sys.stderr)
        raise SystemExit(1)
    if results.get("kp0.2_kv0=False"):
        print("warning: expected 0.2× kp case to fail reach (sluggish tracking)", file=sys.stderr)
        raise SystemExit(1)
    if sluggish is not None and sluggish[2] > 5.0:
        print("warning: expected 0.2× kp to stay numerically calm (low |qvel|)", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
