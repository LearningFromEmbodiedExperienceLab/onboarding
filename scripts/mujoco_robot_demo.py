"""Load the tutorial Piper robot in MuJoCo: scene, state I/O, step.

Run::

    bash scripts/fetch_menagerie_assets.sh
    uv sync --extra sim
    uv run python scripts/mujoco_robot_demo.py

Headless: ``MUJOCO_GL=egl`` (or ``osmesa``) if no display.
"""

from __future__ import annotations

import os

import mujoco as mj
import numpy as np

from sim_common import EE_LINK, FIRST_ARM_ACTUATOR, require_piper_scene, section


def main() -> None:
    scene = require_piper_scene()
    section("1. Load model + data")
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    print(f"scene     {scene.name}  ({scene.parent.name}/)")
    print(f"timestep  {model.opt.timestep:.4f} s")
    print(f"nq nv nu  {model.nq} {model.nv} {model.nu}")
    actuators = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]
    print(f"actuators {actuators}")

    section("2. Kinematic write + FK (no dynamics)")
    data.qpos[:] = 0.0
    data.qpos[0] = 0.5  # joint1 — first arm DoF in qpos
    mj.mj_forward(model, data)
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
    print(f"qpos[0]   {data.qpos[0]:.3f}")
    print(f"{EE_LINK} xpos {np.array2string(data.xpos[ee_id], precision=4)}")

    section("3. Dynamic write (ctrl) + step")
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, FIRST_ARM_ACTUATOR)
    data.ctrl[act_id] = 0.8  # position target for Menagerie position actuator
    n_step = 200
    for _ in range(n_step):
        mj.mj_step(model, data)
    print(f"ctrl[{FIRST_ARM_ACTUATOR}] held at {data.ctrl[act_id]:.2f} for {n_step} steps")
    print(f"qpos[0]   {data.qpos[0]:.4f}")

    section("4. Lockstep control tick (read → ctrl → step)")
    from robotics.context import Timer

    target = 0.3
    timer = Timer("lockstep", print_on_exit=False)
    with timer:
        for _ in range(50):
            err = target - data.qpos[0]
            data.ctrl[act_id] = data.qpos[0] + 0.35 * err  # tiny P on position target
            mj.mj_step(model, data)
    print(f"target q0 {target:.2f}  final q0 {data.qpos[0]:.4f}")
    print(f"wall time   {timer.elapsed * 1e3:.2f} ms")

    section("done")
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print("tip: set MUJOCO_GL=egl for headless GPU rendering if you add a viewer later")


if __name__ == "__main__":
    main()
