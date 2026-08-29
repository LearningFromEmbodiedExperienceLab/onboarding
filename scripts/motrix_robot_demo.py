"""Load the same Piper scene in Motrix: named access, state I/O, step.

Run::

    bash scripts/fetch_menagerie_assets.sh
    uv sync --extra sim
    uv run python scripts/motrix_robot_demo.py

Same MJCF as ``mujoco_robot_demo.py`` — compare APIs side by side.
"""

from __future__ import annotations

import numpy as np
from motrixsim import SceneData, forward_kinematic, load_model, step

from sim_common import EE_LINK, FIRST_ARM_ACTUATOR, require_piper_scene, section


def main() -> None:
    scene = require_piper_scene()
    section("1. Load SceneModel + SceneData")
    model = load_model(str(scene))
    data = SceneData(model)
    print(f"scene       {scene.name}  ({scene.parent.name}/)")
    print(f"num_dof     {data.dof_pos.shape[0]}")
    print(f"actuators   {model.actuator_names}")
    print(f"joints      {[j.name for j in model.joints]}")

    section("2. Kinematic write + FK (no dynamics)")
    data.reset(model)
    joint1 = model.get_joint("joint1")
    joint1.set_dof_pos(data, 0.5)
    forward_kinematic(model, data)
    ee = model.get_link(EE_LINK)
    pos = ee.get_position(data)
    q1 = float(np.asarray(joint1.get_dof_pos(data)).reshape(-1)[0])
    print(f"joint1 pos  {q1:.3f}")
    print(f"{EE_LINK} pos   {np.array2string(pos, precision=4)}")

    section("3. Dynamic write (ctrl) + step")
    data.reset(model)
    act = model.get_actuator(FIRST_ARM_ACTUATOR)
    target = 0.8
    n_step = 200
    for _ in range(n_step):
        act.set_ctrl(data, target)
        step(model, data)
    print(f"ctrl target {target:.2f} for {n_step} steps")
    q1 = float(np.asarray(joint1.get_dof_pos(data)).reshape(-1)[0])
    print(f"joint1 pos  {q1:.4f}")
    print(
        "note: Motrix may warn on unsupported MJCF <option> tags in Menagerie; "
        "dynamics can differ slightly from MuJoCo."
    )

    section("4. Lockstep control tick (read → ctrl → step)")
    from robotics.context import Timer

    target = 0.3
    timer = Timer("lockstep", print_on_exit=False)
    with timer:
        for _ in range(50):
            q = float(np.asarray(joint1.get_dof_pos(data)).reshape(-1)[0])
            act.set_ctrl(data, q + 0.35 * (target - q))
            step(model, data)
    q_final = float(np.asarray(joint1.get_dof_pos(data)).reshape(-1)[0])
    print(f"target q0 {target:.2f}  final q0 {q_final:.4f}")
    print(f"wall time   {timer.elapsed * 1e3:.2f} ms")

    section("done")


if __name__ == "__main__":
    main()
