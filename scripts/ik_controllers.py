"""Walk the IK controller patterns: registry, dunders, timer, seed.

Run::

    uv run python scripts/ik_controllers.py
"""

from __future__ import annotations

import numpy as np

from robotics.context import Timer, numpy_seed
from robotics.ik import IKConfig, IKRegistry, Pose, make_ik


def main() -> None:
    J = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.1, 1.0, 0.1],
            [0.0, 0.0, 1.0],
        ]
    )
    dx = np.array([0.1, 0.0, 0.0])

    print("registered:", list(IKRegistry.instance()))
    for name in IKRegistry.instance():
        ctrl = make_ik(name)
        with Timer(name):
            dq = ctrl(J, dx)
        print(f"  {ctrl!r:24s}  len={len(ctrl)}  dq={np.array2string(dq, precision=4)}")
        print(f"    ctrl[0]={ctrl[0]}")

    # Batched: three copies of J; index last dq per env.
    Jb = np.broadcast_to(J, (3, 3, 3)).copy()
    dxb = np.broadcast_to(dx, (3, 3)).copy()
    dls = make_ik("dls", damping=1e-2)
    dls(Jb, dxb)
    dls[1] = 0.0
    print("damped batch dq after dls[1]=0:\n", dls._dq)

    print("global rand without seed:", np.random.rand(2))
    with numpy_seed(0):
        a = np.random.rand(2)
    with numpy_seed(0):
        b = np.random.rand(2)
    print("inside numpy_seed(0) twice:", a, b, "equal", np.allclose(a, b))
    print("global rand after restore (not stuck at seed 0):", np.random.rand(2))

    pose = Pose(pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    p, q = pose
    print("Pose unpack:", p, q)
    cfg = IKConfig(method="dls", damping=1e-2)
    print("IKConfig.build() →", cfg.build())


if __name__ == "__main__":
    main()
