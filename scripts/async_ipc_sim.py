"""Async IPC demo — simulation process (high rate).

Integrates a tiny 1-DoF plant, reads ``cmd_q`` from shared memory each step,
and publishes ``q`` / ``qdot`` back. Run via ``async_ipc_demo.py``.
"""

from __future__ import annotations

import time

from async_ipc_common import (
    PLANT_DAMPING,
    PLANT_MASS,
    PLANT_STIFFNESS,
    SIM_DT,
    AsyncIpcBus,
    rate_sleep,
)


def sim_loop(bus: AsyncIpcBus, *, duration_s: float) -> None:
    q = 0.0
    qdot = 0.0
    sim_steps = 0
    deadline = time.perf_counter() + duration_s
    next_tick = time.perf_counter()

    while time.perf_counter() < deadline:
        snap = bus.read_snapshot()
        if snap.shutdown:
            break

        cmd_q = snap.cmd_q
        ctrl_ticks = snap.ctrl_ticks

        # Held position target (zero-order hold between controller updates).
        accel = (PLANT_STIFFNESS * (cmd_q - q) - PLANT_DAMPING * qdot) / PLANT_MASS
        qdot += SIM_DT * accel
        q += SIM_DT * qdot
        sim_steps += 1
        sim_time = sim_steps * SIM_DT

        bus.write_state(
            sim_steps=sim_steps,
            ctrl_ticks=ctrl_ticks,
            sim_time=sim_time,
            q=q,
            qdot=qdot,
            cmd_q=cmd_q,
            shutdown=False,
        )

        next_tick = rate_sleep(next_tick, SIM_DT)

    snap = bus.read_snapshot()
    bus.write_state(
        sim_steps=sim_steps,
        ctrl_ticks=snap.ctrl_ticks,
        sim_time=sim_steps * SIM_DT,
        q=q,
        qdot=qdot,
        cmd_q=snap.cmd_q,
        shutdown=True,
    )


def main() -> None:
    bus = AsyncIpcBus.attach()
    try:
        sim_loop(bus, duration_s=2.0)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
