"""Async IPC demo — controller process (lower rate).

Reads plant state from shared memory, computes a new position target, and
writes ``cmd_q`` without blocking the sim process. Run via ``async_ipc_demo.py``.
"""

from __future__ import annotations

import time

from async_ipc_common import CTRL_DT, AsyncIpcBus, command_for_tick, rate_sleep


def controller_loop(bus: AsyncIpcBus, *, duration_s: float) -> None:
    ctrl_ticks = 0
    deadline = time.perf_counter() + duration_s
    next_tick = time.perf_counter()

    while time.perf_counter() < deadline:
        snap = bus.read_snapshot()
        if snap.shutdown:
            break

        cmd_q = command_for_tick(ctrl_ticks)

        ctrl_ticks += 1
        bus.write_command(cmd_q=cmd_q, ctrl_ticks=ctrl_ticks, shutdown=False)

        next_tick = rate_sleep(next_tick, CTRL_DT)

    snap = bus.read_snapshot()
    bus.write_command(cmd_q=snap.cmd_q, ctrl_ticks=ctrl_ticks, shutdown=True)


def main() -> None:
    bus = AsyncIpcBus.attach()
    try:
        controller_loop(bus, duration_s=2.0)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
