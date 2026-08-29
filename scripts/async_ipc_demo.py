"""Launch async sim + controller in separate processes (shared-memory IPC).

The simulation integrates a 1-DoF plant at 500 Hz. The controller reads state
and writes a new position target at 50 Hz. Between controller updates the sim
**holds** the last ``cmd_q`` — the same zero-order hold as the async loop in
``docs/the-simulation-loop.qmd``.

Run::

    uv run python scripts/async_ipc_demo.py

No MuJoCo, network ports, or extra dependencies — stdlib ``multiprocessing`` +
``shared_memory`` only.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from async_ipc_common import CTRL_HZ, SIM_HZ, AsyncIpcBus
from async_ipc_controller import controller_loop
from async_ipc_sim import sim_loop
from sim_common import section


def _run_sim(duration_s: float) -> None:
    bus = AsyncIpcBus.attach()
    try:
        sim_loop(bus, duration_s=duration_s)
    finally:
        bus.close()


def _run_controller(duration_s: float) -> None:
    bus = AsyncIpcBus.attach()
    try:
        controller_loop(bus, duration_s=duration_s)
    finally:
        bus.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="wall-clock run time in seconds (default: 2.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    section("async IPC demo (shared memory)")
    print(f"sim rate      {SIM_HZ:.0f} Hz")
    print(f"controller    {CTRL_HZ:.0f} Hz")
    print(f"duration      {args.duration:.1f} s wall clock")

    bus = AsyncIpcBus.create()

    sim_proc = mp.Process(target=_run_sim, args=(args.duration,), name="sim")
    ctrl_proc = mp.Process(
        target=_run_controller, args=(args.duration,), name="controller"
    )

    t0 = time.perf_counter()
    sim_proc.start()
    ctrl_proc.start()
    sim_proc.join()
    ctrl_proc.join()
    elapsed = time.perf_counter() - t0

    snap = bus.read_snapshot()
    bus.close()
    bus.unlink()

    section("summary")
    ratio = snap.sim_steps / max(snap.ctrl_ticks, 1)
    print(f"sim steps     {snap.sim_steps}  (~{snap.sim_steps / elapsed:.0f} Hz)")
    print(f"ctrl ticks    {snap.ctrl_ticks}  (~{snap.ctrl_ticks / elapsed:.0f} Hz)")
    print(f"sim/ctrl      {ratio:.1f} steps per controller tick")
    print(f"final q       {snap.q:.4f} rad")
    print(f"final cmd_q   {snap.cmd_q:.4f} rad")
    print(f"sim_time      {snap.sim_time:.3f} s (integrated)")
    print(f"wall time     {elapsed:.3f} s")

    if snap.ctrl_ticks < 5 or snap.sim_steps < 50:
        raise SystemExit("too few ticks — demo may have failed to run")
    if ratio < 5:
        raise SystemExit(f"expected ~{SIM_HZ / CTRL_HZ:.0f}x sim steps per ctrl tick")


if __name__ == "__main__":
    main()
