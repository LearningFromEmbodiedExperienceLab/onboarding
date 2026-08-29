"""Async IPC demo — simulation process over ZeroMQ (high rate)."""

from __future__ import annotations

import sys
import time
from queue import Queue

from async_ipc_common import SIM_DT, RunSummary, integrate_plant, rate_sleep
from async_ipc_zmq_common import (
    SLOW_JOINER_WAIT_S,
    StateMsg,
    bind_state_publisher,
    close_sockets,
    connect_command_subscriber,
    drain_latest_command,
    make_context,
    publish_shutdown,
    publish_state,
    term_context,
    wait_for_command,
)


def sim_loop(
    *,
    duration_s: float,
    result_queue: Queue[RunSummary] | None = None,
    startup_timeout_s: float = 2.0,
) -> RunSummary:
    ctx = make_context()
    state_pub = bind_state_publisher(ctx)
    cmd_sub = connect_command_subscriber(ctx)
    time.sleep(SLOW_JOINER_WAIT_S)

    q = 0.0
    qdot = 0.0
    cmd_q = 0.0
    ctrl_ticks = 0
    sim_steps = 0

    first_cmd = wait_for_command(cmd_sub, timeout_s=startup_timeout_s)
    if first_cmd is None:
        print(
            "warning: no command received before startup timeout "
            f"({startup_timeout_s:.1f}s) — continuing with cmd_q=0",
            file=sys.stderr,
        )
    else:
        cmd_q = first_cmd.cmd_q
        ctrl_ticks = first_cmd.ctrl_ticks

    deadline = time.perf_counter() + duration_s
    next_tick = time.perf_counter()

    try:
        while time.perf_counter() < deadline:
            latest_cmd, shutdown = drain_latest_command(cmd_sub)
            if shutdown:
                break
            if latest_cmd is not None:
                cmd_q = latest_cmd.cmd_q
                ctrl_ticks = latest_cmd.ctrl_ticks

            q, qdot = integrate_plant(q, qdot, cmd_q)
            sim_steps += 1
            sim_time = sim_steps * SIM_DT

            publish_state(
                state_pub,
                StateMsg(
                    sim_steps=sim_steps,
                    ctrl_ticks=ctrl_ticks,
                    sim_time=sim_time,
                    q=q,
                    qdot=qdot,
                ),
            )
            next_tick = rate_sleep(next_tick, SIM_DT)

        publish_shutdown(state_pub)
    finally:
        close_sockets(state_pub, cmd_sub)
        term_context(ctx)

    summary = RunSummary(
        sim_steps=sim_steps,
        ctrl_ticks=ctrl_ticks,
        sim_time=sim_steps * SIM_DT,
        q=q,
        qdot=qdot,
        cmd_q=cmd_q,
    )
    if result_queue is not None:
        result_queue.put(summary)
    return summary


def main() -> None:
    sim_loop(duration_s=2.0)


if __name__ == "__main__":
    main()
