"""Async IPC demo — controller process over ZeroMQ (lower rate)."""

from __future__ import annotations

import time

from async_ipc_common import CTRL_DT, command_for_tick, rate_sleep
from async_ipc_zmq_common import (
    SLOW_JOINER_WAIT_S,
    CmdMsg,
    bind_command_publisher,
    close_sockets,
    connect_state_subscriber,
    drain_latest_state,
    make_context,
    publish_command,
    publish_shutdown,
    term_context,
)


def controller_loop(*, duration_s: float) -> int:
    ctx = make_context()
    cmd_pub = bind_command_publisher(ctx)
    state_sub = connect_state_subscriber(ctx)
    time.sleep(SLOW_JOINER_WAIT_S)

    ctrl_ticks = 0
    deadline = time.perf_counter() + duration_s
    next_tick = time.perf_counter()

    try:
        while time.perf_counter() < deadline:
            _state, shutdown = drain_latest_state(state_sub)
            if shutdown:
                break

            cmd_q = command_for_tick(ctrl_ticks)
            ctrl_ticks += 1
            publish_command(cmd_pub, CmdMsg(ctrl_ticks=ctrl_ticks, cmd_q=cmd_q))
            next_tick = rate_sleep(next_tick, CTRL_DT)

        publish_shutdown(cmd_pub)
    finally:
        close_sockets(cmd_pub, state_sub)
        term_context(ctx)

    return ctrl_ticks


def main() -> None:
    ticks = controller_loop(duration_s=2.0)
    print(f"controller ticks: {ticks}")


if __name__ == "__main__":
    main()
