"""ZeroMQ message helpers for the async IPC demo."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import zmq

STATE_ADDR = "tcp://127.0.0.1:15555"
CMD_ADDR = "tcp://127.0.0.1:15556"

STATE_TOPIC = b"state"
CMD_TOPIC = b"cmd"
SHUTDOWN_TOPIC = b"shutdown"

# state: sim_steps, ctrl_ticks, sim_time, q, qdot  (2× int64, 3× float64)
_STATE = struct.Struct("<qqddd")
# cmd: ctrl_ticks, cmd_q  (int64, float64)
_CMD = struct.Struct("<qd")

# Socket timeouts (ms). Prevents blocking forever on a dead peer.
RCV_TIMEOUT_MS = 100
SND_TIMEOUT_MS = 100
LINGER_MS = 0

# Parent waits for child processes: run duration + this slack.
JOIN_GRACE_S = 3.0

# After SUB connect / PUB bind, allow slow-joiner handshake (see ZMQ docs).
SLOW_JOINER_WAIT_S = 0.25


@dataclass(frozen=True)
class StateMsg:
    sim_steps: int
    ctrl_ticks: int
    sim_time: float
    q: float
    qdot: float


@dataclass(frozen=True)
class CmdMsg:
    ctrl_ticks: int
    cmd_q: float


def make_context() -> zmq.Context:
    return zmq.Context()


def _configure_pub(pub: zmq.Socket) -> None:
    pub.setsockopt(zmq.LINGER, LINGER_MS)
    pub.setsockopt(zmq.SNDTIMEO, SND_TIMEOUT_MS)


def _configure_sub(sub: zmq.Socket) -> None:
    sub.setsockopt(zmq.LINGER, LINGER_MS)
    sub.setsockopt(zmq.RCVTIMEO, RCV_TIMEOUT_MS)
    # Do not use CONFLATE here — with multipart topics it can drop frames
    # before poll/recv sees them. We keep the latest message in drain_* instead.


def publish_state(pub: zmq.Socket, msg: StateMsg) -> None:
    payload = _STATE.pack(
        msg.sim_steps,
        msg.ctrl_ticks,
        msg.sim_time,
        msg.q,
        msg.qdot,
    )
    pub.send_multipart([STATE_TOPIC, payload])


def publish_command(pub: zmq.Socket, msg: CmdMsg) -> None:
    payload = _CMD.pack(msg.ctrl_ticks, msg.cmd_q)
    pub.send_multipart([CMD_TOPIC, payload])


def publish_shutdown(pub: zmq.Socket) -> None:
    pub.send_multipart([SHUTDOWN_TOPIC, b""])


def _parse_state(topic: bytes, payload: bytes) -> tuple[StateMsg | None, bool]:
    if topic == SHUTDOWN_TOPIC:
        return None, True
    if topic != STATE_TOPIC:
        return None, False
    sim_steps, ctrl_ticks, sim_time, q, qdot = _STATE.unpack(payload)
    return (
        StateMsg(
            sim_steps=int(sim_steps),
            ctrl_ticks=int(ctrl_ticks),
            sim_time=float(sim_time),
            q=float(q),
            qdot=float(qdot),
        ),
        False,
    )


def _parse_cmd(topic: bytes, payload: bytes) -> tuple[CmdMsg | None, bool]:
    if topic == SHUTDOWN_TOPIC:
        return None, True
    if topic != CMD_TOPIC:
        return None, False
    ctrl_ticks, cmd_q = _CMD.unpack(payload)
    return CmdMsg(ctrl_ticks=int(ctrl_ticks), cmd_q=float(cmd_q)), False


def drain_latest_state(sub: zmq.Socket, *, poll_ms: int = 0) -> tuple[StateMsg | None, bool]:
    """Return the newest state frame and whether sim sent shutdown."""
    latest: StateMsg | None = None
    shutdown = False
    while sub.poll(poll_ms):
        topic, payload = sub.recv_multipart()
        parsed, is_shutdown = _parse_state(topic, payload)
        if is_shutdown:
            shutdown = True
            continue
        if parsed is not None:
            latest = parsed
        poll_ms = 0  # drain any backlog without waiting again
    return latest, shutdown


def drain_latest_command(sub: zmq.Socket, *, poll_ms: int = 0) -> tuple[CmdMsg | None, bool]:
    """Return the newest command and whether a shutdown was received."""
    latest: CmdMsg | None = None
    shutdown = False
    while sub.poll(poll_ms):
        topic, payload = sub.recv_multipart()
        parsed, is_shutdown = _parse_cmd(topic, payload)
        if is_shutdown:
            shutdown = True
            continue
        if parsed is not None:
            latest = parsed
        poll_ms = 0
    return latest, shutdown


def wait_for_command(
    sub: zmq.Socket,
    *,
    timeout_s: float = 2.0,
) -> CmdMsg | None:
    """Block until the first command arrives or ``timeout_s`` elapses."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        remaining_ms = int(max(0.0, deadline - time.perf_counter()) * 1000)
        if remaining_ms == 0:
            break
        latest, shutdown = drain_latest_command(
            sub,
            poll_ms=min(RCV_TIMEOUT_MS, remaining_ms),
        )
        if shutdown:
            return None
        if latest is not None:
            return latest
    return None


def bind_state_publisher(ctx: zmq.Context) -> zmq.Socket:
    pub = ctx.socket(zmq.PUB)
    _configure_pub(pub)
    pub.bind(STATE_ADDR)
    return pub


def connect_state_subscriber(ctx: zmq.Context) -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    _configure_sub(sub)
    sub.setsockopt(zmq.SUBSCRIBE, STATE_TOPIC)
    sub.setsockopt(zmq.SUBSCRIBE, SHUTDOWN_TOPIC)
    sub.connect(STATE_ADDR)
    return sub


def bind_command_publisher(ctx: zmq.Context) -> zmq.Socket:
    pub = ctx.socket(zmq.PUB)
    _configure_pub(pub)
    pub.bind(CMD_ADDR)
    return pub


def connect_command_subscriber(ctx: zmq.Context) -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    _configure_sub(sub)
    sub.setsockopt(zmq.SUBSCRIBE, CMD_TOPIC)
    sub.setsockopt(zmq.SUBSCRIBE, SHUTDOWN_TOPIC)
    sub.connect(CMD_ADDR)
    return sub


def close_sockets(*sockets: zmq.Socket) -> None:
    for sock in sockets:
        try:
            sock.close(linger=LINGER_MS)
        except zmq.ZMQError:
            pass


def term_context(ctx: zmq.Context) -> None:
    ctx.term()
