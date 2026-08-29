"""Shared-memory IPC layout for the async sim / controller demo.

One fixed struct in a ``multiprocessing.shared_memory`` segment. A seqlock
(sequence lock) on ``seq`` lets the reader detect torn reads without a mutex
on the hot path.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from multiprocessing import shared_memory

SHM_NAME = "robotics_async_ipc_demo_v1"

# Binary layout (little-endian): 3× int64, 4× float64, 1× int64.
_LAYOUT = struct.Struct("<qqqddddq")
FIELD_NAMES = (
    "seq",
    "sim_steps",
    "ctrl_ticks",
    "sim_time",
    "q",
    "qdot",
    "cmd_q",
    "shutdown",
)
SHM_SIZE = _LAYOUT.size


@dataclass(frozen=True)
class IpcSnapshot:
    seq: int
    sim_steps: int
    ctrl_ticks: int
    sim_time: float
    q: float
    qdot: float
    cmd_q: float
    shutdown: bool


class AsyncIpcBus:
    """Seqlock-protected observation + command buffer in shared RAM."""

    def __init__(self, shm: shared_memory.SharedMemory) -> None:
        self._shm = shm

    @classmethod
    def create(cls) -> AsyncIpcBus:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        except FileExistsError:
            shared_memory.SharedMemory(name=SHM_NAME).unlink()
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        bus = cls(shm)
        bus._publish(
            seq=0,
            sim_steps=0,
            ctrl_ticks=0,
            sim_time=0.0,
            q=0.0,
            qdot=0.0,
            cmd_q=0.0,
            shutdown=False,
        )
        return bus

    @classmethod
    def attach(cls) -> AsyncIpcBus:
        return cls(shared_memory.SharedMemory(name=SHM_NAME))

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    def _publish(
        self,
        *,
        seq: int,
        sim_steps: int,
        ctrl_ticks: int,
        sim_time: float,
        q: float,
        qdot: float,
        cmd_q: float,
        shutdown: bool,
    ) -> None:
        _LAYOUT.pack_into(
            self._shm.buf,
            0,
            seq,
            sim_steps,
            ctrl_ticks,
            sim_time,
            q,
            qdot,
            cmd_q,
            int(shutdown),
        )

    def read_snapshot(self) -> IpcSnapshot:
        """Read sim state + latest command; retry if a writer was mid-update."""
        while True:
            snap_a = self._unpack_from_shm()
            if snap_a.seq & 1:
                continue
            snap_b = self._unpack_from_shm()
            if snap_a.seq == snap_b.seq:
                return snap_b

    def _unpack_from_shm(self) -> IpcSnapshot:
        values = _LAYOUT.unpack_from(self._shm.buf)
        data = dict(zip(FIELD_NAMES, values, strict=True))
        return IpcSnapshot(
            seq=int(data["seq"]),
            sim_steps=int(data["sim_steps"]),
            ctrl_ticks=int(data["ctrl_ticks"]),
            sim_time=float(data["sim_time"]),
            q=float(data["q"]),
            qdot=float(data["qdot"]),
            cmd_q=float(data["cmd_q"]),
            shutdown=bool(data["shutdown"]),
        )

    def write_state(
        self,
        *,
        sim_steps: int,
        ctrl_ticks: int,
        sim_time: float,
        q: float,
        qdot: float,
        cmd_q: float,
        shutdown: bool,
    ) -> None:
        """Sim process publishes integrated state."""
        current = self.read_snapshot()
        seq = current.seq + 1
        self._publish(
            seq=seq,
            sim_steps=sim_steps,
            ctrl_ticks=ctrl_ticks,
            sim_time=sim_time,
            q=q,
            qdot=qdot,
            cmd_q=cmd_q,
            shutdown=shutdown,
        )
        self._publish(
            seq=seq + 1,
            sim_steps=sim_steps,
            ctrl_ticks=ctrl_ticks,
            sim_time=sim_time,
            q=q,
            qdot=qdot,
            cmd_q=cmd_q,
            shutdown=shutdown,
        )

    def write_command(
        self,
        *,
        cmd_q: float,
        ctrl_ticks: int,
        shutdown: bool = False,
    ) -> None:
        """Controller publishes a new joint target (keeps sim fields intact)."""
        current = self.read_snapshot()
        seq = current.seq + 1
        self._publish(
            seq=seq,
            sim_steps=current.sim_steps,
            ctrl_ticks=ctrl_ticks,
            sim_time=current.sim_time,
            q=current.q,
            qdot=current.qdot,
            cmd_q=cmd_q,
            shutdown=shutdown,
        )
        self._publish(
            seq=seq + 1,
            sim_steps=current.sim_steps,
            ctrl_ticks=ctrl_ticks,
            sim_time=current.sim_time,
            q=current.q,
            qdot=current.qdot,
            cmd_q=cmd_q,
            shutdown=shutdown,
        )

    def request_shutdown(self) -> None:
        snap = self.read_snapshot()
        self.write_command(cmd_q=snap.cmd_q, ctrl_ticks=snap.ctrl_ticks, shutdown=True)


def rate_sleep(next_tick: float, dt: float) -> float:
    """Sleep until ``next_tick + dt``; return the following tick time."""
    next_tick += dt
    delay = next_tick - _now()
    if delay > 0:
        import time

        time.sleep(delay)
    return next_tick


def _now() -> float:
    import time

    return time.perf_counter()


# Default demo rates (Hz).
SIM_HZ = 500.0
CTRL_HZ = 50.0
SIM_DT = 1.0 / SIM_HZ
CTRL_DT = 1.0 / CTRL_HZ

# Parent process: max wait for child exit after run duration.
JOIN_GRACE_S = 3.0

# Simple 1-DoF plant: position target tracking with PD-style acceleration.
PLANT_MASS = 1.0
PLANT_DAMPING = 4.0
PLANT_STIFFNESS = 40.0


def integrate_plant(q: float, qdot: float, cmd_q: float) -> tuple[float, float]:
    """One PD-style integration step toward ``cmd_q``."""
    accel = (PLANT_STIFFNESS * (cmd_q - q) - PLANT_DAMPING * qdot) / PLANT_MASS
    qdot_next = qdot + SIM_DT * accel
    q_next = q + SIM_DT * qdot_next
    return q_next, qdot_next


def command_for_tick(ctrl_ticks: int) -> float:
    """Sinusoidal setpoint used by both IPC demos."""
    t = ctrl_ticks * CTRL_DT
    return 0.35 * math.sin(2.0 * math.pi * 0.5 * t)


@dataclass(frozen=True)
class RunSummary:
    sim_steps: int
    ctrl_ticks: int
    sim_time: float
    q: float
    qdot: float
    cmd_q: float
