"""Render dynamic IK instability for docs (headless MuJoCo).

Shows position-actuator differential IK blowing up when ``timestep`` is too
large for an explicit integrator (Euler @ 20 ms). Stable reference uses
implicitfast @ 20 ms with the same control rate.

Requires: uv sync --extra mujoco, fetch_menagerie_assets.sh, OSMesa, ffmpeg.

Outputs (committed under docs/imgs/ik-tracking/):
  piper_ik_instability.mp4
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ik_tracking_common import (
    ARM_DOF,
    ARM_JOINT_NAMES,
    MAX_STEPS_REACH,
    NOMINAL_Q,
    clip_arm_qpos,
    random_proximal_target,
)
from robotics.ik import make_ik
from sim_common import EE_LINK, require_piper_scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "imgs" / "ik-tracking" / "piper_ik_instability.mp4"

WIDTH = 640
HEIGHT = 480
FPS = 30
FRAME_STRIDE = 1
IK_GAIN = 0.35
MAX_FRAMES = 90


def _load_model_with_target_marker() -> tuple[mj.MjModel, mj.MjData, int]:
    scene = require_piper_scene()
    spec = mj.MjSpec.from_file(str(scene))
    body = spec.worldbody.add_body(name="ik_target", mocap=True)
    body.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.012, 0.0, 0.0],
        rgba=[1.0, 0.25, 0.2, 0.92],
    )
    model = spec.compile()
    data = mj.MjData(model)
    mocap_id = model.body("ik_target").mocapid[0]
    return model, data, mocap_id


def _camera(model: mj.MjModel) -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = float(model.stat.extent * 2.35)
    cam.azimuth = 128.0
    cam.elevation = -22.0
    cam.lookat[:] = model.stat.center + np.array([0.0, 0.0, 0.04])
    return cam


def _sync_all_actuator_ctrl(model: mj.MjModel, data: mj.MjData) -> None:
    for i in range(model.nu):
        if model.actuator_trntype[i] != 0:
            continue
        joint_id = model.actuator_trnid[i, 0]
        qadr = model.jnt_qposadr[joint_id]
        data.ctrl[i] = data.qpos[qadr]


def _annotate_frame(
    rgb: np.ndarray,
    *,
    title: str,
    err_mm: float,
    max_qvel: float,
    subtitle: str,
) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, WIDTH, 40], fill=(0, 0, 0, 200))
    draw.text((8, 4), title, fill=(255, 90, 90), font=font)
    draw.text((8, 20), subtitle, fill=(220, 220, 220), font=font)
    draw.text(
        (8, HEIGHT - 22),
        f"pos err {err_mm:.0f} mm  |  max |qvel| {max_qvel:.1f} rad/s  |  red = target",
        fill=(255, 255, 255),
        font=font,
    )
    return np.asarray(img, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def _record_unstable_reach(
    *,
    integrator: int,
    dt: float,
    seed: int,
) -> list[np.ndarray]:
    model, data, mocap_id = _load_model_with_target_marker()
    model.opt.timestep = dt
    model.opt.integrator = integrator
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
    act_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINT_NAMES
    ]
    q_min = np.array([model.jnt_range[i, 0] for i in range(ARM_DOF)])
    q_max = np.array([model.jnt_range[i, 1] for i in range(ARM_DOF)])
    ik = make_ik("dls", damping=1e-2)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:ARM_DOF] = NOMINAL_Q
    _sync_all_actuator_ctrl(model, data)
    mj.mj_forward(model, data)

    rng = np.random.default_rng(seed)
    target = random_proximal_target(data.xpos[ee_id].copy(), rng)
    data.mocap_pos[mocap_id] = target

    renderer = mj.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = _camera(model)
    frames: list[np.ndarray] = []
    max_qvel_seen = 0.0
    integrator_name = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}[
        integrator
    ]
    title = f"UNSTABLE — dynamic IK, {integrator_name}, dt={dt * 1e3:.0f} ms"
    subtitle = "position actuators + mj_step (one IK tick per physics step)"

    for step in range(MAX_STEPS_REACH):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, ee_id)
        ee_pos = data.xpos[ee_id].copy()
        dx = target - ee_pos
        err = float(np.linalg.norm(dx))
        dq = ik(jacp[:, :ARM_DOF], dx)
        q_des = clip_arm_qpos(data.qpos[:ARM_DOF] + IK_GAIN * dq, q_min, q_max)
        for act_id, q_i in zip(act_ids, q_des, strict=True):
            data.ctrl[act_id] = q_i
        # Keep gripper ctrl aligned; do not overwrite arm targets we just wrote.
        grip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "gripper")
        j7_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "joint7")
        data.ctrl[grip_id] = data.qpos[model.jnt_qposadr[j7_id]]
        mj.mj_step(model, data)

        qvel = float(np.max(np.abs(data.qvel)))
        max_qvel_seen = max(max_qvel_seen, qvel)

        if step % FRAME_STRIDE == 0:
            renderer.update_scene(data, camera=camera)
            rgb = np.asarray(renderer.render(), dtype=np.uint8)
            frames.append(
                _annotate_frame(
                    rgb,
                    title=title,
                    err_mm=err * 1e3,
                    max_qvel=qvel,
                    subtitle=subtitle,
                )
            )
        if len(frames) >= MAX_FRAMES:
            break

    # Hold last frame so viewers see the blown-up state.
    for _ in range(15):
        frames.append(frames[-1].copy())

    print(
        f"recorded {len(frames)} frames, final err={err * 1e3:.1f} mm, "
        f"peak |qvel|={max_qvel_seen:.1f} rad/s"
    )
    return frames


def main() -> None:
    # Benchmark: Euler @ 20 ms fails reach (|qvel| spikes, err ~ 33 mm+).
    frames = _record_unstable_reach(
        integrator=0,  # euler
        dt=0.02,
        seed=0,
    )
    _write_mp4(frames, OUT)
    print(f"wrote {OUT}  ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
