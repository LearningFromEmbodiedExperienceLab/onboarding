"""Render differential IK tracking videos for the docs (headless MuJoCo).

Requires: uv sync --extra mujoco, scripts/fetch_menagerie_assets.sh, OSMesa, ffmpeg.

Environment:
  MUJOCO_GL=osmesa  PYOPENGL_PLATFORM=osmesa  (headless Linux)

Outputs (committed under docs/imgs/ik-tracking/):
  piper_ik_reach.mp4   — proximal target reach (seed 0)
  piper_ik_circle.mp4  — horizontal circle tracking
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Must be set before importing mujoco on headless machines.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ik_tracking_common import (
    ARM_DOF,
    TrackingTest,
    run_differential_ik,
)
from sim_common import EE_LINK, require_piper_scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "imgs" / "ik-tracking"

WIDTH = 640
HEIGHT = 480
FPS = 30
CIRCLE_FRAME_STRIDE = 4
REACH_FRAME_REPEAT = 4
REACH_HOLD_FRAMES = 20
CIRCLE_HOLD_FRAMES = 15


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


class MujocoRecorder:
    """MuJoCo kinematics + headless render for doc videos."""

    def __init__(self, model: mj.MjModel, data: mj.MjData, mocap_id: int) -> None:
        self.model = model
        self.data = data
        self._mocap_id = mocap_id
        self._ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_LINK)
        self._renderer = mj.Renderer(model, height=HEIGHT, width=WIDTH)
        self._camera = _camera(model)
        self._q_min = np.array([model.jnt_range[i, 0] for i in range(ARM_DOF)])
        self._q_max = np.array([model.jnt_range[i, 1] for i in range(ARM_DOF)])
        self.backend = "mujoco"

    @property
    def q_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_min, self._q_max

    def reset(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

    def get_arm_qpos(self) -> np.ndarray:
        return np.array(self.data.qpos[:ARM_DOF], dtype=float)

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[:ARM_DOF] = q

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self._ee_id], dtype=float)

    def forward(self) -> None:
        mj.mj_forward(self.model, self.data)

    def position_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacBody(self.model, self.data, jacp, jacr, self._ee_id)
        return jacp[:, :ARM_DOF].copy()

    def set_target_marker(self, target: np.ndarray) -> None:
        self.data.mocap_pos[self._mocap_id] = target

    def render_frame(self, *, title: str, err_mm: float) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._camera)
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8)
        return _annotate_frame(rgb, title=title, err_mm=err_mm)


def _annotate_frame(rgb: np.ndarray, *, title: str, err_mm: float) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, WIDTH, 28], fill=(0, 0, 0, 180))
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    draw.text(
        (8, HEIGHT - 22),
        f"position error: {err_mm:.1f} mm  |  red sphere = target",
        fill=(255, 255, 255),
        font=font,
    )
    return np.asarray(img, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"no frames to write for {path}")

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


def _record_test(
    robot: MujocoRecorder,
    *,
    test: TrackingTest,
    seed: int,
    title: str,
    frame_stride: int,
    tail_hold: int,
) -> list[np.ndarray]:
    q_min, q_max = robot.q_limits
    rng = np.random.default_rng(seed)
    robot.reset()
    robot.set_arm_qpos(np.zeros(ARM_DOF))
    robot.forward()
    circle_center = robot.get_ee_pos().copy()
    frames: list[np.ndarray] = []

    def on_step(step: int, target: np.ndarray, _ee: np.ndarray, err: float) -> None:
        robot.set_target_marker(target)
        if test is TrackingTest.CIRCLE and step % frame_stride != 0:
            return
        frame = robot.render_frame(title=title, err_mm=err * 1e3)
        repeat = REACH_FRAME_REPEAT if test is TrackingTest.REACH else 1
        for _ in range(repeat):
            frames.append(frame.copy())

    result = run_differential_ik(
        robot,
        test=test,
        q_min=q_min,
        q_max=q_max,
        rng=rng,
        circle_center=circle_center,
        on_step=on_step,
    )
    if not result.success:
        raise RuntimeError(f"{test.value} tracking failed: {result}")

    for _ in range(tail_hold):
        frames.append(frames[-1].copy())
    return frames


def render_videos(seed: int = 0) -> list[Path]:
    model, data, mocap_id = _load_model_with_target_marker()
    robot = MujocoRecorder(model, data, mocap_id)
    outputs: list[Path] = []

    specs = [
        (
            TrackingTest.REACH,
            OUT / "piper_ik_reach.mp4",
            "Diff IK — reach (MuJoCo, seed 0)",
            1,
            REACH_HOLD_FRAMES,
        ),
        (
            TrackingTest.CIRCLE,
            OUT / "piper_ik_circle.mp4",
            "Diff IK — circle track (MuJoCo, seed 0)",
            CIRCLE_FRAME_STRIDE,
            CIRCLE_HOLD_FRAMES,
        ),
    ]

    for test, path, title, stride, hold in specs:
        frames = _record_test(
            robot,
            test=test,
            seed=seed,
            title=title,
            frame_stride=stride,
            tail_hold=hold,
        )
        _write_mp4(frames, path)
        print(f"wrote {path}  ({len(frames)} frames @ {FPS} fps)")
        outputs.append(path)

    return outputs


def main() -> None:
    paths = render_videos(seed=0)
    print("done:", ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()
