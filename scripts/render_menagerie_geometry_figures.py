"""Render visual vs collision geometry figures for the docs (headless MuJoCo).

Requires: uv sync --extra mujoco, scripts/fetch_menagerie_assets.sh, OSMesa (CI/local).

Environment:
  MUJOCO_GL=osmesa  PYOPENGL_PLATFORM=osmesa  (headless Linux)

Outputs (committed under docs/imgs/menagerie/):
  {robot}_visual.png, {robot}_collision.png, {robot}_compare.png
"""

from __future__ import annotations

import os
from pathlib import Path

# Must be set before importing mujoco on headless machines.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"
OUT = ROOT / "docs" / "imgs" / "menagerie"

ROBOTS: dict[str, Path] = {
    "piper": MENAGERIE / "agilex_piper" / "scene.xml",
    "arx_l5": MENAGERIE / "arx_l5" / "scene.xml",
}

# Menagerie convention: group 2 = visual mesh, group 3 = collision geoms.
VISUAL_GROUPS = {0, 2}  # floor + visual
COLLISION_GROUPS = {0, 3}  # floor + collision


def _camera(model: mj.MjModel, data: mj.MjData) -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = float(model.stat.extent * 2.2)
    cam.azimuth = 120.0
    cam.elevation = -18.0
    cam.lookat[:] = model.stat.center
    return cam


def _scene_option(show_groups: set[int]) -> mj.MjvOption:
    opt = mj.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in show_groups else 0
    return opt


def render_rgb(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    width: int = 640,
    height: int = 480,
    show_groups: set[int],
) -> np.ndarray:
    renderer = mj.Renderer(model, height=height, width=width)
    cam = _camera(model, data)
    opt = _scene_option(show_groups)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return np.asarray(renderer.render(), dtype=np.uint8)


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    pad = 10
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(255, 255, 255, 230))
    draw.text((pad, pad), text, fill=(20, 20, 20), font=font)
    return out


def save_robot_figures(name: str, scene_path: Path) -> None:
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Missing {scene_path}. Run: bash scripts/fetch_menagerie_assets.sh"
        )

    model = mj.MjModel.from_xml_path(str(scene_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    visual = render_rgb(model, data, show_groups=VISUAL_GROUPS)
    collision = render_rgb(model, data, show_groups=COLLISION_GROUPS)

    OUT.mkdir(parents=True, exist_ok=True)
    visual_path = OUT / f"{name}_visual.png"
    collision_path = OUT / f"{name}_collision.png"
    compare_path = OUT / f"{name}_compare.png"

    Image.fromarray(visual).save(visual_path)
    Image.fromarray(collision).save(collision_path)

    v = _label(Image.fromarray(visual), "Visual geoms (group 2)")
    c = _label(Image.fromarray(collision), "Collision geoms (group 3)")
    compare = Image.new("RGB", (v.width + c.width, max(v.height, c.height)), (255, 255, 255))
    compare.paste(v, (0, 0))
    compare.paste(c, (v.width, 0))
    compare.save(compare_path)

    print(f"Wrote {visual_path.relative_to(ROOT)}")
    print(f"Wrote {collision_path.relative_to(ROOT)}")
    print(f"Wrote {compare_path.relative_to(ROOT)}")


def main() -> None:
    for name, path in ROBOTS.items():
        save_robot_figures(name, path)


if __name__ == "__main__":
    main()
