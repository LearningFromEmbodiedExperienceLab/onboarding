"""Render visual vs collision geometry figures for the docs (headless MuJoCo).

Requires: uv sync --extra mujoco, scripts/fetch_menagerie_assets.sh, OSMesa (CI/local).

Environment:
  MUJOCO_GL=osmesa  PYOPENGL_PLATFORM=osmesa  (headless Linux)

Outputs (committed under docs/imgs/menagerie/):
  {robot}_visual.png, {robot}_collision.png, {robot}_visual_hull.png,
  {robot}_compare.png (visual | visual-mesh convex hull | collision)
"""

from __future__ import annotations

import os
from pathlib import Path

# Must be set before importing mujoco on headless machines.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco as mj
import numpy as np
import trimesh
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
HULL_GROUPS = {0, 2}  # floor + synthetic hull geoms (group 2 in hull model)

HULL_RGBA = (1.0, 0.55, 0.1, 1.0)


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


def _mesh_world_vertices(model: mj.MjModel, data: mj.MjData, geom_id: int) -> np.ndarray:
    mesh_id = model.geom_dataid[geom_id]
    adr = model.mesh_vertadr[mesh_id]
    num = model.mesh_vertnum[mesh_id]
    v_local = model.mesh_vert[adr : adr + num].reshape(-1, 3)
    gmat = data.geom_xmat[geom_id].reshape(3, 3)
    gpos = data.geom_xpos[geom_id]
    return v_local @ gmat.T + gpos


def build_visual_hull_model(model: mj.MjModel, data: mj.MjData) -> mj.MjModel:
    """World-frame convex hulls of each visual mesh geom (not used in Menagerie contact)."""
    spec = mj.MjSpec()
    spec.modelname = "visual_hulls"
    # Minimal floor and lighting to match Menagerie scenes.
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20
    tex = spec.add_texture(
        type=mj.mjtTexture.mjTEXTURE_2D,
        name="groundplane",
        builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        width=300,
        height=300,
    )
    mat = spec.add_material(name="groundplane", textures=[tex.name], texuniform=True)
    world = spec.worldbody
    world.add_light(pos=(0, 0, 1.5), dir=(0, 0, -1), type=mj.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_geom(
        name="floor",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=(0, 0, 0.05),
        material=mat.name,
    )

    hull_idx = 0
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] != 2:
            continue
        if model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_MESH:
            continue
        v_world = _mesh_world_vertices(model, data, geom_id)
        if len(v_world) < 4:
            continue
        hull = trimesh.convex.convex_hull(
            trimesh.Trimesh(vertices=v_world, process=False)
        )
        mesh_name = f"visual_hull_{hull_idx}"
        hull_idx += 1
        spec.add_mesh(
            name=mesh_name,
            uservert=hull.vertices.reshape(-1).tolist(),
            userface=hull.faces.reshape(-1).tolist(),
        )
        world.add_geom(
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            group=2,
            contype=0,
            conaffinity=0,
            rgba=list(HULL_RGBA),
        )

    if hull_idx == 0:
        raise RuntimeError("No visual mesh geoms found for hull figure.")

    return spec.compile()


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    pad = 10
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(255, 255, 255, 230))
    draw.text((pad, pad), text, fill=(20, 20, 20), font=font)
    return out


def _hstack(images: list[Image.Image]) -> Image.Image:
    w = sum(im.width for im in images)
    h = max(im.height for im in images)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
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

    hull_model = build_visual_hull_model(model, data)
    hull_data = mj.MjData(hull_model)
    mj.mj_forward(hull_model, hull_data)
    visual_hull = render_rgb(hull_model, hull_data, show_groups=HULL_GROUPS)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "visual": OUT / f"{name}_visual.png",
        "collision": OUT / f"{name}_collision.png",
        "visual_hull": OUT / f"{name}_visual_hull.png",
        "compare": OUT / f"{name}_compare.png",
    }

    Image.fromarray(visual).save(paths["visual"])
    Image.fromarray(collision).save(paths["collision"])
    Image.fromarray(visual_hull).save(paths["visual_hull"])

    triple = _hstack(
        [
            _label(Image.fromarray(visual), "Visual meshes (group 2)"),
            _label(
                Image.fromarray(visual_hull),
                "Convex hull of each visual mesh",
            ),
            _label(Image.fromarray(collision), "Author collision (group 3)"),
        ]
    )
    triple.save(paths["compare"])

    for path in paths.values():
        print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    for robot_name, path in ROBOTS.items():
        save_robot_figures(robot_name, path)


if __name__ == "__main__":
    main()
