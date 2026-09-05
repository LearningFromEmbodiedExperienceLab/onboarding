"""Render Stanford Bunny figures for docs/geometric-computing.qmd.

Outputs under ``docs/imgs/geometric-computing/``:

- ``bunny_mesh.png`` — source triangle mesh
- ``bunny_point_cloud_normals.png`` — surface samples + PCA normals
- ``bunny_ray_query.png`` — Möller–Trumbore ray hit
- ``bunny_convex_hull.png`` — single Qhull
- ``bunny_convex_decomposition.png`` — AABB + Qhull parts
- ``bunny_voxels.png`` — occupancy cell centers
- ``bunny_marching_cubes.png`` — EDT → marching-cubes mesh

Run::

    uv sync --extra geometry
    uv run python scripts/render_geometric_computing_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometric_computing_demo import (  # noqa: E402
    estimate_normals_pca,
    load_bunny,
    occupancy_to_sdf,
    qhull_aabb_decompose,
    qhull_mesh,
    ray_mesh_first_hit,
    voxelize_points,
)

OUT = ROOT / "docs" / "imgs" / "geometric-computing"
# Slightly elevated three-quarter view; bunny sits upright (+Z up).
ELEV, AZIM = 22, -55


def _style_ax(ax, title: str, *, lim: float = 0.65) -> None:
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1, 1, 0.85))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def _add_mesh(
    ax,
    mesh,
    *,
    face_color="#9bb8d3",
    edge_color="#3a4a5c",
    alpha: float = 0.55,
    linewidth: float = 0.15,
) -> None:
    tris = mesh.vertices[mesh.faces]
    poly = Poly3DCollection(
        tris,
        alpha=alpha,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidths=linewidth,
    )
    ax.add_collection3d(poly)


def _savefig(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {path}")
    return path


def _sample_surface(mesh, n: int):
    return trimesh.sample.sample_surface(mesh, n)


def fig_mesh(mesh) -> None:
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.75)
    _style_ax(ax, "Stanford Bunny — triangle mesh")
    _savefig(fig, "bunny_mesh.png")


def fig_point_cloud_normals(mesh) -> None:
    cloud, _ = _sample_surface(mesh, 900)
    normals = estimate_normals_pca(cloud, k=20)
    # Orient toward a viewpoint so arrows read as outward.
    viewpoint = np.array([0.0, -1.5, 0.8])
    flip = np.sign(np.sum(normals * (viewpoint - cloud), axis=1))
    flip[flip == 0] = 1.0
    normals = normals * flip[:, None]

    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.12, linewidth=0.05)
    ax.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2], s=4, c="#1f4e79", depthshade=True
    )
    step = 6
    pts = cloud[::step]
    nrm = normals[::step] * 0.06
    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        nrm[:, 0],
        nrm[:, 1],
        nrm[:, 2],
        color="#c0392b",
        linewidth=0.6,
        arrow_length_ratio=0.35,
    )
    _style_ax(ax, "Point cloud + PCA normals")
    _savefig(fig, "bunny_point_cloud_normals.png")


def fig_ray_query(mesh) -> None:
    # Same ray as the demo; expand z so origin and hit both fit.
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([0.0, 0.0, -1.0])
    hit = ray_mesh_first_hit(origin, direction, mesh.vertices, mesh.faces)
    assert hit is not None
    t, _ = hit
    point = origin + t * direction
    end = origin + direction * (t + 0.45)

    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.45)
    ax.plot(
        [origin[0], end[0]],
        [origin[1], end[1]],
        [origin[2], end[2]],
        color="#c0392b",
        linewidth=1.8,
        zorder=5,
    )
    ax.scatter(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        c="#27ae60",
        s=55,
        depthshade=False,
        zorder=6,
        label="origin",
    )
    ax.scatter(
        [point[0]],
        [point[1]],
        [point[2]],
        c="#c0392b",
        s=70,
        depthshade=False,
        zorder=6,
        label="hit",
    )
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    _style_ax(ax, "Ray query (Möller–Trumbore)", lim=0.65)
    ax.set_zlim(-0.65, 2.15)
    ax.set_box_aspect((1, 1, 1.4))
    _savefig(fig, "bunny_ray_query.png")


def fig_convex_hull(mesh) -> None:
    hull = qhull_mesh(np.asarray(mesh.vertices))
    assert hull is not None
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.25, face_color="#7f8c8d", edge_color="#566573")
    _add_mesh(
        ax,
        hull,
        alpha=0.25,
        face_color="#e67e22",
        edge_color="#d35400",
        linewidth=0.35,
    )
    _style_ax(ax, "Single convex hull (Qhull)")
    _savefig(fig, "bunny_convex_hull.png")


def fig_convex_decomposition(mesh) -> None:
    parts = qhull_aabb_decompose(
        np.asarray(mesh.vertices), max_depth=3, min_points=40
    )
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.08, linewidth=0.0)
    for i, part in enumerate(parts):
        color = mcolors.to_hex(cmap(i % 10))
        _add_mesh(
            ax,
            part,
            alpha=0.35,
            face_color=color,
            edge_color="#2c3e50",
            linewidth=0.25,
        )
    _style_ax(ax, f"AABB + Qhull decomposition ({len(parts)} parts)")
    _savefig(fig, "bunny_convex_decomposition.png")


def fig_voxels(mesh) -> None:
    cloud, _ = _sample_surface(mesh, 2000)
    occ, origin, pitch = voxelize_points(cloud, pitch=0.035, margin=0.03)
    ijk = np.argwhere(occ)
    centers = origin + (ijk + 0.5) * pitch
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.1, linewidth=0.0)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        s=8,
        c="#2980b9",
        alpha=0.7,
        depthshade=True,
    )
    _style_ax(ax, "Voxel occupancy (cell centers)")
    _savefig(fig, "bunny_voxels.png")


def fig_marching_cubes(mesh) -> None:
    vox = mesh.voxelized(pitch=0.02)
    sdf = occupancy_to_sdf(vox.matrix, pitch=0.02)
    verts, faces, _normals, _ = marching_cubes(sdf, level=0.0, spacing=(0.02,) * 3)
    mc = trimesh.Trimesh(
        vertices=verts + np.asarray(vox.translation, dtype=float),
        faces=faces,
        process=False,
    )
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, mesh, alpha=0.12, linewidth=0.0, face_color="#95a5a6")
    _add_mesh(
        ax,
        mc,
        alpha=0.65,
        face_color="#1abc9c",
        edge_color="#0e6655",
        linewidth=0.12,
    )
    _style_ax(ax, "Marching cubes (EDT ≈ SDF)")
    _savefig(fig, "bunny_marching_cubes.png")


def main() -> None:
    mesh = load_bunny()
    fig_mesh(mesh)
    fig_point_cloud_normals(mesh)
    fig_ray_query(mesh)
    fig_convex_hull(mesh)
    fig_convex_decomposition(mesh)
    fig_voxels(mesh)
    fig_marching_cubes(mesh)
    print("done")


if __name__ == "__main__":
    main()
