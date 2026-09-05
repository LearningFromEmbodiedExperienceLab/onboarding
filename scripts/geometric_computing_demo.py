"""Geometric computing demo on the Stanford Bunny.

Point-cloud normals, ray–mesh hits, voxel occupancy, and marching cubes from an
approximate SDF (Euclidean distance transform of the occupancy grid).

Run::

    uv sync --extra geometry
    uv run python scripts/geometric_computing_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BUNNY_PATH = ROOT / "assets" / "stanford_bunny" / "bunny.obj"


def load_bunny(*, target_extent: float = 1.0) -> trimesh.Trimesh:
    """Load the vendored Stanford Bunny, center it, and scale to unit-ish size."""
    mesh = trimesh.load(BUNNY_PATH, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a single mesh at {BUNNY_PATH}")
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    extent = float(np.max(mesh.extents))
    if extent <= 0:
        raise ValueError("degenerate bunny mesh")
    mesh.apply_scale(target_extent / extent)
    return mesh


def estimate_normals_pca(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Unit normals from local PCA (smallest eigenvector)."""
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=min(k, len(points)))
    normals = np.zeros_like(points)
    for i, nb in enumerate(idxs):
        X = points[nb]
        C = np.cov(X - X.mean(axis=0), rowvar=False)
        _, V = np.linalg.eigh(C)
        n = V[:, 0]
        # Orient toward +z viewpoint at (0, 0, 3) for a stable demo sign.
        view = np.array([0.0, 0.0, 3.0]) - points[i]
        if np.dot(n, view) < 0:
            n = -n
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
    return normals


def mesh_vertex_normals_at_samples(
    mesh: trimesh.Trimesh, face_indices: np.ndarray
) -> np.ndarray:
    """Per-sample normals from the supporting face (area-weighted face normals)."""
    return np.asarray(mesh.face_normals[face_indices], dtype=float)


def ray_triangle(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    eps: float = 1e-8,
) -> float | None:
    """Möller–Trumbore. Returns t >= 0 or None."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)
    if abs(det) < eps:
        return None
    inv_det = 1.0 / det
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = np.dot(edge2, qvec) * inv_det
    if t < 0.0:
        return None
    return float(t)


def ray_mesh_first_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[float, int] | None:
    """First hit (t, face_index); O(n_faces) scan for teaching."""
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    best: tuple[float, int] | None = None
    for fi, (i0, i1, i2) in enumerate(faces):
        t = ray_triangle(origin, direction, vertices[i0], vertices[i1], vertices[i2])
        if t is None:
            continue
        if best is None or t < best[0]:
            best = (t, fi)
    return best


def voxelize_points(
    points: np.ndarray,
    *,
    pitch: float = 0.03,
    margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Binary occupancy from points. Returns (occ, origin, pitch)."""
    origin = points.min(axis=0) - margin
    extent = points.max(axis=0) + margin - origin
    shape = np.maximum(np.ceil(extent / pitch).astype(int), 1)
    ijk = np.floor((points - origin) / pitch).astype(int)
    ijk = np.clip(ijk, 0, shape - 1)
    occ = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return occ, origin, pitch


def world_to_voxel(
    points: np.ndarray, origin: np.ndarray, pitch: float
) -> np.ndarray:
    return np.floor((points - origin) / pitch).astype(int)


def occupancy_to_sdf(occ: np.ndarray, pitch: float) -> np.ndarray:
    """Approximate SDF via EDT: negative inside occupied cells."""
    outside = distance_transform_edt(~occ)
    inside = distance_transform_edt(occ)
    return (outside - inside) * pitch


def main() -> None:
    mesh = load_bunny()
    print(f"bunny  verts={len(mesh.vertices)}  faces={len(mesh.faces)}")
    print(f"       watertight={mesh.is_watertight}  extents={mesh.extents}")

    print("\n=== Point cloud normals (PCA vs face normals) ===")
    cloud, face_id = trimesh.sample.sample_surface(mesh, 1500)
    normals = estimate_normals_pca(cloud, k=24)
    face_n = mesh_vertex_normals_at_samples(mesh, face_id)
    # Align PCA sign to face normal for the comparison metric.
    flip = np.sign(np.sum(normals * face_n, axis=1))
    flip[flip == 0] = 1.0
    align = np.sum((normals * flip[:, None]) * face_n, axis=1)
    print(
        f"samples={len(cloud)}  n·n_face mean={align.mean():.4f}  "
        f"min={align.min():.4f}"
    )

    print("\n=== Ray × bunny mesh (Möller–Trumbore) ===")
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([0.0, 0.0, -1.0])
    hit = ray_mesh_first_hit(origin, direction, mesh.vertices, mesh.faces)
    assert hit is not None, "expected a hit toward the centered bunny"
    t, fi = hit
    point = origin + t * direction
    print(f"hit t={t:.4f}  point={point}  face={fi}")

    print("\n=== Voxel occupancy (bunny samples → grid) ===")
    occ, grid_origin, pitch = voxelize_points(cloud, pitch=0.03, margin=0.04)
    filled = int(occ.sum())
    i_on = world_to_voxel(cloud[0][None, :], grid_origin, pitch)[0]
    i_out = world_to_voxel(np.array([[2.0, 2.0, 2.0]]), grid_origin, pitch)[0]
    shape = np.array(occ.shape)

    def in_bounds(ijk: np.ndarray) -> bool:
        return bool(np.all((ijk >= 0) & (ijk < shape)))

    hit_on = bool(in_bounds(i_on) and occ[tuple(i_on)])
    hit_out = bool(in_bounds(i_out) and occ[tuple(i_out)])
    print(
        f"grid={occ.shape}  pitch={pitch:.3f}  filled={filled}  "
        f"surface_cell_occupied={hit_on}  outside_occupied={hit_out}"
    )
    assert hit_on and not hit_out

    print("\n=== Marching cubes (occupancy EDT ≈ SDF → mesh) ===")
    # Slightly denser voxelization of the *mesh* for a cleaner isosurface.
    vox = mesh.voxelized(pitch=0.02)
    sdf = occupancy_to_sdf(vox.matrix, pitch=0.02)
    verts, faces, _normals, _ = marching_cubes(sdf, level=0.0, spacing=(0.02,) * 3)
    # skimage verts are in grid index space scaled by spacing; map to world via voxel translation.
    mc = trimesh.Trimesh(
        vertices=verts + np.asarray(vox.translation, dtype=float),
        faces=faces,
        process=False,
    )
    print(
        f"verts={len(mc.vertices)} faces={len(mc.faces)}  "
        f"watertight={mc.is_watertight}  volume={mc.volume:.4f}"
    )
    # Vertex-cloud proximity (avoids rtree); rough surface fidelity check.
    tree = cKDTree(np.asarray(mesh.vertices))
    d_mc, _ = tree.query(mc.vertices)
    print(f"mean dist(MC verts → bunny verts)={d_mc.mean():.4f}")
    print("\ndone")


if __name__ == "__main__":
    main()
