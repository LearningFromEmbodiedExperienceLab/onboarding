"""Geometric computing demo: point-cloud normals, ray–mesh hits, marching cubes.

Run::

    uv sync --extra geometry
    uv run python scripts/geometric_computing_demo.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh


def sample_sphere_cloud(
    n: int = 800,
    radius: float = 1.0,
    noise: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return radius * vec + rng.normal(scale=noise, size=vec.shape)


def estimate_normals_pca(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Unit normals from local PCA (smallest eigenvector). Signs are arbitrary."""
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=min(k, len(points)))
    normals = np.zeros_like(points)
    for i, nb in enumerate(idxs):
        X = points[nb]
        C = np.cov(X - X.mean(axis=0), rowvar=False)
        _, V = np.linalg.eigh(C)
        n = V[:, 0]
        # Orient roughly outward from origin (demo only — use viewpoint in real code).
        if np.dot(n, points[i]) < 0:
            n = -n
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
    return normals


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
    """First hit (t, face_index) along a ray; O(n_faces) scan for teaching."""
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    best: tuple[float, int] | None = None
    for fi, (i0, i1, i2) in enumerate(faces):
        t = ray_triangle(origin, direction, vertices[i0], vertices[i1], vertices[i2])
        if t is None:
            continue
        if best is None or t < best[0]:
            best = (t, fi)
    return best


def sphere_sdf_grid(
    n: int = 48,
    radius: float = 1.0,
    pad: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sdf, spacing) with sdf shape (n, n, n); negative inside."""
    half = radius + pad
    xs = np.linspace(-half, half, n)
    spacing = np.array([xs[1] - xs[0]] * 3)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    sdf = np.sqrt(X * X + Y * Y + Z * Z) - radius
    return sdf.astype(np.float64), spacing


def main() -> None:
    print("=== Point cloud normals (PCA) ===")
    cloud = sample_sphere_cloud()
    normals = estimate_normals_pca(cloud, k=20)
    # True outward normals ≈ normalized positions for a sphere centered at 0.
    true = cloud / np.linalg.norm(cloud, axis=1, keepdims=True)
    align = np.abs(np.sum(normals * true, axis=1))  # ignore residual flips
    print(f"points={len(cloud)}  |n·n_true| mean={align.mean():.4f}  min={align.min():.4f}")

    print("\n=== Ray × triangle mesh (Möller–Trumbore) ===")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    origin = np.array([0.0, 0.0, 3.0])
    direction = np.array([0.0, 0.0, -1.0])
    hit = ray_mesh_first_hit(origin, direction, mesh.vertices, mesh.faces)
    assert hit is not None, "expected a hit toward the unit sphere"
    t, fi = hit
    point = origin + t * direction
    print(f"hit t={t:.4f}  point={point}  face={fi}  |point|={np.linalg.norm(point):.4f}")

    print("\n=== Marching cubes (sphere SDF → mesh) ===")
    sdf, spacing = sphere_sdf_grid(n=40, radius=1.0)
    verts, faces, _normals, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
    # skimage returns verts in (z, y, x) order relative to array axes with indexing ij
    # After spacing=, coordinates are in the same axis order as the volume (i, j, k).
    # Shift from grid corner (0,0,0) to world centered at origin:
    n = sdf.shape[0]
    half = (n - 1) * spacing[0] / 2.0
    verts_world = verts - half
    radii = np.linalg.norm(verts_world, axis=1)
    print(
        f"verts={len(verts_world)} faces={len(faces)}  "
        f"radius mean={radii.mean():.4f}  std={radii.std():.4f}"
    )
    mc_mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    print(f"watertight={mc_mesh.is_watertight}  volume={mc_mesh.volume:.4f} (4/3 π ≈ {4/3 * np.pi:.4f})")
    print("\ndone")


if __name__ == "__main__":
    main()
