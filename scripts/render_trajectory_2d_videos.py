"""Render 2D Bézier and B-spline animations for the trajectory chapter.

Outputs (under docs/imgs/trajectory-parameterization/):
  bezier_2d.mp4
  bspline_2d.mp4

Run::

    uv sync --extra sim
    uv run python scripts/render_trajectory_2d_videos.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trajectory_viz_common import (
    convex_hull_2d,
    cubic_bezier,
    interpolating_spline_2d,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "imgs" / "trajectory-parameterization"

WIDTH = 640
HEIGHT = 480
FPS = 30
DURATION_S = 6.0
PAD = 48

# Normalized coordinates in [0, 1] × [0, 1] (y up).
BEZIER_CTRL = (
    np.array([0.08, 0.12]),
    np.array([0.22, 0.78]),
    np.array([0.62, 0.68]),
    np.array([0.88, 0.15]),
)
BSPLINE_WAYPOINTS = np.array(
    [
        [0.10, 0.18],
        [0.24, 0.72],
        [0.42, 0.55],
        [0.64, 0.22],
        [0.86, 0.62],
    ],
    dtype=float,
)


def _world_to_px(xy: np.ndarray) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    px = PAD + x * (WIDTH - 2 * PAD)
    py = HEIGHT - PAD - y * (HEIGHT - 2 * PAD)
    return int(round(px)), int(round(py))


def _draw_grid(draw: ImageDraw.ImageDraw) -> None:
    grid_color = (230, 233, 238)
    for i in range(1, 5):
        t = i / 5.0
        x = PAD + t * (WIDTH - 2 * PAD)
        y = HEIGHT - PAD - t * (HEIGHT - 2 * PAD)
        draw.line([(x, PAD), (x, HEIGHT - PAD)], fill=grid_color, width=1)
        draw.line([(PAD, y), (WIDTH - PAD, y)], fill=grid_color, width=1)
    draw.rectangle(
        [PAD, PAD, WIDTH - PAD, HEIGHT - PAD],
        outline=(180, 186, 196),
        width=1,
    )


def _polyline(draw: ImageDraw.ImageDraw, pts: np.ndarray, **kwargs) -> None:
    if len(pts) < 2:
        return
    draw.line([_world_to_px(p) for p in pts], **kwargs)


def _draw_frame(
    *,
    title: str,
    curve: np.ndarray,
    markers: np.ndarray,
    marker_labels: list[str],
    hull: np.ndarray | None,
    control_edges: np.ndarray | None,
    dot_u: float,
) -> np.ndarray:
    img = Image.new("RGB", (WIDTH, HEIGHT), (252, 252, 253))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    _draw_grid(draw)

    if hull is not None and len(hull) >= 3:
        hull_px = [_world_to_px(p) for p in hull]
        draw.polygon(hull_px, fill=(255, 228, 220, 90), outline=(255, 190, 170, 140))

    if control_edges is not None:
        _polyline(draw, control_edges, fill=(120, 150, 210), width=2)

    _polyline(draw, curve, fill=(25, 95, 185), width=3)

    for i, pt in enumerate(markers):
        px, py = _world_to_px(pt)
        r = 7
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255), outline=(25, 95, 185), width=2)
        label = marker_labels[i] if i < len(marker_labels) else ""
        if label:
            draw.text((px + 9, py - 9), label, fill=(40, 50, 65), font=font)

    # Moving dot along curve by arc-length fraction dot_u ∈ [0, 1].
    idx = int(round(dot_u * (len(curve) - 1)))
    idx = max(0, min(len(curve) - 1, idx))
    dpx, dpy = _world_to_px(curve[idx])
    dr = 9
    draw.ellipse(
        [dpx - dr, dpy - dr, dpx + dr, dpy + dr],
        fill=(230, 95, 45),
        outline=(180, 60, 20),
        width=2,
    )

    draw.rectangle([0, 0, WIDTH, 30], fill=(20, 28, 40))
    draw.text((10, 8), title, fill=(255, 255, 255), font=font)
    draw.text(
        (10, HEIGHT - 24),
        "Drag handles in the interactive demo below (built book only).",
        fill=(90, 98, 110),
        font=font,
    )
    return np.asarray(img, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], path: Path) -> None:
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
        str(FPS),
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


def _animate(curve: np.ndarray, **draw_kwargs) -> list[np.ndarray]:
    n_frames = int(DURATION_S * FPS)
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        frames.append(_draw_frame(dot_u=u, curve=curve, **draw_kwargs))
    hold = int(0.5 * FPS)
    for _ in range(hold):
        frames.append(frames[-1].copy())
    return frames


def render_bezier_video() -> Path:
    u = np.linspace(0.0, 1.0, 240)
    curve = cubic_bezier(*BEZIER_CTRL, u)
    ctrl = np.stack(BEZIER_CTRL)
    hull = convex_hull_2d(ctrl)
    path = OUT / "bezier_2d.mp4"
    frames = _animate(
        curve,
        title="Cubic Bézier — control polygon, hull, u: 0 → 1",
        markers=ctrl,
        marker_labels=["P0", "P1", "P2", "P3"],
        hull=hull,
        control_edges=ctrl,
    )
    _write_mp4(frames, path)
    print(f"wrote {path}  ({len(frames)} frames @ {FPS} fps)")
    return path


def render_bspline_video() -> Path:
    u = np.linspace(0.0, 1.0, 320)
    curve = interpolating_spline_2d(BSPLINE_WAYPOINTS, u)
    path = OUT / "bspline_2d.mp4"
    labels = [f"W{i}" for i in range(len(BSPLINE_WAYPOINTS))]
    frames = _animate(
        curve,
        title="Cubic B-spline — interpolating waypoints in the plane",
        markers=BSPLINE_WAYPOINTS,
        marker_labels=labels,
        hull=None,
        control_edges=BSPLINE_WAYPOINTS,
    )
    _write_mp4(frames, path)
    print(f"wrote {path}  ({len(frames)} frames @ {FPS} fps)")
    return path


def main() -> None:
    render_bezier_video()
    render_bspline_video()
    print("done")


if __name__ == "__main__":
    main()
