"""Shared paths and guards for MuJoCo / Motrix robot demos."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Menagerie scene: AgileX Piper arm + ground plane (see docs/asset-files.qmd).
PIPER_SCENE = REPO_ROOT / "third_party/mujoco_menagerie/agilex_piper/scene.xml"

# End-effector link used in FK / control demos (Piper wrist / gripper base).
EE_LINK = "link8"

FIRST_ARM_ACTUATOR = "joint1"


def require_piper_scene() -> Path:
    """Return the Piper scene path or exit with fetch instructions."""
    if not PIPER_SCENE.is_file():
        raise SystemExit(
            "Piper Menagerie scene not found.\n"
            "  bash scripts/fetch_menagerie_assets.sh\n"
            "  uv sync --extra sim"
        )
    return PIPER_SCENE


def section(title: str) -> None:
    print(f"\n=== {title} ===")
