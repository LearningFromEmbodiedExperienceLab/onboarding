#!/usr/bin/env bash
# Generate type stubs for MuJoCo (C++ / pybind11 bindings) so the editor
# can hover and complete mj.MjModel, mj.mj_step, etc.
#
# Usage, from the onboarding folder:
#   uv sync --extra mujoco
#   bash scripts/generate_mujoco_stubs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/typings"
mkdir -p "$OUT"

echo "Generating MuJoCo stubs into $OUT/mujoco/"
uv run --no-sync --with pybind11-stubgen \
  pybind11-stubgen mujoco -o "$OUT" --ignore-all-errors

echo "Done. Reload the editor window if hover types do not appear yet."
