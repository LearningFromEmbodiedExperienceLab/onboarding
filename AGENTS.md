# AGENTS.md

## Cursor Cloud specific instructions

This repo is a tiny single-package Python project (`robotics`) used as an onboarding
tutorial. It is a plain library plus runnable example scripts — there are **no**
servers, databases, queues, containers, or network ports, so nothing needs to be
"started" and left running.

Environment:
- Managed with `uv` (see `uv.lock` + `pyproject.toml`). The startup update script
  already runs `uv sync`, which creates `.venv/` and installs the editable
  `robotics` package plus `numpy`/`einops`. You normally don't need to reinstall.
- Always invoke Python through `uv run` (e.g. `uv run python scripts/train.py`) so
  the correct `.venv` is used, as the readme recommends.

Running / smoke test (this is the end-to-end check):
- `uv run python scripts/train.py` — imports the library and runs quaternion math +
  an IK solver. Other demo scripts: `scripts/ik_controllers.py`,
  `scripts/rotations.py`, `scripts/print_env.py`.
- Scripts under `scripts/` import a sibling `helpers.py` and rely on the script
  directory being on `sys.path[0]`; this works automatically with
  `uv run python scripts/<name>.py` (don't `cd` into `scripts/` to run them).

Gotchas:
- `scripts/debug_ik.py` intentionally hits an interactive `breakpoint()` (pdb) — it
  is a debugging demo and will exit with `bdb.BdbQuit` when run non-interactively.
  This is expected, not an environment failure.

Tests / lint:
- There is **no** automated test suite and no linter declared in dependencies.
- The project's effective "lint" is Pyright type-checking (configured under
  `[tool.pyright]` in `pyproject.toml`). Run it against the project venv so
  third-party imports resolve:
  `uvx pyright --pythonpath /workspace/.venv/bin/python src/robotics`.
  Note: `src/robotics/ik/registry.py` has pre-existing type warnings (intentional
  tutorial code); a clean file to sanity-check against is `src/robotics/math_utils.py`.

Optional extras (only needed for the corresponding demo scripts, not installed by
default): `torch`, `mujoco`, `notebooks`, `debug` — install with
`uv sync --extra <name>` (e.g. `uv sync --extra torch`). `torch`/`mujoco` are large.
