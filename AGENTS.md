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
- `uv run python scripts/async_ipc_demo.py` — async sim (500 Hz) + controller
  (50 Hz) in separate processes via shared memory (stdlib only).
- `uv sync --extra ipc && uv run python scripts/async_ipc_zmq_demo.py` — same
  rates over ZeroMQ PUB/SUB on localhost.
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
default): `torch`, `mujoco`, `sim`, `ipc`, `geometry`, `notebooks`, `debug` — install with
`uv sync --extra <name>`. **`sim`** = `mujoco` + `motrixsim` (robotics sim chapters);
**`ipc`** = `pyzmq` (ZeroMQ async IPC demo); **`geometry`** = `scipy` + `trimesh` +
`scikit-image` (`scripts/geometric_computing_demo.py`); **`mujoco`** alone is enough for
menagerie figure rendering and MuJoCo-only scripts.
`torch`/`motrixsim` are large.

Robotics sim smoke test (after `bash scripts/fetch_menagerie_assets.sh`):
- `uv sync --extra sim`
- `uv run python scripts/mujoco_robot_demo.py`
- `uv run python scripts/motrix_robot_demo.py`
- `uv run python scripts/ik_tracking_demo.py` — differential IK tracking (reach + circle), headless
- `uv run python scripts/benchmark_ik_sim_dt.py` — max stable sim `dt` for dynamic IK + position actuators (MuJoCo integrators; `--quick` for smoke)
- `uv run python scripts/ik_substeps_demo.py` — lockstep vs substepped dynamic IK (Euler)
- `uv run python scripts/actuator_control_demo.py` — position vs motor actuators @ 50 Hz vs substep torque
- `uv run python scripts/pd_gain_stability_demo.py` — high `kp` / zero `kv` PD instability (Euler @ 10 ms)
- Regenerate doc videos: `uv run python scripts/render_ik_tracking_videos.py` (needs `mujoco` extra + ffmpeg)
- Regenerate trajectory chapter clips: `uv run python scripts/render_trajectory_2d_videos.py` (needs `sim` extra + ffmpeg)
- Regenerate instability clip: `uv run python scripts/render_ik_instability_video.py`
- Headless MuJoCo rendering: `MUJOCO_GL=egl` or `osmesa` (+ `PYOPENGL_PLATFORM=osmesa` for OSMesa); see `docs/environment-variables.qmd#displays-opengl-headless`.
- **Env var lookup:** `docs/environment-variables.qmd` (appendix at end of book); concept in `docs/environment-management.qmd#environment-variables`.

Docs (Quarto):
- The tutorial is a Quarto book under `docs/` (one `.qmd` per section). The root
  `readme.md` is just a short pointer to it.
- Quarto is a **standalone CLI**, not a Python/pip package, and is not installed
  by `uv sync`. Install it from <https://quarto.org/docs/get-started/> if it is
  missing (`quarto --version` to check).
- Build/preview with `quarto render docs` / `quarto preview docs`; output goes to
  the gitignored `docs/_site/`.
- Chapters use pandoc auto-generated heading ids for anchors; cross-chapter links
  are written as `other-file.qmd#anchor`. The debugger heading has an explicit
  `{#vscode-debugger}` id because its pandoc id would otherwise include a period
  (from `launch.json`). Reuse an explicit `{#id}` for any new heading whose title
  contains punctuation you want to link to.
- **Advanced** (after Robotics Basics): `inter-process-communication.qmd` (IPC:
  shared memory, ZMQ, LCM; runnable `async_ipc_demo.py` + `async_ipc_zmq_demo.py`),
  `trajectory-parameterization.qmd`, `geometric-computing.qmd` (explicit vs implicit
  shape; `scripts/geometric_computing_demo.py` needs `uv sync --extra geometry`), and
  `advanced-more-topics.qmd` (placeholder). No scripts in-repo yet for LCM.
- Pages currently contain no executable (`{python}`) Quarto cells, so rendering needs
  only the Quarto CLI. Adding executable cells later would require `jupyter` in
  the env (e.g. via a new `docs` extra).
- **External references:** curated PEP / install / simulator links live in
  `docs/references.qmd` (book chapter before the glossary). Chapters link inline
  or to `#references` anchors — extend that page when adding new cited tools.
- **In-browser exercises** on `tensor-array-indexing.qmd` use Pyodide + NumPy
  (`docs/tensor-exercises.js`, loaded via `tensor-exercises.html`). They need network
  access on first **Check** (CDN). No `uv` / local kernel required for readers.
- **Trajectory splines** on `trajectory-parameterization.qmd` use vanilla canvas JS
  (`docs/trajectory-splines.js`) — draggable Bézier control points and B-spline
  waypoints; no network required.
- GitHub Pages: `.github/workflows/publish-docs.yml` renders on push to
  `master`/`main` (when `docs/` changes) and deploys `docs/_site/`. One-time UI:
  Settings → Pages → Source: **GitHub Actions**. Live URL is set in
  `docs/_quarto.yml` (`website.site-url`).
- Reading progress: client-side only via `docs/progress.js` + `progress.css`
  (`localStorage` key `onboarding-book-progress:v1`). Marks `##` sections done;
  highlights unread entries in the page TOC and chapter status in the sidebar.
- Menagerie vendor assets: `third_party/assets.lock.json` +
  `scripts/fetch_menagerie_assets.sh` → gitignored vendor trees (`mujoco_menagerie/`,
  `piper_ros/`, `arx_model/`).
  Doc figures: `scripts/render_menagerie_geometry_figures.py` (needs `uv sync --extra
  mujoco`, OSMesa headless). CI: `.github/workflows/regenerate-menagerie-images.yml`.
  Doc figure script also needs `trimesh`/`scipy` (included in `mujoco` extra).
- Common bug warnings in HTML docs use `.common-bug-warning` in `docs/styles.css`
  (red text); apply `{.common-bug-warning}` on future footgun callouts.
