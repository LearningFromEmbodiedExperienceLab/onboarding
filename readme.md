# Python Basics

## Environment management

Python itself is just an interpreter. The libraries you `import` (NumPy, PyTorch, Hydra, …) are files sitting somewhere on disk. An **environment** is an isolated copy of Python plus the packages installed into it. Two projects that need different versions of the same library each get their own box, so they do not fight.

Always know **which Python you are talking to**. These two should agree:

```bash
which python
python -c "import sys; print(sys.executable)"
```

If they point at `/usr/bin/python3` (or some other env you did not mean), you are not in the environment you think you are. `ModuleNotFoundError` is almost always "wrong interpreter", not "the package does not exist".

### conda vs uv

**conda** (Miniconda / Mamba) manages *environments and more than Python*. It can install compilers, CUDA toolkits, and native libraries that are not pip wheels. Use it when a dependency is not a Python package — or when a stack is only distributed as a conda package.

**uv** is what we use day to day. It replaces `venv` + `pip` + lockfiles and is much faster. It:

- creates a virtualenv (usually `.venv/` in the project)
- installs Python packages from `pyproject.toml` / `uv.lock`
- can pin a Python version (`uv python pin 3.11`)
- runs a command *inside that env* without you having to remember `activate`

Rule of thumb: **uv for Python work**. Reach for conda only when you need non-Python bits. You can still run `uv` / `pip` *inside* a conda env — the env is just "which Python binary and which `site-packages`".

### Everyday uv

```bash
# one-off env in the current folder
uv venv --python 3.11
uv pip install numpy torch

# run without activating — always this project's env
uv run python -c "import torch; print(torch.__version__)"

# same thing after activating (optional)
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

If the folder has a `pyproject.toml`, the usual loop is:

```bash
uv sync                  # create / update .venv from the lockfile
uv run python train.py
```

Prefer `uv run …` over relying on a lingering `conda activate` / `source .venv/bin/activate` in your shell. It is harder to accidentally use the wrong env.

### How `import` finds things

`import numpy` does not search the internet. Python walks a list of directories called `sys.path` and looks for `numpy.py`, or a folder `numpy/` (a package), or a compiled extension of that name.

```python
import sys
print("\n".join(sys.path))
```

Typical entries, **in order**:

1. **The script's directory** (or `""` = current working directory in a REPL). This is why `import helpers` works when `helpers.py` sits next to the file you just ran.
2. Directories from the **[`PYTHONPATH`](#environment-variables)** environment variable, if you set it.
3. The environment's **`site-packages`**. That is where `uv pip install numpy` put NumPy. Each environment has its own `site-packages`. Switch env → different (or missing) `numpy`.

So "the environment" is: **this Python binary** + **this `site-packages` on `sys.path`**. Local files win if they appear earlier on the list. A classic footgun:

```text
my_project/
  numpy.py      # you named a file numpy.py
  train.py      # import numpy  → loads YOUR file, not the real library
```

The script directory is searched first, so your `numpy.py` shadows the installed package. Do not name files after third-party libraries.

`import x` and `from y import z` use the same lookup; only the syntax differs.

Quick checks when something will not import:

```bash
python -c "import sys; print(sys.executable)"   # wrong env?
python -c "import numpy; print(numpy.__file__)" # which copy did I get?
```

### Environment variables

A conda/uv **environment** is a Python + `site-packages`. An **environment variable** is a different thing: a `NAME=value` string the parent process (your shell, `launch.json`, a job scheduler) hands to the child. Python reads them through `os.environ`. They are how we pass *process-wide* knobs without changing code.

`scripts/print_env.py` prints the ones we care about and the `sys.path` they affect:

```bash
uv run python scripts/print_env.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src uv run python scripts/print_env.py
```

The prefix form (`NAME=value command`) lasts **one command**. `export NAME=value` lasts for the rest of that shell (and everything you start from it). Prefer the prefix for experiments: it cannot leak into the next job. Do not put `CUDA_VISIBLE_DEVICES` in `~/.bashrc` unless you really want every process on that machine pinned to the same GPU.

Check what the **Python process** sees, not only the shell:

```bash
echo "$CUDA_VISIBLE_DEVICES"                          # the shell
uv run python -c "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"
```

If those disagree, F5 / `uv run` / a notebook kernel is not the same process as that terminal.

#### `PYTHONPATH`

Colon-separated directories prepended to `sys.path` (item 2 in the list above). That is why `PYTHONPATH=src python scripts/train.py` can `import robotics` without an editable install — and why we still prefer `pip install -e .` (see [Project structure](#project-structure)).

```bash
PYTHONPATH=src uv run python -c "import robotics; print(robotics.__file__)"
```

Several paths: `PYTHONPATH=/a:/b` on Linux. A stale `PYTHONPATH` in your bashrc is a common source of "I import the *wrong* `robotics`".

The editor does **not** read your shell's `PYTHONPATH`. That is `extraPaths` (see [Editor and stubs](#editor-and-stubs)).

#### `CUDA_VISIBLE_DEVICES`

CUDA enumerates GPUs as `0, 1, 2, …`. This variable is a **mask**: the process only sees the IDs you list, **renumbered from zero**.

```bash
nvidia-smi                    # physical IDs on this machine
CUDA_VISIBLE_DEVICES=1 uv run python scripts/print_env.py
```

| You set | The process sees |
|---|---|
| unset | all GPUs, `cuda:0` is physical 0 |
| `1` | only physical 1, which appears as `cuda:0` |
| `1,3` | those two, as `cuda:0` and `cuda:1` |
| `""` or `-1` | no GPU (CPU fallback if the library allows it) |

So `tensor.to("cuda:0")` after `CUDA_VISIBLE_DEVICES=1` is **physical GPU 1**. That is the usual way to share a box without everyone stacking on GPU 0. Check `nvidia-smi` first.

PyTorch example (needs `torch` installed; not in the default extra):

```python
import os, torch
print(os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count())
```

#### Other knobs you will meet

| Variable | Typical use |
|---|---|
| `PYTHONBREAKPOINT` | debugger for `breakpoint()` (`ipdb.set_trace`, or `0` to disable) |
| `MUJOCO_GL` | MuJoCo GL backend (`egl` on a headless server, `glfw` on a desktop) |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | cap CPU threads so many jobs do not oversubscribe |
| `CUDA_LAUNCH_BLOCKING` | `1` makes CUDA kernels run synchronously (see [Bells and whistles](#bells-and-whistles)) |

The GUI debugger is a *different* parent process. Set the same names under `"env"` in `launch.json` (see [VS Code / Cursor debugger](#vs-code--cursor-debugger-and-launchjson)); a prefix in the terminal does not apply to F5.

## Project structure

The previous section is "where Python looks". This section is "what it looks *for*, and how we put *our* code on that path".

### Module vs package

A **module** is one importable file. `math_utils.py` is the module `math_utils`:

```python
# math_utils.py
def quat_mul(a, b):
    ...
```

```python
import math_utils
math_utils.quat_mul(q1, q2)
```

A **package** is a directory of modules. The directory name is the import prefix. We always include `__init__.py` so the intent is obvious (Python also has *namespace packages* without it; we do not rely on that):

```text
robotics/
  __init__.py           # may be empty, or re-export public names
  math_utils.py
  ik/
    __init__.py
    differential.py
```

```python
from robotics.math_utils import quat_mul
from robotics.ik.differential import damped_lstsq
```

`robotics.ik` is a **subpackage**: a package nested in another. The dotted name is just the folder path with `/` replaced by `.`.

`__init__.py` can re-export so callers do not have to know the internal layout:

```python
# robotics/__init__.py
from robotics.math_utils import quat_mul
```

```python
from robotics import quat_mul   # instead of robotics.math_utils
```

Keep that thin. A huge `__init__.py` that imports everything makes every `import robotics` slow and creates circular-import traps.

### What Python actually searches for

For `import robotics`, Python still walks `sys.path`. At each directory it tries, in order:

- `robotics.py` (a module), then
- `robotics/` (a package)

The **first match wins**. That is why "it works in the notebook but not as a script" is common: you started the notebook from a different working directory, so a different folder sat at `sys.path[0]`.

Relative imports (`from .math_utils import quat_mul`) only work *inside* a package, and only when that package was imported as a package — not when you run the file as a script (`python robotics/math_utils.py`). Run scripts as scripts; import library code as a package.

### Making our code importable

This folder *is* the example project:

```text
onboarding/
  pyproject.toml
  src/
    robotics/              # this is what you import
      __init__.py
      math_utils.py
      ik/
        __init__.py
        differential.py
  scripts/
    helpers.py             # neighbor of train.py — not a package
    train.py               # you run this; you do not import it
```

Three patterns, from "fine for a one-off" to "how we actually work".

**1. Same folder (scripts talking to neighbors)**

`scripts/train.py` can do `from helpers import greet` because running `python scripts/train.py` puts `scripts/` on `sys.path[0]`. Fine for a tiny helper that only that script needs. Breaks as soon as two scripts in different folders both want `helpers`, or you want `from robotics.ik import …` from a notebook started elsewhere.

**2. [`PYTHONPATH`](#environment-variables) (temporary; do not rely on this in a team)**

```bash
PYTHONPATH=src python scripts/train.py
```

Puts `src/` on `sys.path`. Convenient for an afternoon, fragile: everyone must remember the same [environment variable](#environment-variables), and IDEs / debuggers will not.

**3. Install the package into the environment (what we actually do)**

`pyproject.toml` tells the installer the package name, its dependencies, and that importable code lives under `src/`. Then:

```bash
uv sync                 # or: pip install -e .
uv run python scripts/train.py
```

### Why `pip install -e .`

The `src/` layout is a firewall. `robotics/` is **not** a child of the directory you run from — it sits one level down, under `src/`. Python never searches inside `src/` on its own.

Walk through `python scripts/train.py` *before* installing:

1. Python puts `…/onboarding/scripts` on `sys.path[0]`.
2. It looks there for `robotics.py` or `robotics/` — not present (`helpers.py` is, which is why `import helpers` still works).
3. It looks in the env's `site-packages` — `robotics` is not there either.
4. `ModuleNotFoundError: No module named 'robotics'`.

The same failure happens from a REPL or notebook unless the working directory happens to be `src/` (do not depend on that).

`pip install -e .` does two jobs:

1. **Install dependencies** listed in `pyproject.toml` into the environment (here, NumPy) — the same mechanism as `pip install numpy`.
2. **Register our package** in that environment's `site-packages`, so `import robotics` uses the same lookup as `import numpy`.

The **`-e` is the part that matters while you are writing code**. It means *editable* (also called a develop install).

| Command | What lands in `site-packages` | Edit `src/robotics/math_utils.py` |
|---|---|---|
| `pip install .` | A **copy** of the package | Python still imports the stale copy until you reinstall |
| `pip install -e .` | A **link** (`.pth` / egg-link) pointing at `src/` | Next `import` sees your live files |

Research is edit-run-edit. A copied install would force a reinstall after every change. Editable is the only mode that is usable.

`uv sync` does this editable install of the current project automatically. `uv pip install -e .` and `pip install -e .` are the explicit form — same effect. You only need to re-run the install when **metadata** changes (new dependency, renamed package), not when you edit `.py` files.

After it works, check that you got *your* tree, not a copy:

```bash
uv run python -c "import robotics; print(robotics.__file__)"
# …/onboarding/src/robotics/__init__.py
```

Then check that it does **not** depend on the current working directory:

```bash
cd /tmp
uv run --directory /path/to/onboarding python -c "import robotics; print(robotics.__file__)"
```

If that still prints a path under `src/`, the package lives on the environment — which is the whole point.

### `src/` layout

Importable packages live under `src/`; scripts you *run* live under `scripts/`. Installing via `pyproject.toml` is what makes `from robotics.ik.differential import damped_lstsq` work from `scripts/train.py`, a notebook, or a test.

Do not give a script the same name as a package you need (`robotics.py` next to the `robotics/` package). First match on `sys.path` wins.

### A mental model

| You want to… | Mechanism |
|---|---|
| Use `numpy`, `torch`, … | Installed into the env's `site-packages` |
| Use *our* library from anywhere in that env | `pyproject.toml` + `uv sync` / `uv pip install -e .` |
| Run a one-off script that imports a neighbor file | Script directory on `sys.path` (ok for small tutorials) |
| Debug `ModuleNotFoundError` | `sys.executable` (wrong env), then `sys.path` (never installed / shadowed) | 


## Debugging, REPL and Notebooks

Print-debugging gets you the value. A **debugger** lets you stop, look around, and try expressions *in the frame that failed*. A **REPL** / notebook is the same idea without a frozen stack: you type, see a figure, change one number.

The example to pause is `scripts/debug_ik.py`: a rank-1 Jacobian so `JᵀJ` is singular without damping. The notebook `notebooks/inspect_ik.ipynb` plots residual vs λ using the same function.

### `breakpoint()`, pdb, ipdb, pdbpp

`breakpoint()` is the built-in hook (Python 3.7+). It drops you into a debugger in **that** stack frame — no signal, no IDE required.

```bash
uv run python scripts/debug_ik.py
```

You get a `(Pdb)` prompt (or `(Pdb++)` once pdbpp is installed). Useful commands:

| Key | What it does |
|---|---|
| `n` | Next line in *this* function |
| `s` | Step **into** the next call (`damped_lstsq`) |
| `c` | Continue to the next breakpoint or the end |
| `l` / `ll` | Show source around here |
| `p expr` / `pp expr` | Print / pretty-print |
| `u` / `d` | Up / down the call stack |
| `q` | Quit |

At the first stop, `dq` does not exist yet. `n` over the `damped_lstsq` line, then `p dq`, `p J @ dq - dx`. `s` instead of `n` to land inside `src/robotics/ik/differential.py`.

**pdb** is the stdlib debugger. You can also start the script *under* it from line 1:

```bash
uv run python -m pdb scripts/debug_ik.py
```

**ipdb** is pdb with IPython tab-completion and nicer tracebacks. [`PYTHONBREAKPOINT`](#environment-variables) selects which debugger `breakpoint()` starts:

```bash
uv pip install ipdb
PYTHONBREAKPOINT=ipdb.set_trace uv run python scripts/debug_ik.py
```

**pdbpp** (pdb++) is a drop-in: install it and `breakpoint()` / `python -m pdb` become the nicer UI **with no code change**. That is why we use it.

```bash
uv sync --extra debug
uv run python scripts/debug_ik.py
# (Pdb++) sticky    ← side-by-side source; stay in this mode
```

[`PYTHONBREAKPOINT=0`](#environment-variables) disables every `breakpoint()` for one run (handy when you left one in by mistake).

`breakpoint()` is for "I know this line is interesting". An exception you did not expect is **post-mortem**:

```bash
uv run python -m pdb -c continue scripts/debug_ik.py
```

or, in IPython / a notebook after a crash, `%debug`.

### VS Code / Cursor debugger and `launch.json`

The GUI debugger is the same idea with a mouse: click the gutter to set a breakpoint, press F5, hover locals. It does **not** read your shell. It uses a **launch configuration** plus the selected interpreter.

`.vscode/launch.json` in this folder:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Current file",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "python": "${workspaceFolder}/.venv/bin/python",
            "justMyCode": true
        },
        {
            "name": "debug_ik",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/debug_ik.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "python": "${workspaceFolder}/.venv/bin/python",
            "justMyCode": false
        }
    ]
}
```

What the keys mean:

| Key | Why it is there |
|---|---|
| `type: debugpy` | The Python debugger extension |
| `program` | The file to run (`${file}` = whatever tab is active) |
| `cwd` | Working directory (same as "where you would have `cd`'d") |
| `python` | **Which** interpreter. Pin `.venv` so F5 is not the system Python |
| `console` | `integratedTerminal` so `input()` and pdb work |
| `justMyCode` | `true` skips NumPy/site-packages. `false` on `debug_ik` so **Step Into** reaches `damped_lstsq` |
| `args` | CLI arguments, as a JSON list of strings |
| `env` | Environment variables for **this** debug session (see [Environment variables](#environment-variables)). The **print_env** config sets `CUDA_VISIBLE_DEVICES` and `PYTHONPATH` |

Run: open `scripts/debug_ik.py` → red gutter dot on the `dq = …` line → Run and Debug → **debug_ik** (or Current file) → F5. The Variables / Watch panes are `p` / `pp` with less typing.

These configs apply when this folder is the workspace root. If you opened a parent folder, copy the entries into *that* `.vscode/launch.json` and fix the paths.

Do not fight the GUI debugger with `breakpoint()` *and* a gutter breakpoint on the same line unless you mean to stop twice.

### IPython and notebooks

`python` is a REPL. **IPython** is the same loop with tab-completion, `?` help, and magics (`%timeit`, `%debug`).

```bash
uv sync --extra notebooks
uv run ipython
```

```python
from robotics.ik.differential import damped_lstsq
damped_lstsq?
```

A **notebook** (`*.ipynb`) is IPython split into cells, with figures inline. Open `notebooks/inspect_ik.ipynb` and pick the `.venv` kernel. The plot is residual and `‖dq‖` against damping for the same rank-1 `J` — the kind of one-off figure that does not belong in `src/`.

Use notebooks for:

- plots and visual checks
- probing a checkpoint / array you do not want to write a script for yet
- `%debug` after a cell fails

Do **not** put library code in a notebook. Restarting the kernel does not make a 200-cell file a package; `import robotics` does. If you find yourself copy-pasting a cell into three notebooks, it is a function — move it to `src/`.

Notebooks are JSON with output blobs. They diff poorly; do not commit huge executed outputs. `.ipynb_checkpoints/` is already gitignored.

## GitHub and version control

Git is a history of snapshots **on your machine**. GitHub is a server that stores a copy of that history so other people (and future you) can `clone` it. The link between the two is a **remote**.

### Create and publish

A local repo is just a `.git/` folder. Publishing means: create an empty repo on GitHub, then push your commits there.

From a project folder that is **not** a git repo yet:

```bash
git init
git add .
git commit -m "Initial commit"
```

Then either:

```bash
# GitHub CLI: creates the repo and sets origin in one step
gh repo create myproject --private --source=. --remote=origin --push
```

or create the repo in the GitHub UI (empty, no README), then:

```bash
git remote add origin git@github.com:YOUR_USER/myproject.git
git push -u origin main
```

`-u` remembers that local `main` tracks `origin/main`, so later `git push` / `git pull` need no extra arguments.

`git add .` respects `.gitignore`. Put `.venv/`, `__pycache__/`, `*.pt`, `outputs/`, and anything with secrets in there **before** the first commit. History is forever: deleting a file later does not remove it from old commits.

### Remotes

A remote is a **named URL**. The conventional name for "the GitHub copy of this repo" is `origin`.

```bash
git remote -v
# origin  git@github.com:YOUR_USER/myproject.git  (fetch)
# origin  git@github.com:YOUR_USER/myproject.git  (push)
```

You can have more than one remote (`upstream` for the repo you forked from, a second GitHub org, …). Day to day you only need `origin`.

Change the URL without re-cloning (HTTPS → SSH, renamed repo, moved org):

```bash
git remote set-url origin git@github.com:YOUR_USER/myproject.git
git remote -v    # confirm
```

`git clone` sets `origin` for you. `git init` does not — that is why the first publish needs `git remote add`.

### SSH vs HTTPS

GitHub accepts two URL shapes:

| | HTTPS | SSH (recommended) |
|---|---|---|
| URL | `https://github.com/USER/repo.git` | `git@github.com:USER/repo.git` |
| Auth | Personal access token, often every push | SSH key on this machine, once |
| Typical pain | Token expiry, credential helpers, "Authentication failed" | First-time key setup, then silent |

HTTPS is fine for a one-off clone of a **public** repo. For a repo you push to every day, use SSH.

One-time setup on a new machine:

```bash
ssh-keygen -t ed25519 -C "you@university.edu"
# press enter for the default path; set a passphrase
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Paste the **public** key (`*.pub`) into GitHub → Settings → SSH and GPG keys. Test:

```bash
ssh -T git@github.com
# Hi YOUR_USER! You've successfully authenticated...
```

If `git remote -v` still shows `https://…`, switch it with `git remote set-url` as above. Cursor / VS Code's Git panel uses the same remotes as the terminal.

### Branch, develop, squash-and-merge

`main` should be the last **known-good** snapshot, not a scratchpad. Work on a branch:

```bash
git checkout main
git pull
git checkout -b feature/damped-ik
# edit, then:
git add -p                 # review hunks; do not `git add .` on autopilot
git commit -m "Add damped least-squares IK."
git push -u origin feature/damped-ik
```

Open a **pull request** on GitHub (branch → `main`). Review happens there. Merge with **Squash and merge** (the GitHub button), not a regular merge commit.

- Your branch may have ten "wip" / "fix typo" commits. That is fine.
- Squash folds them into **one** commit on `main`, with one message you write at merge time.
- `main`'s history stays linear and readable. The messy branch can be deleted.

After the PR is merged:

```bash
git checkout main
git pull
git branch -d feature/damped-ik
git push origin --delete feature/damped-ik
```

Do not commit directly to `main` on a shared repo. Do not `git push --force` to `main`. Force-pushing a feature branch you own is occasionally needed after a rebase; it is never the default.

### Large files: `.gitignore` first, Git LFS only if they must live in the repo

Git stores **every version of every file** in the history. A 200 MB checkpoint committed once stays in the clone forever, even if you delete it in a later commit. That is why clones become slow and GitHub will reject files over 100 MB.

**Do not put these in git at all** (ignore them; keep them on disk / object storage / Hugging Face):

- environments (`.venv/`, `venv/`)
- training outputs (`outputs/`, `wandb/`, `*.pt`, `*.ckpt`, `*.npz` datasets)
- rendered videos, logs, `__pycache__/`

This folder's `.gitignore` already excludes `.venv/` and generated MuJoCo stubs. Add lines as you add new artifact types.

**Git LFS** is for files that *belong* with the source and are large-but-stable: robot meshes (`.obj`, `.usd`), a small canonical MJCF mesh pack, a fixture binary a test must have. LFS stores a **pointer** in git and the bytes on a separate LFS server.

```bash
git lfs install
git lfs track "*.obj" "*.usd"
git add .gitattributes    # this file is what the team shares
git add assets/robot.obj
git commit -m "Track robot mesh with Git LFS."
```

Everyone who clones needs `git lfs install` once on that machine; otherwise they get tiny pointer files instead of the real meshes.

LFS is not a dataset dump. GitHub's LFS quota is small. Multi-gigabyte motion clips and checkpoints should live outside the repo (and the README should say where). If you accidentally committed a large file, fixing history is painful — ask before rewriting `main`.

## Making Life Easier

The interpreter will run whatever you wrote. The editor can only help if it *understands* the code. Two things make that possible: **type hints** in our Python, and **stubs** for libraries that have no Python source.

### Type hints

A type hint is a note on a name that says what it is supposed to be. Python **does not enforce** this at runtime (passing a `str` where you annotated `float` still runs). The editor, and optional checkers like Pyright, read the notes.

```python
def damped_lstsq(J, dx, damping=1e-3):
    ...
```

becomes (see `src/robotics/ik/differential.py`):

```python
def damped_lstsq(J: np.ndarray, dx: np.ndarray, damping: float = 1e-3) -> np.ndarray:
```

What you get:

- Hover over `damped_lstsq` in `scripts/train.py` and the editor shows the signature.
- Pass `damping="lots"` and the editor underlines it *before* you run anything.
- Autocomplete on `data.` only works if the editor knows `data`'s type.

Annotate arguments and return types first. That is most of the value. Variable annotations (`qpos: np.ndarray = …`) are optional.

Hints are not conversions. This still runs, and is still wrong:

```python
def f(x: float) -> float:
    return x

f("hello")  # runtime: ok; editor: error
```

Keep hints honest and simple (`np.ndarray`, `float`, `str`, `list[int]`). Over-precise NumPy shapes are optional later; a wrong hint is worse than none.

### Editor and stubs

The language server (Pylance in VS Code, Cursor's Pyright) is a second program. It does **not** use your shell's [`PYTHONPATH`](#environment-variables). It uses:

1. The **selected interpreter** (status bar → `.venv/bin/python`). That is how it finds `numpy` and an editable `robotics`.
2. **`extraPaths`** — extra folders to search, the editor's analogue of [`PYTHONPATH`](#environment-variables).
3. **`stubPath`** — a folder of `.pyi` files that overlay types onto installed packages.

`.vscode/settings.json` in this folder sets both (Cursor reads `cursorpyright.analysis.*`; VS Code reads `python.analysis.*`). They apply when this folder is the workspace root; if you opened a parent folder, copy the same keys into that workspace's settings.

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/src",
        "${workspaceFolder}/typings"
    ],
    "python.analysis.stubPath": "${workspaceFolder}/typings",
    "cursorpyright.analysis.extraPaths": [
        "${workspaceFolder}/src",
        "${workspaceFolder}/typings"
    ],
    "cursorpyright.analysis.stubPath": "${workspaceFolder}/typings"
}
```

`extraPaths` is why you add sibling source trees the editor should see (`src/` here; in a larger workspace, other packages that are not installed into this env). `stubPath` is for **types without source**.

The same keys can live in `pyproject.toml` under `[tool.pyright]`, so the checker is not editor-only.

#### Why MuJoCo needs stubs

`import mujoco` works at runtime. The library is a **C++ extension** (pybind11). There is no `mujoco.py` full of Python for the editor to parse, so every name is `Unknown`: no hover, no completion, noisy red squiggles.

A **stub** is a `.pyi` file: types only, never executed. `typings/mujoco/__init__.pyi` might contain lines like:

```python
def mj_step(m: MjModel, d: MjData, nstep: int = 1) -> None: ...
```

The editor reads that; `python scripts/mujoco_hello.py` still runs the real C++ module.

#### Generate them

```bash
uv sync --extra mujoco
bash scripts/generate_mujoco_stubs.sh
```

The script runs `pybind11-stubgen` against the *installed* `mujoco`, writing `typings/mujoco/*.pyi`. Then open `scripts/mujoco_hello.py`, hover `MjModel` / `mj_step`. If types do not appear, reload the window.

`pybind11-stubgen` **imports the package and inspects it**. Stubs must be regenerated when you bump the MuJoCo version. Generated files can include machine-specific OpenGL backends (`glfw` vs `egl`); that is fine for local editor use. Do not treat them as a hand-written API.

Select the `.venv` interpreter so `import mujoco` in the editor is the same copy the stub generator saw.

## Useful Patterns

The IK folder (`src/robotics/ik/`) is one small design: several solvers, one call site. The patterns below are how that stays extensible. Run the walkthrough:

```bash
uv run python scripts/ik_controllers.py
```

### Dunder (magic) methods

Names like `__len__` and `__call__` are how Python implements `len(x)`, `x[i]`, `x()`. You rarely invent new ones; you implement the ones the language already calls.

`IKController` (`src/robotics/ik/base.py`):

| Dunder | Syntax | Here |
|---|---|---|
| `__call__` | `dq = ctrl(J, dx)` | compute and remember last `dq` |
| `__len__` | `len(ctrl)` | batch size of that `dq` |
| `__getitem__` / `__setitem__` | `ctrl[i]`, `ctrl[i] = 0` | read / override per-env `dq` |
| `__repr__` | `repr(ctrl)` | debugger / logs |

`ctrl(J, dx)` is the same object as `ctrl.__call__(J, dx)`. `__init__` is the constructor, not a dunder you use at the call site. Do not implement `__eq__` unless you know what "equal controllers" means.

### Subclassing, registry, singleton

`IKController.compute` is abstract. `TransposeIK`, `PinvIK`, and `DampedIK` only implement that method — call, length, and indexing come free.

A **registry** is a dict `name → class` so config strings (`"dls"`) construct objects without `if name == …`. `@register("dls")` on the class (a **decorator** that runs at import) fills the table.

```python
from robotics.ik import make_ik

ctrl = make_ik("dls", damping=1e-2)
dq = ctrl(J, dx)
```

The table itself is a **singleton**: `IKRegistry()` and `IKRegistry.instance()` are the same process-wide object (`__new__` returns the cached instance). That is appropriate for "the list of known solvers". It is **not** appropriate for a controller — each `make_ik` returns a **new** instance with its own last `dq`. If two threads shared one `DampedIK`, `__setitem__` would fight.

Importing `robotics.ik` loads `solvers.py` so the `@register` lines run. A solver you never import is invisible.

### Decorators

A decorator is a function that takes a function (or class) and returns a replacement. `@register("pinv")` is executed at **class definition**, not at each `compute`.

`timed` in `src/robotics/context.py` is a decorator that wraps a call in a `Timer`. Use it on a helper you always want timed. Use the context manager (next) when you only want a few lines.

`functools.wraps` copies `__name__` / `__doc__` onto the wrapper so `help()` and traces still make sense. Always use it.

### Context managers

`with x:` calls `x.__enter__()` on the way in and `x.__exit__(…)` on the way out — **including** if the block raises. That is the point: release / restore cannot be forgotten.

**Timer** (`Timer` in `src/robotics/context.py`) — later sections reuse this for host-side profiling:

```python
from robotics.context import Timer

timer = Timer("dls")
with timer:
    dq = ctrl(J, dx)
print(timer.elapsed)  # seconds
```

On CUDA, pair it with a synchronize or CUDA events ([Bells and whistles](#bells-and-whistles)); otherwise you time the launch, not the kernel.

**Controlled random state** — pin `np.random` for a reproducible snippet, then put the global RNG back so the rest of the process is not stuck on seed `0`:

```python
from robotics.context import numpy_seed

with numpy_seed(0):
    noise = np.random.randn(*dx.shape)
```

New code should prefer an isolated generator, `rng = np.random.default_rng(0)`, which never touches globals. `numpy_seed` exists for blocks that still call `np.random.randn` (or a library that does). Torch's analogue is `with torch.random.fork_rng(): torch.manual_seed(0)`.

`contextlib.contextmanager` turns a generator with one `yield` into `__enter__` / `__exit__` so you do not write a class for the seed helper. `Timer` is a class because callers need `.elapsed` after the block.

### NamedTuple and dataclass

A **dict** of `{"pos": …, "quat": …}` loses typos until runtime (`pose["poss"]`). A two-element tuple ` (pos, quat)` loses names. `NamedTuple` and `dataclass` are named fields with almost no boilerplate. Both live in `src/robotics/ik/types.py`.

**`NamedTuple`** is a tuple with names. Immutable *slots*, unpackable, indexable:

```python
from robotics.ik import Pose

pose = Pose(pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
pose.pos          # by name
pose[0]           # still a tuple
pos, quat = pose  # unpack
# pose.pos = np.ones(3)  # AttributeError — the field binding is frozen
pose.pos[:] = 0           # the array *inside* is still mutable
```

Use it for a small, fixed record you pass around (a pose, an `(dq, residual)` pair). Do not put a growing config here: no default factories, no `__post_init__` validation beyond what you write by hand, and `replace` is clunkier (`pose._replace(pos=…)`).

**`dataclass`** is a class the compiler fills in (`__init__`, `__repr__`, `__eq__`). Defaults, validation, and `replace` are the reason we use it for configs:

```python
from dataclasses import replace
from robotics.ik import IKConfig

cfg = IKConfig(method="dls", damping=1e-2)
ctrl = cfg.build()                 # make_ik under the hood
stiffer = replace(cfg, damping=0.1)
```

`__post_init__` runs after the generated `__init__` — `IKConfig(damping=-1)` raises. `@dataclass(frozen=True)` makes field assignment fail, like a NamedTuple (arrays inside are still mutable).

| | `NamedTuple` | `dataclass` |
|---|---|---|
| Unpack / `pose[0]` | yes | no (unless you add it) |
| Defaults, `__post_init__` | awkward | natural |
| `replace` | `_replace` | `dataclasses.replace` |
| Typical use | pose, small result | Hydra-style config, many optional fields |

Neither replaces a tensor. Do not `.item()` a CUDA batch into a NamedTuple in a hot loop ([Bells and whistles](#bells-and-whistles)). For a config you serialize or override from a YAML, a dataclass (or `OmegaConf`) is the usual choice.

# PyTorch and NumPy

## Tensor / Array indexing

NumPy `ndarray` and `torch.Tensor` share one indexing grammar. Learn it once. Examples below are NumPy (`scripts/index_tensors.py`); replace `np` with `torch` and the **shapes** match. View vs copy is the main place the two libraries disagree — see the table at the end.

The layout we care about in simulation is batched:

```text
x.shape == (n_env, n_body, 3)    # env, body, xyz
```

### Prefer tensor ops over Python loops

A Python `for` over environments runs on the CPU, one row at a time. The array/GPU path applies the same op to the whole batch in one kernel.

```python
# don't
out = np.empty_like(x)
for i in range(n_env):
    out[i] = x[i] * 2

# do
out = x * 2
```

Loops over a **tiny** axis (3 XYZ components, 4 quaternion entries) are sometimes clearer and not the bottleneck. Loops over `n_env` (hundreds to thousands) are.

### Slicing (basic indexing)

Integers, `start:stop:step`, `...`, and `None` (new axis). This is **basic indexing**. In NumPy a slice is a **view**: it aliases the same memory.

| Expression | Shape from `(4, 3, 3)` | Meaning |
|---|---|---|
| `x[0]` | `(3, 3)` | env 0; the env axis **drops** |
| `x[0:1]` | `(1, 3, 3)` | env 0; the env axis **stays** |
| `x[:, 1, :]` | `(4, 3)` | body 1 of every env |
| `x[..., 0]` | `(4, 3)` | the `x` coordinate of every body |
| `x[:, None, :, :]` | `(4, 1, 3, 3)` | insert an axis (for broadcasting) |

```python
y = x[0]
y[0, 0] = -1   # x[0, 0, 0] is now -1  (same storage)
```

`...` means "every axis I did not name". Useful when rank varies (`(…, 3)` positions, `(…, 4)` quaternions). `None` inserts an axis of size 1 — that is how you *enable* broadcasting.

### Broadcasting

Arithmetic and advanced indexing use the **same** shape rule. Get this solid before `x[env_ids, body_ids, dim_ids]`.

Align shapes **from the right**. For each axis:

- equal sizes → ok
- one of them is `1` → stretch it
- one array is shorter → pretend it has leading `1`s
- otherwise → `ValueError` / runtime error

```text
  (n_env, n_body, 3)
* (      n_body, 1)
  -------------------
  (n_env, n_body, 3)
```

`mass[:, None]` turns a per-body vector `(n_body,)` into `(n_body, 1)` so it scales all three xyz of that body, in every env — no loop.

```python
mass = np.array([1.0, 10.0, 100.0])   # (n_body,)
x * mass[:, None]                     # (n_env, n_body, 3)
```

Pairwise distances without a double `for`: insert axes so every env is paired with every env.

```python
roots = x[:, 0, :]                              # (n_env, 3)
delta = roots[:, None, :] - roots[None, :, :]  # (n_env, 1, 3) - (1, n_env, 3)
dist = np.linalg.norm(delta, axis=-1)          # (n_env, n_env)
```

Broadcasting is silent when the shapes *happen* to match. That is the bug class.

```python
np.zeros((4, 4)) + np.arange(4)
# (4, 4) + (4,) → last axis matches, so `arange` is added to *every row*.
# Per-column?  np.arange(4)[:, None]  → (4, 1)

np.zeros((4, 3)) + np.arange(4)
# (4, 3) + (4,) → 3 vs 4 → error. Lucky.

x * mass
# (n_env, n_body, 3) * (n_body,) → 3 vs n_body → error unless n_body == 3.
# You wanted mass[:, None].
```

Print shapes. `keepdims=True` on reductions (`mean`, `sum`) is there so the result still broadcasts.

### Advanced indexing

Index with **integer arrays** or **boolean masks**. NumPy always returns a **copy**. Several index arrays are **broadcast against each other first** (the rule above), then used as a gather: `out[…] = x[env[…], body[…], dim[…]]`. The output shape is that broadcast shape.

```bash
uv run python scripts/index_tensors.py
```

**Homogeneous zip** — all 1-D, same length. One output per pair, *not* a grid:

```python
env_ids = [0, 2];  body_ids = [1, 0];  dim_ids = [2, 1]
x[env_ids]                      # (2, 3, 3)   two whole envs
x[env_ids, body_ids]           # (2, 3)
x[env_ids, body_ids, dim_ids]  # (2,)        x[0,1,2] and x[2,0,1]
# out[k] == x[env_ids[k], body_ids[k], dim_ids[k]]
```

Same list of bodies (or dims) for every selected env is the other homogeneous case: `body_ids` has shape `(B,)` and you insert axes so it stretches across `K`.

**Heterogeneous** — *some* envs, and **each of those envs has its own bodies and its own dims**. The index tensors are 2-D (or 3-D); they are not a shared 1-D list. You broadcast them into a common `(K, B, D)` grid of triples.

```python
# env 0 → bodies 0,2 and dims x,z
# env 2 → bodies 1,0 and dims y,x
env_ids  = np.array([0, 2])            # (K,)
body_ids = np.array([[0, 2], [1, 0]])  # (K, B)
dim_ids  = np.array([[0, 2], [1, 0]])  # (K, D)

e = env_ids[:, None, None]   # (K, 1, 1)
b = body_ids[:, :, None]     # (K, B, 1)
d = dim_ids[:, None, :]     # (K, 1, D)
out = x[e, b, d]             # (K, B, D)
# out[k, i, j] == x[env_ids[k], body_ids[k, i], dim_ids[k, j]]
```

`None` is doing the same job as `mass[:, None]`: size-1 axes that stretch. Without them, `(K,)`, `(K, B)`, `(K, D)` do not broadcast.

If the dim choice also varies **per body** (not just per env), `dim_ids` is already `(K, B, D)` and you only expand `env_ids` and `body_ids`:

```python
out = x[env_ids[:, None, None], body_ids[:, :, None], dim_ids]
```

| You want | Index shapes after `None` | `out` |
|---|---|---|
| zip, one triple per `k` | `(K,)`, `(K,)`, `(K,)` | `(K,)` |
| same bodies & dims for every selected env | `(K,1,1)`, `(1,B,1)`, `(1,1,D)` | `(K, B, D)` |
| **per-env bodies and per-env dims** | `(K,1,1)`, `(K,B,1)`, `(K,1,D)` | `(K, B, D)` |
| per-env, per-body dims | `(K,1,1)`, `(K,B,1)`, `(K,B,D)` | `(K, B, D)` |

Forget `None` and you get a zip or a shape error. Print `e.shape, b.shape, d.shape, out.shape` until it is a reflex.

Boolean masks follow the copy rule: `x[x[..., 2] > 0]` flattens every `True` location. They do not give you a `(K, B, D)` block; use integer ids when you need that structure.

On **CUDA**, the index must live on the **same device** as `x`. A Python `list` (or a CPU `LongTensor` / NumPy array) is converted on the host and copied to the GPU. That copy [synchronizes](#bells-and-whistles). Keep `env_ids` as `torch.long` on `x.device` for the whole step — do not rebuild them with `.tolist()` inside the loop. A single Python `int` (`x[3]`) is fine; a list of ids is not.

```python
# don't — list → CPU tensor → blocking H2D copy every call
x[ [0, 2, 5] ]
x[env_ids.tolist()]

# do — index already on the GPU
x[env_ids]   # env_ids.dtype == torch.long, env_ids.device == x.device
```

### Indexed get vs indexed set

**Get** (`y = x[idx]`) builds an array. If `idx` was advanced indexing, `y` is a copy: mutating `y` does **not** change `x`.

**Set** (`x[idx] = value`) writes through into `x`. `value` must broadcast to the shape of `x[idx]`.

```python
y = x[env_ids, body_ids]   # 1-D zip: copy, shape (K, 3)
y[:] = 0                   # x unchanged

x[e, b, d] = 0             # heterogeneous write; value broadcasts to (K, B, D)
x[e, b, d] = np.zeros((2, 1, 2))  # same, via broadcasting
```

A slice on the left of `=` is a view, so `x[0] = 0` also writes into `x`. The rule is: assignment into an indexing expression writes to `x`; assignment into a name bound earlier to a *copy* does not.

### View vs copy

| | NumPy | PyTorch |
|---|---|---|
| `x[0]`, `x[:, 1]`, `x[..., :2]` | **view** | **view** (must be contiguous enough) |
| `x[env_ids]`, `x[env_ids, body_ids]` | **copy** | **copy** (new tensor) |
| duplicate storage | `y = x.copy()` | `y = x.clone()` |
| same storage, new shape | `x.reshape(…)` may copy | `x.view(…)` fails if not contiguous; `x.reshape` / `x.contiguous()` |

Torch-only: `x.view(4, -1)` is a view and **requires** contiguous memory. Prefer `reshape` unless you know the tensor is contiguous — details in [Shape manipulation](#shape-manipulation).

In-place math on a view (`x[0] *= 0`) mutates `x`. In-place math on a copy does not. When a gradient-related error says a tensor does not own its storage, you indexed a view.

`scripts/index_tensors.py` walks broadcasting, then the per-env gather, and prints shapes. Read the shapes first, the values second.

### Shape manipulation

Indexing *selects*. These ops *rearrange* axes without (usually) changing values. `scripts/reshape_tensors.py` prints each case.

NumPy and PyTorch names differ. **einops** is the same in both — use it for anything that is more than a single `unsqueeze`.

#### `view` vs `reshape`

The product of sizes must stay the same: `(4, 3, 3)` can become `(4, 9)` or `(12, 3)`, not `(4, 4, 3)`.

| | PyTorch | NumPy |
|---|---|---|
| Same storage, new shape | `x.view(4, 9)` — **fails** if not contiguous | `x.reshape` *may* be a view |
| Always succeed | `x.reshape(4, 9)` — copies if it must | `x.reshape(4, 9)` |
| Force contiguous | `x.contiguous().view(…)` | `np.ascontiguousarray` |

`transpose` / `permute` only **relabel** axes; memory order is unchanged, so the tensor is often **not** contiguous. Then `view` raises. `reshape` is the default.

**Do not** use `ndarray.view` in NumPy for this. That method is a **dtype pun** (`float64` → `uint8`), not a reshape. Torch `Tensor.view` is a reshape.

```python
x.reshape(n_env, -1)          # (n_env, n_body * 3)
x.permute(0, 2, 1)           # torch: (n_env, 3, n_body); not contiguous
# x.view(n_env, 3, n_body)   # may fail
x.reshape(n_env, 3, n_body)  # always ok
```

#### `squeeze` / `unsqueeze`

`unsqueeze(d)` (NumPy: `np.expand_dims`, or `x[..., None]`) inserts a size-1 axis. That is broadcasting's `None`.

`squeeze` **removes** size-1 axes. `squeeze()` with no argument drops *every* 1. If you ever run with `n_env == 1`, a bare `squeeze()` will also delete the batch axis and the next line that expected `(n_env, …)` breaks. Always `squeeze(d)` / `squeeze(-1)`.

```python
mass[:, None]                 # (B,) → (B, 1)     unsqueeze
x.unsqueeze(-1)              # torch; same as x[..., None]
x.squeeze(-1)                # only the last axis, if it is 1
```

#### `expand` vs `repeat`

Both make a size-1 axis look longer. They are not the same.

**`expand`** (PyTorch) is a **view**: it does not copy. You can only expand axes that are already size 1. Writes through an expanded tensor are undefined / errors — treat it as read-only. NumPy's equivalent is broadcasting (or `np.broadcast_to`, also read-only).

**`repeat`** **copies** the data. Use it when you need independent storage (or when einops is doing a named repeat).

```python
# torch
ones = torch.ones(n_env, 1, 3)
ones.expand(n_env, n_body, 3)     # view, no extra memory
ones.repeat(1, n_body, 1)         # copy
ones.expand(n_env, n_body, 3).clone()  # if you must write
```

`repeat` of a large batch is a common silent memory blow-up. If broadcasting or `expand` would do, prefer that.

#### einops (recommended)

`permute` + `reshape` + `unsqueeze` chains are how axis bugs are born: the numbers are meaningless and the next reader cannot tell `n_env` from `n_body`. einops names the axes.

```python
from einops import rearrange, repeat, reduce

# (n_env, n_body, 3) → (n_env, 3, n_body)
rearrange(x, "n b xyz -> n xyz b")

# flatten bodies and xyz for an MLP
rearrange(x, "n b xyz -> n (b xyz)")

# split a feature dim you packed earlier
rearrange(feat, "n (b xyz) -> n b xyz", b=n_body, xyz=3)

# NHWC image → NCHW
rearrange(img, "n h w c -> n c h w")

# copy a hidden state across time (this *is* a repeat / copy)
repeat(hx, "n h -> n t h", t=T)

# mean over bodies
reduce(x, "n b xyz -> n xyz", "mean")
```

`...` means "whatever leading axes": `rearrange(feat, "... (m c) -> ... m c", m=1)` works for `(n, d)` and `(t, n, d)`.

Parentheses **compose** axes (`(b xyz)` is one dim of size `b * xyz`); a split needs the sizes on the right-hand side (`b=n_body`).

einops works on NumPy and torch (and keeps the tensor on the same device). Prefer it over `view`/`permute` whenever more than one axis moves. Keep `unsqueeze`/`None` for a single broadcasting axis — that is still clearer as `mass[:, None]`.

```bash
uv run python scripts/reshape_tensors.py
```

## Writing batch-agnostic operations

A helper is **batch-agnostic** when it does not care how many leading axes you stacked: the same function handles a single quaternion `(4,)`, a batch `(n_env, 4)`, and bodies `(n_env, n_body, 4)`.

The recipe, used throughout `src/robotics/math_utils.py`:

1. **Feature axes last.** Quaternions are `(..., 4)`, vectors `(..., 3)`, Jacobians `(..., m, n)`. Leading axes are "batch" and may be missing.
2. Index with `...` and reduce with `axis=-1` (torch: `dim=-1`). Never `x[0]` / `x[:, 0]` unless you *mean* a batch slot.
3. Reductions that must broadcast back use `keepdims=True` (`keepdim=True` in torch).
4. Two inputs that should line up: `np.broadcast_arrays` / `torch.broadcast_tensors`.
5. If an algorithm is only written for rank 2, flatten then restore: `batch = q.shape[:-1]; q = q.reshape(-1, 4); …; return out.reshape(batch + (4,))`.

```python
# don't — only works for a single quaternion
w0, x0, y0, z0 = q

# do — last axis, any leading shape
w0, x0, y0, z0 = np.moveaxis(q, -1, 0)
```

`normalize` is the smallest complete example: the norm is `(..., 1)` so it divides every component.

```python
def normalize(x, eps=1e-6):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
```

`quat_rotate` / `quat_mul` / `yaw_rotate` follow the same pattern (`..., 4` and `..., 3`). `damped_lstsq` is the matrix analogue: `J` is `(..., m, n)`, `dx` is `(..., m)` — `np.swapaxes` / `matmul` on the last two axes, `np.eye(n)` broadcasts into `(..., n, n)`.

`yaw_rotate(yaw[:, None], vec)` is [broadcasting](#broadcasting) again: a per-env yaw against `(n_env, n_body, 3)` needs a size-1 body axis, same as `mass[:, None]`.

```bash
uv run python scripts/batch_ops.py
```

The script calls the *same* functions at three ranks. If a new helper only works for `(n_env, 4)`, it is not done.

## CUDA

Two different pieces of software share the name "CUDA". Mixing them up is why `nvidia-smi` works but `nvcc` is missing, or why a wheel installs and then `torch.cuda.is_available()` is false.

**The driver** is the kernel module that talks to the GPU. If `nvidia-smi` prints a table of cards, the driver is fine. You need it on **every** machine that runs GPU code. It does not compile anything.

**nvcc** is the CUDA *compiler* (part of the CUDA toolkit). It turns `.cu` files into GPU binaries. You need it when you **build** an extension from source (a custom CUDA kernel, some robotics / rendering packages). You do **not** need it to *run* a PyTorch wheel: that wheel already contains compiled kernels and a matching CUDA runtime.

| You are… | Need driver? | Need nvcc? |
|---|---|---|
| Running `tensor.cuda()` / training with a PyTorch CUDA wheel | yes | no |
| `pip install` a package that compiles CUDA on this machine | yes | yes (toolkit compatible with that PyTorch) |
| CPU-only torch | no | no |

The driver advertises a **maximum** CUDA version it can run. The PyTorch wheel's tag (`cu128`, …) must not exceed that. `nvidia-smi` shows "CUDA Version" in the header — that is the driver's cap, not the toolkit you compiled with.

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Pick a GPU with [`CUDA_VISIBLE_DEVICES`](#environment-variables). After that mask, this process's `cuda:0` is the first *visible* card, not necessarily physical 0.

PyTorch tensors live on a **device**. NumPy arrays do not. `x.cuda()` / `x.to("cuda")` copies; ops on `cuda` tensors run on the GPU. Mixing `cpu` and `cuda` in one expression is an error. Keep a batch on one device; `.cpu().numpy()` is a host copy (and a sync — next section).

## Bells and whistles

On CPU, the next Python line runs after the math finished. On CUDA, `x @ x` **queues a kernel** and Python continues. The GPU may still be working. That is why a training step looks instant until you print a loss or call `.item()`.

**Host-device sync** (Python waits for the GPU) happens when you need a value on the CPU, including:

| Call | Why it waits |
|---|---|
| `tensor.item()` | one scalar on the host |
| `tensor.cpu()` / `.numpy()` | copy to host memory |
| `print(tensor)` | has to format values |
| `cuda_tensor[python_list]` or `cuda_tensor[cpu_index]` | index is built/copied on the host |
| `torch.cuda.synchronize()` | explicit wait |
| some implicit copies (e.g. `cpu_tensor.copy_(cuda_tensor)`) | host must see the data |

Until a sync, **timing with `time.perf_counter()` is lying**: you measured launch overhead, not the matmul. Use CUDA events, or synchronize first:

```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
y = x @ x
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end), "ms")
```

`scripts/cuda_async.py` prints launch time vs time-to-finished. Skip it if you have no GPU.

**Debugging:** CUDA errors are often reported at the *next* sync, not at the kernel that faulted. [`CUDA_LAUNCH_BLOCKING=1`](#environment-variables) makes each kernel wait before Python continues so the traceback points at the right line. It is slow; use it only to hunt a crash.

Do not sync inside a hot loop (no `.item()` per env). Sync once per step if you need a logged scalar. Keep `print` of large CUDA tensors out of the inner loop.

Advanced indexing is the easy one to miss: `x[env_ids]` is async if `env_ids` is already a CUDA `long` tensor; `x[[0, 2, 5]]` is not. Build ids on device once (or keep them there from the sim). See [Advanced indexing](#advanced-indexing).



# Robotics Basics

Two engines, one set of ideas. **MuJoCo** is the academic default (`mjModel` / `mjData`). **Motrix** (`motrixsim`) loads the same MJCF/URDF into `SceneModel` / `SceneData` and adds a named-object API plus a viewer.

Two maps show up constantly:

- **Forward kinematics (FK):** joint coordinates → body and site poses. “Where is the gripper if the joints are `q`?”
- **Inverse kinematics (IK):** a desired pose → joint coordinates (or a joint step `dq`). “What `q` puts the gripper there?”

Every subsection below shows the concept once, then the two APIs side by side. A small shared MJCF will be the running example; the IK case study reuses `src/robotics/ik/`.

## Kinematics vs dynamics

**Kinematics** is the geometry of motion: poses, joint angles, and how they relate. No mass, no force, no “will it actually get there.” FK and IK are kinematic. Setting `qpos` and running FK *teleports* the robot to that configuration.

**Dynamics** is Newton’s laws: given inertia, gravity, contact, and actuator forces, what acceleration happens next, then integrate. `mj_step` / `motrixsim.step` is dynamics (plus the FK needed to update poses). Mass, COM ([inertial frame](#link-frame-vs-inertial-frame)), and actuators matter here; they do not change FK.

| | Kinematics | Dynamics |
|---|---|---|
| Question | Where is it if `q` is this? | What happens if I apply these forces? |
| Inputs | `q` (and maybe `q̇` as geometry) | forces / `ctrl`, plus `q`, `q̇`, inertia |
| Typical call | FK, Jacobian, IK | `step`, inverse dynamics (`q̈` → `τ`) |
| Failure mode | Unreachable pose | Too weak, slips, tips, oscillates |

**Inverse dynamics** is not IK. IK: pose → `q`. Inverse dynamics: a desired `q̈` (or trajectory) → the torques that would produce it. Our `DampedIK` is the former.

Writing **controls** (`ctrl`) is the dynamic path: the integrator consumes them. Writing **state** (`qpos`) is kinematic: you skip the physics of getting there. The [state I/O](#state-sensors-and-actuators) and [simulation loop](#the-simulation-loop) sections use that split.

## Scene vocabulary

Names that appear in both formats and both APIs:

| Idea | Typical name | What it is |
|---|---|---|
| Rigid piece with inertia | body / link | Mass, pose |
| Allowed motion | joint | Hinge, slide, ball, free… |
| Collision / visual shape | geom | Sphere, capsule, mesh; not a DoF |
| Named frame, no mass | site | End-effector, sensor origin |
| Map `ctrl` → force/position | actuator | Motor, position, velocity |

URDF says *link*; MJCF and MuJoCo say *body*. Motrix exposes both `Body` (rigid body) and `Link` (kinematic frame). Joints, geoms, sites, and actuators are the same ideas in both.

*(To fill: a one-page picture of a 2-DoF arm; URDF vs MJCF cheat-sheet.)*

## Rotations and frames

Poses are a position plus a rotation. Reading a body pose (FK) or sending a target to IK only makes sense after you fix two conventions: **how the rotation is stored**, and **which frame it is in**.

### Rotation representations

All helpers in `src/robotics/math_utils.py` are batch-agnostic on the last axis. **Every quaternion in this repo is `(w, x, y, z)`** (scalar first). That matches MuJoCo `xquat`. ROS, SciPy, and many USD/Isaac stacks use **`(x, y, z, w)`**. Mixing them looks like a 90° bug, not an import error.

```python
from robotics.math_utils import quat_from_euler_xyz, quat_rotate, wxyz_to_xyzw, xyzw_to_wxyz

q = quat_from_euler_xyz(np.array([0.0, 0.0, np.pi / 2]))  # (w, x, y, z)
quat_rotate(q, np.array([1.0, 0.0, 0.0]))                 # → +Y
# quat_rotate(wxyz_to_xyzw(q), …)  # wrong — treats x as w
```

| Representation | Shape | Pros / cons |
|---|---|---|
| Rotation matrix | `(..., 3, 3)` | Apply with `@`; 9 numbers, not unique numerically |
| Quaternion `(w, x, y, z)` | `(..., 4)` | Compose with `quat_mul`; **`q` and `-q` are the same rotation** |
| XYZ Euler (RPY) | `(..., 3)` | Human-readable yaw; gimbal lock; convention-dependent |
| Axis-angle | `(..., 3)` | Axis × radians; good for small `δR` in IK |
| Yaw only | `(...,)` | Planar base; `yaw_rotate` |

Conversions (same file): `matrix_from_quat`, `quat_from_euler_xyz` / `euler_from_quat`, `axis_angle_from_quat` / `quat_from_angle_axis`, `wxyz_to_xyzw` / `xyzw_to_wxyz`. Euler here is **intrinsic XYZ** (roll about X, then pitch Y, then yaw Z). Other libraries’ “RPY” may differ — convert through a matrix or quaternion, do not copy three floats blindly.

`wrap_to_pi` is for *angles*, not for comparing quaternions. To compare two orientations, use a geodesic (`quat_mul(quat_conjugate(q0), q1)` then `axis_angle` magnitude), not `q0 - q1`.

```bash
uv run python scripts/rotations.py
```

### Link frame vs inertial frame

“Inertial” is used in two ways. Do not mix them.

**World / Newtonian inertial frame.** A frame fixed to the ground (MuJoCo’s world, `xpos` in world). Body poses from FK are usually *this* frame. Velocities may be world or body — check the API (`cvel` vs `qvel`).

**Inertial frame of a body** (COM frame). URDF `<inertial origin>` and MJCF `<inertial pos quat>` define a second frame on the *same* rigid body: origin at the center of mass, axes along principal inertia. The **link / body frame** is where the joint, geoms, and sites attach. They need not coincide.

```text
world
  └── body / link frame     ← joints, geoms, sites, FK `xpos` / `xquat`
        └── inertial frame   ← COM, `ipos` / `iquat`; MuJoCo `xipos` / `ximat`
```

Setting `pos` on a geom does **not** move the COM unless inertia is computed `from_geom` (or you edit `<inertial>`). A mesh whose visual origin is not the COM is the usual source of “it tips over in sim.” When reading poses: site/body pose is the **link** frame; if you need the COM in world, use the inertial quantities (`xipos` in MuJoCo), not `xpos`.

Motrix `Body` vs `Link` follows the same split: the rigid body (inertia) vs a kinematic frame you hang sensors and EE sites on.

## Asset files

How a robot is stored on disk: **URDF** (ROS, tree of links) vs **MJCF** (MuJoCo XML: compiler, defaults, actuators, sensors, keyframes). Motrix `load_model` accepts MJCF, URDF, or USD. We will author **one MJCF** and load it in both simulators.

Mesh paths, compiler flags, and “inertia from geom” belong here — not in the control loop.

*(To fill: the tutorial MJCF; `mujoco.MjModel.from_xml_path` vs `motrixsim.load_model`.)*

## Two simulators

A simulator is **model** (constant: tree, inertias, actuator gears) plus **data** (state that changes every step).

| | MuJoCo | Motrix |
|---|---|---|
| Model | `mjModel` | `SceneModel` |
| State | `mjData` | `SceneData` |
| Advance | `mj_step(m, d)` | `motrixsim.step(model, data)` |
| FK only | `mj_fwdPosition` | `forward_kinematic(model, data)` |

`step` is **dynamics** (integrate forces). FK-only is **kinematics** (update poses from `q` with no forces) — see [Kinematics vs dynamics](#kinematics-vs-dynamics). That distinction matters for IK and for “did I just teleport `qpos`?”.

*(To fill: install extras, a 10-line load-and-step in each API, `MUJOCO_GL`.)*

## Forward kinematics

FK, defined above: joint coordinates → body/site poses. No forces. After writing `qpos` / `dof_pos`, you must run the simulator’s FK (or a full `step`) before `xpos` / link poses are valid.

We will not derive the product-of-exponentials by hand; we will *ask the simulator* and check it against `quat_rotate` from `src/robotics/math_utils.py` ([Rotations and frames](#rotations-and-frames)). Sites are the EE frames for the IK case study.

*(To fill: read `d.site_xpos` vs Motrix `Site` pose; batch-agnostic shapes.)*

## State, sensors, and actuators

The I/O contract, independent of engine:

- **Read** configuration and velocity (`qpos`/`qvel` vs `dof_pos`/`dof_vel`), body/site poses, sensors.
- **Write controls** (`data.ctrl` / `actuator_ctrls`, or named `MotorActuator`) — the **dynamic** path: forces into `step`.
- **Write state** (set `qpos`, reset, keyframe, mocap) — **kinematic**; follow with FK or `reset`. Do not fight the integrator by setting `qpos` every step unless you mean to.

Sensors are declared in MJCF and read after `step` (or after the documented sensor stage). Named lookup (Motrix `get_sensor_values`, MuJoCo `sensor` adr) vs slicing a flat array.

*(To fill: a table of the actual attribute names; a tiny PD-on-actuator example in both APIs.)*

## The simulation loop

How simulated time moves:

```text
write ctrl  →  step (integrate + FK + sensors)  →  read state
```

Two *control* schedules, not “does the window redraw on another thread”.

**Synchronous (lockstep).** One policy tick per physics step (or per fixed decimation you still wait for):

```text
obs = read(state)
ctrl = policy(obs)     # Python waits
write(ctrl)
step()                 # then the next obs is from this action
```

Easy to debug; `Timer` around `policy` vs `step` is honest. Unrealistic: a real robot’s motors do not freeze while the GPU thinks.

**Asynchronous (policy ‖ physics).** Physics keeps stepping at `timestep`. The policy runs at a slower rate (and/or takes wall-clock time). Until a new action arrives, the last `ctrl` is **held** (zero-order hold). Optional **delay**: the action computed from `obs(t)` is applied only at `t + Δ`.

```text
every physics step:
    write(held_ctrl)
    step()
    if control period elapsed:
        obs = read(state)           # may be stale vs when ctrl will apply
        held_ctrl = policy(obs)    # or kick this off and apply next period
```

That is closer to onboard compute + a 1 kHz PD loop. Policies that only work lockstep often degrade here (delay, multiple physics steps per action, observation from an older `ctrl`).

Viewer/render (MuJoCo viewer, Motrix `RenderApp`) is a separate issue: do not confuse UI FPS with the control period.

*(To fill: lockstep vs `n_substeps` hold vs delayed action, same MJCF, plot tracking error; `Timer` on policy vs sim.)*

## Case study: IK tracking

Put the pieces together. A site tracks a target pose:

1. FK → site pose (this section).
2. Pose error → `dq` via `make_ik("dls")` (or Motrix `ik.DlsSolver` / `IkChain` for comparison).
3. Write joints or actuator targets; `step`.
4. Loop.

Same `IKConfig` / registry as [Useful Patterns](#useful-patterns). Success is residual vs time, not a screenshot.

*(To fill: one script, two backends; optional Motrix built-in IK vs our NumPy solvers.)*


# Glossary and Terminology

