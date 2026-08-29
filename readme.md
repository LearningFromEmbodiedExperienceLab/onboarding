# Onboarding

Tiny example package (`robotics`) plus a hands-on tutorial for getting productive
in this codebase.

## Quickstart

```bash
uv sync                              # create .venv (numpy, einops)
uv run python scripts/train.py       # smoke test: imports the library, runs IK
```

- Library code lives in `src/robotics/` (import it).
- Runnable examples live in `scripts/` (run them with `uv run python scripts/<name>.py`).

## Tutorial

The full tutorial (Python environments & imports, project structure, debugging,
version control, NumPy/PyTorch, and robotics basics) lives in `docs/` as a
[Quarto](https://quarto.org) book, split into one short chapter per topic so
sections can be edited and reviewed independently.

Build or preview it locally:

```bash
quarto preview docs     # live-reloading local preview
quarto render docs      # static HTML into docs/_site
```

Install the Quarto CLI from <https://quarto.org/docs/get-started/> (it is a
standalone tool, separate from the Python environment). Start reading at
`docs/index.qmd`.
