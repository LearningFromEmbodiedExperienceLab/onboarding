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

Install the [Quarto CLI](https://quarto.org/docs/get-started/) separately (standalone
tool, not a Python package). PEPs, install guides, and simulator docs are collected in
[`docs/references.qmd`](docs/references.qmd) (rendered as the **References** chapter).
Start reading at `docs/index.qmd`.

## Publishing (GitHub Pages)

Pushes to `master`/`main` that touch `docs/` run
[`.github/workflows/publish-docs.yml`](.github/workflows/publish-docs.yml),
which renders the book and deploys to GitHub Pages.

**One-time setup (GitHub UI, after the workflow is on the default branch):**

1. Repo **Settings → Pages → Build and deployment → Source:** choose **GitHub Actions**.
2. Merge or push the workflow, then either push a docs change or run **Actions → Publish docs → Run workflow**.
3. When the workflow succeeds, the site is at
   <https://learningfromembodiedexperiencelab.github.io/onboarding/>.

Do the UI step *after* the workflow file exists on GitHub; until then there is
nothing for Pages to deploy.
