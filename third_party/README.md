# Third-party assets

Robot models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
are fetched locally — not stored in this repository.

- **Lock:** `menagerie.lock.json` (pinned git ref + sparse paths)
- **Fetch:** `bash scripts/fetch_menagerie_assets.sh` from repo root
- **Checkout:** `mujoco_menagerie/` (gitignored)

See [Asset files](../docs/asset-files.qmd#menagerie-vendor-assets) in the tutorial.
