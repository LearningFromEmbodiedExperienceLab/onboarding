# Third-party assets

Robot models and upstream URDFs are fetched locally — not stored in this repository.

- **Lock:** `assets.lock.json` (pinned git refs + sparse paths for each vendor)
- **Fetch:** `bash scripts/fetch_menagerie_assets.sh` from repo root
- **Checkouts:** `mujoco_menagerie/`, `piper_ros/`, `arx_model/` (all gitignored)

See [Asset files](../docs/asset-files.qmd) in the tutorial.
