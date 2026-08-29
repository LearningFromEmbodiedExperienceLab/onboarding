#!/usr/bin/env bash
# Fetch pinned MuJoCo Menagerie robot folders (sparse checkout).
# Assets stay outside git — see third_party/menagerie.lock.json and .gitignore.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/menagerie.lock.json"
DEST="$ROOT/third_party/mujoco_menagerie"

if [[ ! -f "$LOCK" ]]; then
  echo "Missing lock file: $LOCK" >&2
  exit 1
fi

eval "$(
  python3 - "$LOCK" << 'PY'
import json, shlex, sys
lock = json.load(open(sys.argv[1]))
print(f"REPO={shlex.quote(lock['repo'])}")
print(f"REF={shlex.quote(lock['ref'])}")
print("PATHS=(" + " ".join(shlex.quote(p) for p in lock["paths"]) + ")")
PY
)"

if [[ -d "$DEST/.git" ]]; then
  echo "Updating menagerie checkout in $DEST"
  git -C "$DEST" fetch --depth 1 origin "$REF"
  git -C "$DEST" checkout "$REF"
  git -C "$DEST" sparse-checkout set "${PATHS[@]}"
else
  echo "Cloning menagerie (sparse) into $DEST"
  mkdir -p "$(dirname "$DEST")"
  git clone --filter=blob:none --sparse "$REPO" "$DEST"
  git -C "$DEST" sparse-checkout set "${PATHS[@]}"
  git -C "$DEST" checkout "$REF"
fi

echo "Menagerie ready at $REF:"
for p in "${PATHS[@]}"; do
  echo "  - $DEST/$p"
done
