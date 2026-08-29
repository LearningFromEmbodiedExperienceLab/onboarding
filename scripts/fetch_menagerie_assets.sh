#!/usr/bin/env bash
# Fetch pinned third-party asset trees (Menagerie MJCF + upstream URDFs).
# See third_party/assets.lock.json and .gitignore.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/assets.lock.json"

if [[ ! -f "$LOCK" ]]; then
  echo "Missing lock file: $LOCK" >&2
  exit 1
fi

export ROOT
python3 - "$LOCK" << 'PY'
import json
import os
import shlex
import subprocess
import sys

lock_path = sys.argv[1]
root = os.environ["ROOT"]
vendors = json.load(open(lock_path))["vendors"]

for vendor in vendors:
    repo = vendor["repo"]
    ref = vendor["ref"]
    paths = vendor["paths"]
    dest = os.path.join(root, vendor["dest"])
    vid = vendor["id"]
    print(f"=== {vid} -> {dest} @ {ref[:12]}…")

    if os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "-C", dest, "fetch", "--depth", "1", "origin", ref], check=True)
        subprocess.run(["git", "-C", dest, "checkout", ref], check=True)
        subprocess.run(["git", "-C", dest, "sparse-checkout", "set", *paths], check=True)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", repo, dest],
            check=True,
        )
        subprocess.run(["git", "-C", dest, "sparse-checkout", "set", *paths], check=True)
        subprocess.run(["git", "-C", dest, "checkout", ref], check=True)

    for path in paths:
        print(f"  - {os.path.join(dest, path)}")

print("All vendor trees ready.")
PY
