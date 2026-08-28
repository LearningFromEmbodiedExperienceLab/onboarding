"""Print the process environment that *this* Python sees.

Prefix the command to try values for one run::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src uv run python scripts/print_env.py
"""

import os
import sys

KEYS = (
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONBREAKPOINT",
    "MUJOCO_GL",
)


def main() -> None:
    print("executable:", sys.executable)
    for key in KEYS:
        print(f"{key}={os.environ.get(key)!r}")
    print("sys.path:")
    for entry in sys.path:
        print(" ", repr(entry))


if __name__ == "__main__":
    main()
