from __future__ import annotations

import runpy
import sys
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parent / "optic" / "vptta.py"


if __name__ == "__main__":
    old_path = sys.path[:]
    try:
        for item in [str(ENTRYPOINT.parent), str(ENTRYPOINT.parent.parent)]:
            if item not in sys.path:
                sys.path.insert(0, item)
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    finally:
        sys.path[:] = old_path

