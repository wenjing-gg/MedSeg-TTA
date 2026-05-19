from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("STDR") / "3select_active_samples_w_256.py"
    runpy.run_path(str(target), run_name="__main__")
