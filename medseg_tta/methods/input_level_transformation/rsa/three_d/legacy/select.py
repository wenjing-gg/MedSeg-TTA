from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("rsa") / "2_select.py"
    runpy.run_path(str(target), run_name="__main__")
