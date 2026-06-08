from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medseg_tta.common.legacy import run_legacy_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_legacy_entrypoint("vptta", "polyp.py"))
