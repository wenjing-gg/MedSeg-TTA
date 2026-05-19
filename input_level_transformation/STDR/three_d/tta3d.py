import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


repo_root = _root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from medseg_tta.common.legacy import run_legacy_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_legacy_entrypoint("stdr", "tta3d.py"))
