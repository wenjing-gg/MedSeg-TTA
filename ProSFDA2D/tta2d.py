import sys
from pathlib import Path


def _root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'medseg_tta').is_dir():
            return parent
    return here.parent


repo_root = _root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from medseg_tta.common.legacy import run_legacy_entrypoint


if __name__ == '__main__':
    raise SystemExit(run_legacy_entrypoint('prosfda_2d', 'tta2d.py'))
