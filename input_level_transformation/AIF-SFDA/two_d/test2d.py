from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = PROJECT_ROOT / "medseg_tta" / "methods" / "input_level_transformation" / "aif_sfda" / "two_d" / "legacy"
ENTRYPOINT = LEGACY_ROOT / "procedures" / "test.py"


def _print_help() -> int:
    print("usage: input_level_transformation/AIF-SFDA/two_d/test2d.py [legacy arguments]")
    print()
    print("AIF-SFDA legacy test wrapper")
    print("Forward target: medseg_tta.methods.input_level_transformation.aif_sfda.two_d.legacy/procedures/test.py")
    print()
    print("Default injected args: --model_name AIF_SFDA")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        raise SystemExit(_print_help())
    if "--model_name" not in argv:
        argv = ["--model_name", "AIF_SFDA", *argv]
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(ENTRYPOINT), *argv]
        for item in [str(ENTRYPOINT.parent), str(LEGACY_ROOT), str(PROJECT_ROOT)]:
            if item not in sys.path:
                sys.path.insert(0, item)
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path

