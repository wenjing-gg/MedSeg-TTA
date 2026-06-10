from __future__ import annotations

import runpy
import sys
from pathlib import Path


LEGACY_ROOT = Path(__file__).resolve().parent
ENTRYPOINT = LEGACY_ROOT / "procedures" / "validation.py"


def _has_arg(argv: list[str], name: str) -> bool:
    return name in argv or any(item.startswith(f"{name}=") for item in argv)


def _print_help() -> int:
    print("usage: input_level_transformation/AIF-SFDA/two_d/validation2d.py [legacy arguments]")
    print()
    print("AIF-SFDA validation entrypoint")
    print("Forward target: procedures/validation.py")
    print()
    print("Default injected args: --model_name AIF_SFDA")
    print("Use --metrics_as_sort_index to choose the checkpoint-selection metric.")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        raise SystemExit(_print_help())
    if not _has_arg(argv, "--model_name"):
        argv = ["--model_name", "AIF_SFDA", *argv]
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(ENTRYPOINT), *argv]
        for item in [str(ENTRYPOINT.parent), str(LEGACY_ROOT)]:
            if item not in sys.path:
                sys.path.insert(0, item)
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
