import runpy
import sys
from pathlib import Path
from ..registry import find_method


def legacy_root(spec):
    return Path(__file__).resolve().parents[1] / 'methods' / spec.paradigm_slug / spec.slug / 'legacy'


def legacy_help(method, entry):
    spec = find_method(method)
    print(f"usage: {spec.source_dir}/{entry} [legacy arguments]")
    print()
    print(f"{spec.name} legacy entrypoint wrapper")
    print(f"Forward target: {spec.package}.legacy/{entry}")
    print()
    print('This lightweight help does not import optional medical-imaging dependencies.')
    print('Run without --help to execute the original entrypoint from the unified package copy.')
    return 0


def run_legacy_entrypoint(method, entry, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--help' in argv or '-h' in argv:
        return legacy_help(method, entry)
    spec = find_method(method)
    root = legacy_root(spec)
    path = root / entry
    if not path.is_file():
        raise FileNotFoundError(path)
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(path), *argv]
        for item in [str(root), str(path.parent)]:
            if item not in sys.path:
                sys.path.insert(0, item)
        runpy.run_path(str(path), run_name='__main__')
        return 0
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
