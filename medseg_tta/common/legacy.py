from __future__ import annotations

import runpy
import sys
from pathlib import Path

from ..registry import resolve_entry, resolve_method


def _method_base(spec):
    return Path(__file__).resolve().parents[1] / "methods" / spec.paradigm_slug / spec.slug


def legacy_root(spec, dimension: str) -> Path:
    return _method_base(spec) / dimension / "legacy"


def common_legacy_root(spec) -> Path:
    return _method_base(spec) / "common" / "legacy"


def legacy_roots(spec, dimension: str) -> list[Path]:
    roots = [legacy_root(spec, dimension)]
    common_root = common_legacy_root(spec)
    if common_root.is_dir():
        roots.append(common_root)
    return roots


def legacy_entry_path(spec, dimension: str, entry: str) -> Path:
    for root in legacy_roots(spec, dimension):
        path = root / entry
        if path.is_file():
            return path
    raise FileNotFoundError(entry)


def legacy_help(method, entry):
    spec, forced_dimension = resolve_method(method)
    dimension, rel_entry = resolve_entry(spec, entry, forced_dimension=forced_dimension)
    display_path = f"{spec.source_dir}/{dimension}/{rel_entry}"
    print(f"usage: {display_path} [legacy arguments]")
    print()
    print(f"{spec.name} legacy entrypoint wrapper")
    print(f"Forward target: {spec.package}.{dimension}.legacy/{rel_entry}")
    print()
    print("This lightweight help does not import optional medical-imaging dependencies.")
    print("Run without --help to execute the original entrypoint from the unified package copy.")
    return 0


def run_legacy_entrypoint(method, entry, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        return legacy_help(method, entry)
    spec, forced_dimension = resolve_method(method)
    dimension, rel_entry = resolve_entry(spec, entry, forced_dimension=forced_dimension)
    path = legacy_entry_path(spec, dimension, rel_entry)
    roots = legacy_roots(spec, dimension)
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(path), *argv]
        for item in [str(path.parent), *(str(root) for root in roots)]:
            if item not in sys.path:
                sys.path.insert(0, item)
        runpy.run_path(str(path), run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
