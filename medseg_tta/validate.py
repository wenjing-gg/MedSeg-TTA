import importlib

from .common.legacy import legacy_entry_path, legacy_root
from .registry import METHOD_ALIASES, METHODS, PARADIGMS, TABLE_METHODS


def validate_structure():
    errors = []
    paradigm_slugs = {paradigm.slug for paradigm in PARADIGMS}
    method_slugs = [method.slug for method in METHODS]
    if len(method_slugs) != len(set(method_slugs)):
        errors.append("duplicate method slugs")
    if len(METHOD_ALIASES) != len(set(METHOD_ALIASES)):
        errors.append("duplicate method aliases")
    for method in METHODS:
        if method.paradigm_slug not in paradigm_slugs:
            errors.append(f"{method.slug}: unknown paradigm {method.paradigm_slug}")
        try:
            importlib.import_module(method.package)
        except Exception as exc:
            errors.append(f"{method.slug}: cannot import {method.package}: {exc}")
        for dimension in method.dimensions:
            try:
                importlib.import_module(f"{method.package}.{dimension}")
            except Exception as exc:
                errors.append(f"{method.slug}: cannot import {method.package}.{dimension}: {exc}")
            root = legacy_root(method, dimension)
            if not root.is_dir():
                errors.append(f"{method.slug}: missing {dimension} legacy root {root}")
                continue
            for entry in method.entries_by_dimension.get(dimension, ()):
                try:
                    legacy_entry_path(method, dimension, entry)
                except FileNotFoundError:
                    errors.append(f"{method.slug}: missing {dimension} entry {entry}")
    for row in TABLE_METHODS:
        if row["paradigm_slug"] not in paradigm_slugs:
            errors.append(f"{row['name']}: unknown table paradigm {row['paradigm_slug']}")
    return tuple(errors)


def main():
    errors = validate_structure()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("structure_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
