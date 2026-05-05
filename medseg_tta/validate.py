import importlib
from .common.legacy import legacy_root
from .registry import METHODS, PARADIGMS, TABLE_METHODS


def validate_structure():
    errors = []
    paradigm_slugs = {paradigm.slug for paradigm in PARADIGMS}
    method_slugs = [method.slug for method in METHODS]
    if len(method_slugs) != len(set(method_slugs)):
        errors.append('duplicate method slugs')
    for method in METHODS:
        if method.paradigm_slug not in paradigm_slugs:
            errors.append(f'{method.slug}: unknown paradigm {method.paradigm_slug}')
        try:
            importlib.import_module(method.package)
        except Exception as exc:
            errors.append(f'{method.slug}: cannot import {method.package}: {exc}')
        root = legacy_root(method)
        if not root.is_dir():
            errors.append(f'{method.slug}: missing legacy root {root}')
            continue
        for entry in method.entries:
            if not (root / entry).is_file():
                errors.append(f'{method.slug}: missing entry {entry}')
    for row in TABLE_METHODS:
        if row['paradigm_slug'] not in paradigm_slugs:
            errors.append(f"{row['name']}: unknown table paradigm {row['paradigm_slug']}")
    return tuple(errors)


def main():
    errors = validate_structure()
    if errors:
        for error in errors:
            print(error)
        return 1
    print('structure_ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
