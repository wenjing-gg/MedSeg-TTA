import argparse
import json
from .registry import METHODS, PARADIGMS, TABLE_METHODS, find_method, methods_by_paradigm, table_by_paradigm
from .common.legacy import run_legacy_entrypoint
from .validate import validate_structure


def _rows(methods):
    for method in methods:
        yield {
            'slug': method.slug,
            'name': method.name,
            'paradigm_slug': method.paradigm_slug,
            'paradigm': method.paradigm,
            'modality': method.modality,
            'dimension': method.dimension,
            'status': method.status,
            'entries': list(method.entries),
        }


def list_methods(args):
    methods = methods_by_paradigm(args.paradigm) if args.paradigm else METHODS
    data = list(_rows(methods))
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    width = max(len(item['name']) for item in data)
    if args.flat or args.paradigm:
        for item in data:
            print(f"{item['name']:<{width}}  {item['paradigm']}  {item['modality']}  {item['dimension']}  {item['status']}")
        return 0
    for paradigm in PARADIGMS:
        grouped = [item for item in data if item['paradigm_slug'] == paradigm.slug]
        if not grouped:
            continue
        print(paradigm.name)
        for item in grouped:
            print(f"  {item['name']:<{width}}  {item['modality']}  {item['dimension']}  {item['status']}")
    return 0


def show_method(args):
    method = find_method(args.method)
    if args.json:
        print(json.dumps(next(_rows([method])), ensure_ascii=False, indent=2))
        return 0
    print(f"Name: {method.name}")
    print(f"Slug: {method.slug}")
    print(f"Paradigm: {method.paradigm}")
    print(f"Paradigm slug: {method.paradigm_slug}")
    print(f"Original modality: {method.modality}")
    print(f"Original dimension: {method.dimension}")
    print(f"Package: {method.package}")
    print(f"Legacy source: {method.source_dir}")
    print(f"Summary: {method.summary}")
    print('Entrypoints:')
    for entry in method.entries:
        print(f"  - {entry}")
    return 0


def table(args):
    rows = table_by_paradigm(args.paradigm) if args.paradigm else TABLE_METHODS
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    width = max(len(row['name']) for row in rows)
    if args.flat or args.paradigm:
        for row in rows:
            print(f"{row['name']:<{width}}  {row['status']:<9}  {row['paradigm']}  {row['original_modality']}  {row['original_dimension']}")
        return 0
    for paradigm in PARADIGMS:
        grouped = [row for row in rows if row['paradigm_slug'] == paradigm.slug]
        if not grouped:
            continue
        print(paradigm.name)
        for row in grouped:
            print(f"  {row['name']:<{width}}  {row['status']:<9}  {row['original_modality']}  {row['original_dimension']}")
    return 0


def list_paradigms(args):
    data = []
    grouped = methods_by_paradigm()
    table_grouped = table_by_paradigm()
    for paradigm in PARADIGMS:
        data.append({
            'slug': paradigm.slug,
            'name': paradigm.name,
            'included_methods': [method.name for method in grouped[paradigm.slug]],
            'table_methods': [row['name'] for row in table_grouped[paradigm.slug]],
        })
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    for item in data:
        print(f"{item['name']} ({item['slug']})")
        print(f"  included: {', '.join(item['included_methods']) or '-'}")
        print(f"  table: {', '.join(item['table_methods'])}")
    return 0


def run_legacy(args):
    return run_legacy_entrypoint(args.method, args.entry, args.args)


def validate(args):
    errors = validate_structure()
    if args.json:
        print(json.dumps({'ok': not errors, 'errors': list(errors)}, ensure_ascii=False, indent=2))
        return 0 if not errors else 1
    if errors:
        for error in errors:
            print(error)
        return 1
    print('structure_ok')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='medseg_tta')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('list-methods')
    p.add_argument('--json', action='store_true')
    p.add_argument('--flat', action='store_true')
    p.add_argument('--paradigm')
    p.set_defaults(func=list_methods)
    p = sub.add_parser('show-method')
    p.add_argument('method')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=show_method)
    p = sub.add_parser('table')
    p.add_argument('--json', action='store_true')
    p.add_argument('--flat', action='store_true')
    p.add_argument('--paradigm')
    p.set_defaults(func=table)
    p = sub.add_parser('list-paradigms')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=list_paradigms)
    p = sub.add_parser('validate-structure')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=validate)
    p = sub.add_parser('run-legacy')
    p.add_argument('method')
    p.add_argument('entry')
    p.add_argument('args', nargs=argparse.REMAINDER)
    p.set_defaults(func=run_legacy)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
