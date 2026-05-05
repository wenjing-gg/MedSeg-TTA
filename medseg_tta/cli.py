import argparse
import json
from .registry import METHODS, TABLE_METHODS, find_method
from .common.legacy import run_legacy_entrypoint


def _rows(methods):
    for method in methods:
        yield {
            'slug': method.slug,
            'name': method.name,
            'paradigm': method.paradigm,
            'modality': method.modality,
            'dimension': method.dimension,
            'status': method.status,
            'entries': list(method.entries),
        }


def list_methods(args):
    data = list(_rows(METHODS))
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    width = max(len(item['name']) for item in data)
    for item in data:
        print(f"{item['name']:<{width}}  {item['paradigm']}  {item['modality']}  {item['dimension']}")
    return 0


def show_method(args):
    method = find_method(args.method)
    if args.json:
        print(json.dumps(next(_rows([method])), ensure_ascii=False, indent=2))
        return 0
    print(f"Name: {method.name}")
    print(f"Slug: {method.slug}")
    print(f"Paradigm: {method.paradigm}")
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
    rows = TABLE_METHODS
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    width = max(len(row['name']) for row in rows)
    for row in rows:
        print(f"{row['name']:<{width}}  {row['status']:<9}  {row['paradigm']}  {row['original_modality']}  {row['original_dimension']}")
    return 0


def run_legacy(args):
    return run_legacy_entrypoint(args.method, args.entry, args.args)


def build_parser():
    parser = argparse.ArgumentParser(prog='medseg_tta')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('list-methods')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=list_methods)
    p = sub.add_parser('show-method')
    p.add_argument('method')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=show_method)
    p = sub.add_parser('table')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=table)
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
