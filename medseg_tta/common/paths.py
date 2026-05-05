from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def method_legacy_root(paradigm_slug, slug):
    return project_root() / 'medseg_tta' / 'methods' / paradigm_slug / slug / 'legacy'
