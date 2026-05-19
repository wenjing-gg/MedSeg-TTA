from __future__ import annotations

from dataclasses import dataclass


DIMENSION_LABELS = {
    "two_d": "2D",
    "three_d": "3D",
}
DIMENSION_ALIASES = {
    "2d": "two_d",
    "two_d": "two_d",
    "twod": "two_d",
    "2_d": "two_d",
    "3d": "three_d",
    "three_d": "three_d",
    "threed": "three_d",
    "3_d": "three_d",
}


@dataclass(frozen=True)
class ParadigmSpec:
    slug: str
    name: str


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    name: str
    source_dir: str
    paradigm_slug: str
    paradigm: str
    modality: str
    status: str
    summary: str
    dimensions: tuple[str, ...]
    entries_by_dimension: dict[str, tuple[str, ...]]
    aliases: dict[str, str]
    package: str

    @property
    def dimension_labels(self) -> tuple[str, ...]:
        return tuple(DIMENSION_LABELS[dimension] for dimension in self.dimensions)

    @property
    def display_dimensions(self) -> str:
        return "/".join(self.dimension_labels)


def _normalize(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def normalize_dimension(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize(value)
    if normalized not in DIMENSION_ALIASES:
        raise KeyError(value)
    return DIMENSION_ALIASES[normalized]


def _build_spec(item: dict) -> MethodSpec:
    dimensions = tuple(item["dimensions"])
    entries_by_dimension = {
        dimension: tuple(item["entries_by_dimension"][dimension])
        for dimension in dimensions
    }
    aliases = {
        _normalize(alias): dimension
        for alias, dimension in item.get("aliases", {}).items()
    }
    return MethodSpec(
        slug=item["slug"],
        name=item["name"],
        source_dir=item["source_dir"],
        paradigm_slug=item["paradigm_slug"],
        paradigm=item["paradigm"],
        modality=item["modality"],
        status=item["status"],
        summary=item["summary"],
        dimensions=dimensions,
        entries_by_dimension=entries_by_dimension,
        aliases=aliases,
        package=f"medseg_tta.methods.{item['paradigm_slug']}.{item['slug']}",
    )


_PARADIGM_DATA = [
    {"slug": "input_level_transformation", "name": "Input-level Transformation"},
    {"slug": "feature_level_alignment", "name": "Feature-level Alignment"},
    {"slug": "output_level_regularization", "name": "Output-level Regularization"},
    {"slug": "prior_estimation", "name": "Prior Estimation"},
]

_METHOD_DATA = [
    {
        "slug": "dg_tta",
        "name": "DG-TTA",
        "source_dir": "output_level_regularization/DG-TTA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "modality": "MRI/CT",
        "status": "available",
        "summary": "Domain-generalization style test-time adaptation with consistency regularization and spatial/intensity augmentation utilities.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py"],
            "three_d": ["tta3dCT.py", "test_target_tta.py"],
        },
    },
    {
        "slug": "sattca",
        "name": "SaTTCA",
        "source_dir": "output_level_regularization/SaTTCA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "modality": "CT",
        "status": "available",
        "summary": "Scale-aware test-time click adaptation with click-mask generation, entropy/click losses, and 2D/3D entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["sattc.py", "tta2d.py"],
            "three_d": ["tta3dCT.py", "tta3dMRI.py"],
        },
    },
    {
        "slug": "grata",
        "name": "GraTa",
        "source_dir": "feature_level_alignment/GraTa",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "modality": "OCT/CT",
        "status": "available",
        "summary": "Gradient-based test-time adaptation optimizer with unified 2D OCT and 3D CT integration entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py", "test_target_tta.py", "GraTa-master/TTA.py"],
            "three_d": ["tta3dCT.py", "grata_3d.py", "grata_wrapper.py", "GraTa-master/TTA.py"],
        },
        "aliases": {"grata_3d": "three_d"},
    },
    {
        "slug": "testfit",
        "name": "TestFit",
        "source_dir": "feature_level_alignment/Testfit",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "modality": "CT/PATH",
        "status": "available",
        "summary": "Patch/window-level online adaptation using entropy minimization over sliding-window inference.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py"],
            "three_d": ["tta3dCT.py", "testfit.py"],
        },
    },
    {
        "slug": "tent",
        "name": "TENT",
        "source_dir": "output_level_regularization/tent",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "modality": "General Image",
        "status": "available",
        "summary": "Fully test-time entropy minimization with batch-normalization affine parameter updates.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tent2d.py", "tta2d.py"],
            "three_d": ["tent.py", "tta3d.py", "tta3dCT.py"],
        },
    },
    {
        "slug": "prosfda",
        "name": "ProSFDA",
        "source_dir": "prior_estimation/ProSFDA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "OCT/CT",
        "status": "available",
        "summary": "Prompt-learning source-free adaptation implementation with unified 2D and 3D entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": [
                "tta2d.py",
                "prosfda/training/run_training.py",
                "prosfda/inference/run_inference.py",
            ],
            "three_d": [
                "tta3dCT.py",
                "prosfda/training/run_training.py",
                "prosfda/inference/run_inference.py",
            ],
        },
        "aliases": {
            "prosfda_2d": "two_d",
            "prosfda_3d": "three_d",
        },
    },
    {
        "slug": "adami",
        "name": "AdaMI",
        "source_dir": "prior_estimation/AdaMI",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "MRI/CT",
        "status": "available",
        "summary": "Adversarial mutual-information based source-free adaptation with bundled 2D, 3D CT, and 3D BRATS legacy entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py"],
            "three_d": ["tta3d.py", "tta3dCT.py"],
        },
        "aliases": {
            "adami_2d": "two_d",
            "adami_3d": "three_d",
        },
    },
    {
        "slug": "pass",
        "name": "PASS",
        "source_dir": "prior_estimation/PASS",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "OCT/MRI/CT",
        "status": "available",
        "summary": "Prior-aware source-free adaptation with bundled 2D, 3D BRATS, and 3D CT legacy entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py"],
            "three_d": ["tta3d.py", "tta3dCT.py"],
        },
        "aliases": {
            "pass_2d": "two_d",
            "pass_3d": "three_d",
        },
    },
    {
        "slug": "exploring_tta",
        "name": "ExploringTTA",
        "source_dir": "prior_estimation/ExploringTTA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "US",
        "status": "available",
        "summary": "Experiment harness for TENT, entropy-KL, histogram matching, and filter-inspection adaptation variants.",
        "dimensions": ["three_d"],
        "entries_by_dimension": {
            "three_d": ["test_target_tta.py", "tta3dCT.py"],
        },
    },
    {
        "slug": "sfda_fsm",
        "name": "SFDA-FSM",
        "source_dir": "input_level_transformation/SFDA-FSM",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "modality": "Endoscope",
        "status": "available",
        "summary": "Source-free domain adaptation with Fourier style mining, domain inversion, CDD, and CADC components.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py", "tta2d_inf.py", "tools/train_adapt.py", "tools/test.py"],
            "three_d": ["tta3d.py", "tta3dCT.py"],
        },
    },
    {
        "slug": "dl_tta",
        "name": "DL-TTA",
        "source_dir": "input_level_transformation/DL-TTA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "modality": "PATH/MRI",
        "status": "available",
        "summary": "Memory-guided source-free adaptation with bundled 2D and 3D BRATS legacy entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py"],
            "three_d": ["tta3d.py"],
        },
        "aliases": {
            "dltta_2d": "two_d",
            "dltta_3d": "three_d",
        },
    },
    {
        "slug": "stdr",
        "name": "STDR",
        "source_dir": "input_level_transformation/STDR",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "modality": "MRI",
        "status": "available",
        "summary": "Sample-selection driven 3D BRATS adaptation workflow with source-feature saving, anchor clustering, active-sample selection, and finetuning entrypoints.",
        "dimensions": ["three_d"],
        "entries_by_dimension": {
            "three_d": [
                "tta3d.py",
                "save_source.py",
                "cluster_anchors.py",
                "select_active_samples.py",
            ],
        },
        "aliases": {
            "stdr_3d": "three_d",
        },
    },
    {
        "slug": "rsa",
        "name": "RSA",
        "source_dir": "input_level_transformation/RSA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "modality": "MRI",
        "status": "available",
        "summary": "Reliable source approximation workflow with translation, reliability-based sample selection, and segmentation finetuning stages.",
        "dimensions": ["three_d"],
        "entries_by_dimension": {
            "three_d": ["translate.py", "select.py", "tta3d.py"],
        },
        "aliases": {
            "rsa_3d": "three_d",
        },
    },
    {
        "slug": "upl_sfda",
        "name": "UPL-SFDA",
        "source_dir": "input_level_transformation/UPL-SFDA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "modality": "CMR/MRI",
        "status": "available",
        "summary": "Uncertainty-aware pseudo-label guided source-free adaptation with bundled 2D and 3D legacy MRI segmentation entrypoints.",
        "dimensions": ["two_d", "three_d"],
        "entries_by_dimension": {
            "two_d": ["tta2d.py"],
            "three_d": ["tta3d.py", "tta3dCT.py"],
        },
    },
]

TABLE_METHODS = [
    {
        "name": "AIF-SFDA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "missing",
    },
    {
        "name": "STDR",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "MRI",
        "original_dimension": "3D",
        "status": "available",
    },
    {
        "name": "RSA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "MRI",
        "original_dimension": "3D",
        "status": "available",
    },
    {
        "name": "SFDA-FSM",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "Endoscope",
        "original_dimension": "2D",
        "status": "available",
    },
    {
        "name": "DL-TTA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "PATH",
        "original_dimension": "2D",
        "status": "available",
    },
    {
        "name": "GraTa",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "available",
    },
    {
        "name": "UDA-MIMA",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "MRI/CT",
        "original_dimension": "3D",
        "status": "missing",
    },
    {
        "name": "DeTTA",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "CT",
        "original_dimension": "2D",
        "status": "missing",
    },
    {
        "name": "TestFit",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "CT/PATH",
        "original_dimension": "General",
        "status": "available",
    },
    {
        "name": "DANN",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "MRI",
        "original_dimension": "3D",
        "status": "missing",
    },
    {
        "name": "SmaRT",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "MRI",
        "original_dimension": "3D",
        "status": "missing",
    },
    {
        "name": "DG-TTA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "MRI/CT",
        "original_dimension": "3D",
        "status": "available",
    },
    {
        "name": "SaTTCA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "CT",
        "original_dimension": "3D",
        "status": "available",
    },
    {
        "name": "UPL-SFDA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "CMR/MRI",
        "original_dimension": "General",
        "status": "available",
    },
    {
        "name": "TENT",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "General Image",
        "original_dimension": "General",
        "status": "available",
    },
    {
        "name": "ProSFDA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "available",
    },
    {
        "name": "ExploringTTA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "US",
        "original_dimension": "3D",
        "status": "available",
    },
    {
        "name": "PASS",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "available",
    },
    {
        "name": "VPTTA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "missing",
    },
    {
        "name": "AdaMI",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "MRI/CT",
        "original_dimension": "3D",
        "status": "available",
    },
]

PARADIGMS = tuple(ParadigmSpec(**item) for item in _PARADIGM_DATA)
METHODS = tuple(_build_spec(item) for item in _METHOD_DATA)
METHODS_BY_SLUG = {method.slug: method for method in METHODS}

METHOD_ALIASES = {}
for method in METHODS:
    for alias, dimension in method.aliases.items():
        METHOD_ALIASES[alias] = (method.slug, dimension)


def dimension_label(dimension: str) -> str:
    return DIMENSION_LABELS[dimension]


def find_paradigm(key: str) -> ParadigmSpec:
    normalized = _normalize(key)
    for paradigm in PARADIGMS:
        if normalized in {paradigm.slug, _normalize(paradigm.name)}:
            return paradigm
    raise KeyError(key)


def resolve_method(key: str) -> tuple[MethodSpec, str | None]:
    normalized = _normalize(key)
    if normalized in METHOD_ALIASES:
        slug, dimension = METHOD_ALIASES[normalized]
        return METHODS_BY_SLUG[slug], dimension
    for method in METHODS:
        names = {method.slug, _normalize(method.name)}
        if normalized in names:
            return method, None
    raise KeyError(key)


def find_method(key: str) -> MethodSpec:
    method, _ = resolve_method(key)
    return method


def method_dimensions(method: MethodSpec) -> tuple[str, ...]:
    return method.dimensions


def methods_by_paradigm(paradigm: str | None = None, dimension: str | None = None):
    dimension = normalize_dimension(dimension)
    if paradigm is None:
        grouped = {p.slug: tuple(m for m in METHODS if m.paradigm_slug == p.slug) for p in PARADIGMS}
        if dimension is None:
            return grouped
        return {
            key: tuple(method for method in methods if dimension in method.dimensions)
            for key, methods in grouped.items()
        }
    spec = find_paradigm(paradigm)
    methods = tuple(method for method in METHODS if method.paradigm_slug == spec.slug)
    if dimension is None:
        return methods
    return tuple(method for method in methods if dimension in method.dimensions)


def available_methods():
    return tuple(method for method in METHODS if method.status == "available")


def method_entries(method: MethodSpec, dimension: str | None = None):
    if dimension is None:
        return {item: method.entries_by_dimension[item] for item in method.dimensions}
    normalized = normalize_dimension(dimension)
    if normalized not in method.entries_by_dimension:
        raise KeyError(dimension)
    return method.entries_by_dimension[normalized]


def resolve_entry(method: MethodSpec, entry: str, forced_dimension: str | None = None) -> tuple[str, str]:
    entry = entry.strip().replace("\\", "/")
    if not entry:
        raise FileNotFoundError(entry)
    if "/" in entry:
        maybe_dimension, rel_entry = entry.split("/", 1)
        try:
            normalized_dimension = normalize_dimension(maybe_dimension)
        except KeyError:
            normalized_dimension = None
        if normalized_dimension is not None:
            if normalized_dimension not in method.entries_by_dimension:
                raise KeyError(f"{method.slug} has no {maybe_dimension} entrypoint set")
            if rel_entry not in method.entries_by_dimension[normalized_dimension]:
                raise FileNotFoundError(entry)
            return normalized_dimension, rel_entry
    search_dimensions = (forced_dimension,) if forced_dimension else method.dimensions
    matches = [
        (dimension, entry)
        for dimension in search_dimensions
        if entry in method.entries_by_dimension.get(dimension, ())
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous entry '{entry}' for {method.slug}; use two_d/{entry} or three_d/{entry}"
        )
    raise FileNotFoundError(entry)


def table_by_paradigm(paradigm: str | None = None):
    if paradigm is None:
        return {p.slug: tuple(row for row in TABLE_METHODS if row["paradigm_slug"] == p.slug) for p in PARADIGMS}
    spec = find_paradigm(paradigm)
    return tuple(row for row in TABLE_METHODS if row["paradigm_slug"] == spec.slug)
