from dataclasses import dataclass


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
    dimension: str
    status: str
    summary: str
    entries: tuple[str, ...]
    package: str


def _build_spec(item):
    return MethodSpec(
        slug=item['slug'],
        name=item['name'],
        source_dir=item['source_dir'],
        paradigm_slug=item['paradigm_slug'],
        paradigm=item['paradigm'],
        modality=item['modality'],
        dimension=item['dimension'],
        status=item['status'],
        summary=item['summary'],
        entries=tuple(item['entries']),
        package=f"medseg_tta.methods.{item['paradigm_slug']}.{item['slug']}",
    )


def _normalize(value):
    return value.lower().replace('-', '_').replace(' ', '_')


_PARADIGM_DATA = [
    {
        "slug": "input_level_transformation",
        "name": "Input-level Transformation"
    },
    {
        "slug": "feature_level_alignment",
        "name": "Feature-level Alignment"
    },
    {
        "slug": "output_level_regularization",
        "name": "Output-level Regularization"
    },
    {
        "slug": "prior_estimation",
        "name": "Prior Estimation"
    }
]
_METHOD_DATA = [
    {
        "slug": "dg_tta",
        "name": "DG-TTA",
        "source_dir": "DG-TTA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "modality": "MRI/CT",
        "dimension": "3D",
        "status": "available",
        "summary": "Domain-generalization style test-time adaptation with consistency regularization and spatial/intensity augmentation utilities.",
        "entries": [
            "tta2d.py",
            "tta3dCT.py",
            "test_target_tta.py"
        ]
    },
    {
        "slug": "sattca",
        "name": "SaTTCA",
        "source_dir": "SaTTCA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "modality": "CT",
        "dimension": "3D",
        "status": "available",
        "summary": "Scale-aware test-time click adaptation with click-mask generation, entropy/click losses, and 2D/3D entrypoints.",
        "entries": [
            "sattc.py",
            "tta2d.py",
            "tta3dCT.py",
            "tta3dMRI.py"
        ]
    },
    {
        "slug": "grata",
        "name": "GraTa",
        "source_dir": "GraTa",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "modality": "OCT",
        "dimension": "2D",
        "status": "available",
        "summary": "Gradient-based test-time adaptation optimizer and 2D segmentation integration.",
        "entries": [
            "tta2d.py",
            "test_target_tta.py",
            "GraTa-master/TTA.py"
        ]
    },
    {
        "slug": "grata_3d",
        "name": "GraTa-3D",
        "source_dir": "GraTa-3d",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "modality": "CT",
        "dimension": "3D",
        "status": "available",
        "summary": "3D adaptation wrapper around the GraTa optimizer for CT segmentation experiments.",
        "entries": [
            "tta3dCT.py",
            "grata_3d.py",
            "grata_wrapper.py",
            "GraTa-master/TTA.py"
        ]
    },
    {
        "slug": "testfit",
        "name": "TestFit",
        "source_dir": "Testfit",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "modality": "CT/PATH",
        "dimension": "General",
        "status": "available",
        "summary": "Patch/window-level online adaptation using entropy minimization over sliding-window inference.",
        "entries": [
            "tta2d.py",
            "tta3dCT.py",
            "testfit.py"
        ]
    },
    {
        "slug": "tent",
        "name": "TENT",
        "source_dir": "tent",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "modality": "General Image",
        "dimension": "General",
        "status": "available",
        "summary": "Fully test-time entropy minimization with batch-normalization affine parameter updates.",
        "entries": [
            "tent.py",
            "tent2d.py",
            "tta2d.py",
            "tta3d.py",
            "tta3dCT.py"
        ]
    },
    {
        "slug": "prosfda_2d",
        "name": "ProSFDA-2D",
        "source_dir": "ProSFDA2D",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "OCT",
        "dimension": "2D",
        "status": "available",
        "summary": "Prompt-learning source-free adaptation implementation with PLS/FAS components for 2D segmentation.",
        "entries": [
            "tta2d.py",
            "prosfda/training/run_training.py",
            "prosfda/inference/run_inference.py"
        ]
    },
    {
        "slug": "prosfda_3d",
        "name": "ProSFDA-3D",
        "source_dir": "ProSFDA3D",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "CT",
        "dimension": "3D",
        "status": "available",
        "summary": "Local 3D extension of ProSFDA with prompt-aware UNet variants and CT TTA trainer.",
        "entries": [
            "tta3dCT.py",
            "prosfda/training/run_training.py",
            "prosfda/inference/run_inference.py"
        ]
    },
    {
        "slug": "exploring_tta",
        "name": "ExploringTTA",
        "source_dir": "ExploringTTA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "modality": "US",
        "dimension": "3D",
        "status": "available",
        "summary": "Experiment harness for TENT, entropy-KL, histogram matching, and filter-inspection adaptation variants.",
        "entries": [
            "test_target_tta.py",
            "tta3dCT.py"
        ]
    },
    {
        "slug": "sfda_fsm",
        "name": "SFDA-FSM",
        "source_dir": "SFDA-FSM",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "modality": "Endoscope",
        "dimension": "2D",
        "status": "available",
        "summary": "Source-free domain adaptation with Fourier style mining, domain inversion, CDD, and CADC components.",
        "entries": [
            "tta2d.py",
            "tta2d_inf.py",
            "tta3d.py",
            "tta3dCT.py",
            "tools/train_adapt.py",
            "tools/test.py"
        ]
    }
]
TABLE_METHODS = [
    {
        "name": "AIF-SFDA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "missing"
    },
    {
        "name": "STDR",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "MRI",
        "original_dimension": "2D",
        "status": "missing"
    },
    {
        "name": "RSA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "MRI",
        "original_dimension": "2D",
        "status": "skipped"
    },
    {
        "name": "SFDA-FSM",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "Endoscope",
        "original_dimension": "2D",
        "status": "available"
    },
    {
        "name": "DL-TTA",
        "paradigm_slug": "input_level_transformation",
        "paradigm": "Input-level Transformation",
        "original_modality": "PATH",
        "original_dimension": "2D",
        "status": "missing"
    },
    {
        "name": "GraTa",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "available"
    },
    {
        "name": "UDA-MIMA",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "MRI/CT",
        "original_dimension": "3D",
        "status": "missing"
    },
    {
        "name": "DeTTA",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "CT",
        "original_dimension": "2D",
        "status": "missing"
    },
    {
        "name": "TestFit",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "CT/PATH",
        "original_dimension": "General",
        "status": "available"
    },
    {
        "name": "DANN",
        "paradigm_slug": "feature_level_alignment",
        "paradigm": "Feature-level Alignment",
        "original_modality": "MRI",
        "original_dimension": "3D",
        "status": "missing"
    },
    {
        "name": "SmaRT",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "MRI",
        "original_dimension": "3D",
        "status": "missing"
    },
    {
        "name": "DG-TTA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "MRI/CT",
        "original_dimension": "3D",
        "status": "available"
    },
    {
        "name": "SaTTCA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "CT",
        "original_dimension": "3D",
        "status": "available"
    },
    {
        "name": "UPL-SFDA",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "CMR/MRI",
        "original_dimension": "General",
        "status": "missing"
    },
    {
        "name": "TENT",
        "paradigm_slug": "output_level_regularization",
        "paradigm": "Output-level Regularization",
        "original_modality": "General Image",
        "original_dimension": "General",
        "status": "available"
    },
    {
        "name": "ProSFDA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "available"
    },
    {
        "name": "ExploringTTA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "US",
        "original_dimension": "3D",
        "status": "available"
    },
    {
        "name": "PASS",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "missing"
    },
    {
        "name": "VPTTA",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "OCT",
        "original_dimension": "2D",
        "status": "missing"
    },
    {
        "name": "AdaMI",
        "paradigm_slug": "prior_estimation",
        "paradigm": "Prior Estimation",
        "original_modality": "MRI/CT",
        "original_dimension": "3D",
        "status": "missing"
    }
]
PARADIGMS = tuple(ParadigmSpec(**item) for item in _PARADIGM_DATA)
METHODS = tuple(_build_spec(item) for item in _METHOD_DATA)


def find_paradigm(key):
    normalized = _normalize(key)
    for paradigm in PARADIGMS:
        if normalized in {paradigm.slug, _normalize(paradigm.name)}:
            return paradigm
    raise KeyError(key)


def find_method(key):
    normalized = _normalize(key)
    for method in METHODS:
        names = {method.slug, _normalize(method.name)}
        if normalized in names or any(name.startswith(f'{normalized}_') for name in names):
            return method
    raise KeyError(key)


def available_methods():
    return tuple(method for method in METHODS if method.status == 'available')


def methods_by_paradigm(paradigm=None):
    if paradigm is None:
        return {p.slug: tuple(m for m in METHODS if m.paradigm_slug == p.slug) for p in PARADIGMS}
    spec = find_paradigm(paradigm)
    return tuple(method for method in METHODS if method.paradigm_slug == spec.slug)


def table_by_paradigm(paradigm=None):
    if paradigm is None:
        return {p.slug: tuple(row for row in TABLE_METHODS if row['paradigm_slug'] == p.slug) for p in PARADIGMS}
    spec = find_paradigm(paradigm)
    return tuple(row for row in TABLE_METHODS if row['paradigm_slug'] == spec.slug)
