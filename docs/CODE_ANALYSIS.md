# Code Analysis

The processed implementation is organized by adaptation paradigm first, then by canonical method name. Dimension handling now lives inside each method root.

## Canonical Layout

All included methods follow this shape:

```text
medseg_tta/methods/<paradigm>/<method>/
  __init__.py
  two_d/            # optional
    __init__.py
    legacy/
  three_d/          # optional
    __init__.py
    legacy/
  common/           # optional, only when shared support code is needed
    __init__.py
    legacy/
```

Top-level wrapper directories mirror the same convention:

```text
<MethodName>/
  two_d/
  three_d/
```

## Method Summary

### Input-level Transformation

- SFDA-FSM
  - Package: `medseg_tta.methods.input_level_transformation.sfda_fsm`
  - Source directory: `SFDA-FSM`
  - Dimensions: `two_d`, `three_d`
  - Shared support code: `common/legacy`

### Feature-level Alignment

- GraTa
  - Package: `medseg_tta.methods.feature_level_alignment.grata`
  - Source directory: `GraTa`
  - Dimensions: `two_d`, `three_d`
  - Notes: previous `grata` and `grata_3d` packages are unified; `grata_3d` remains as an import alias stub.

- TestFit
  - Package: `medseg_tta.methods.feature_level_alignment.testfit`
  - Source directory: `Testfit`
  - Dimensions: `two_d`, `three_d`
  - Shared support code: `common/legacy`

### Output-level Regularization

- DG-TTA
  - Package: `medseg_tta.methods.output_level_regularization.dg_tta`
  - Source directory: `DG-TTA`
  - Dimensions: `two_d`, `three_d`
  - Shared support code: `common/legacy`

- SaTTCA
  - Package: `medseg_tta.methods.output_level_regularization.sattca`
  - Source directory: `SaTTCA`
  - Dimensions: `two_d`, `three_d`
  - Shared support code: `common/legacy`

- TENT
  - Package: `medseg_tta.methods.output_level_regularization.tent`
  - Source directory: `tent`
  - Dimensions: `two_d`, `three_d`
  - Shared support code: `common/legacy`

### Prior Estimation

- ProSFDA
  - Package: `medseg_tta.methods.prior_estimation.prosfda`
  - Source directory: `ProSFDA`
  - Dimensions: `two_d`, `three_d`
  - Notes: previous `prosfda_2d` and `prosfda_3d` packages are unified; both old package names remain as import alias stubs.

- ExploringTTA
  - Package: `medseg_tta.methods.prior_estimation.exploring_tta`
  - Source directory: `ExploringTTA`
  - Dimensions: `three_d`

## Shared Structure

- `medseg_tta.registry` records canonical method metadata, available dimensions, per-dimension entrypoints, and legacy aliases.
- `medseg_tta.cli` exposes grouped method listing, dimension filtering, method details, structure validation, and legacy entrypoint forwarding.
- `medseg_tta.validate` checks registry-to-filesystem consistency, package importability, dimension package importability, and registered entrypoint resolution.
- `medseg_tta.common.legacy` keeps top-level wrappers lightweight and resolves dimension-aware legacy roots at runtime.
- `medseg_tta.models` remains the shared backbone area; this refactor does not reorganize it beyond compatibility needs.

## Exclusions

- RSA is not copied, cleaned, validated, or wrapped in this release.
- Checkpoints, generated results, cached bytecode, local nested `.git` directories, and large binary assets are excluded.
