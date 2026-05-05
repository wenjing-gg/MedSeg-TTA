# Code Analysis

The processed implementation is organized by adaptation paradigm first, then by method. Sanitized legacy method code is stored under `medseg_tta/methods/<paradigm>/<method>/legacy`, while shared backbone copies remain under `medseg_tta/models`.

## Paradigm Organization

### Input-level Transformation

- Method: SFDA-FSM
- Package: `medseg_tta.methods.input_level_transformation.sfda_fsm`
- Legacy root: `medseg_tta/methods/input_level_transformation/sfda_fsm/legacy`
- Source directory: `SFDA-FSM`
- Main entrypoints: `tta2d.py`, `tta2d_inf.py`, `tta3d.py`, `tta3dCT.py`, `tools/train_adapt.py`, `tools/test.py`
- Role: Source-free domain adaptation with Fourier style mining, domain inversion, CDD, and CADC components.

### Feature-level Alignment

- Method: GraTa
- Package: `medseg_tta.methods.feature_level_alignment.grata`
- Legacy root: `medseg_tta/methods/feature_level_alignment/grata/legacy`
- Source directory: `GraTa`
- Main entrypoints: `tta2d.py`, `test_target_tta.py`, `GraTa-master/TTA.py`
- Role: Gradient-based test-time adaptation optimizer and 2D segmentation integration.

- Method: GraTa-3D
- Package: `medseg_tta.methods.feature_level_alignment.grata_3d`
- Legacy root: `medseg_tta/methods/feature_level_alignment/grata_3d/legacy`
- Source directory: `GraTa-3d`
- Main entrypoints: `tta3dCT.py`, `grata_3d.py`, `grata_wrapper.py`, `GraTa-master/TTA.py`
- Role: 3D adaptation wrapper around the GraTa optimizer for CT segmentation experiments.

- Method: TestFit
- Package: `medseg_tta.methods.feature_level_alignment.testfit`
- Legacy root: `medseg_tta/methods/feature_level_alignment/testfit/legacy`
- Source directory: `Testfit`
- Main entrypoints: `tta2d.py`, `tta3dCT.py`, `testfit.py`
- Role: Patch/window-level online adaptation using entropy minimization over sliding-window inference.

### Output-level Regularization

- Method: DG-TTA
- Package: `medseg_tta.methods.output_level_regularization.dg_tta`
- Legacy root: `medseg_tta/methods/output_level_regularization/dg_tta/legacy`
- Source directory: `DG-TTA`
- Main entrypoints: `tta2d.py`, `tta3dCT.py`, `test_target_tta.py`
- Role: Domain-generalization style test-time adaptation with consistency regularization and spatial/intensity augmentation utilities.

- Method: SaTTCA
- Package: `medseg_tta.methods.output_level_regularization.sattca`
- Legacy root: `medseg_tta/methods/output_level_regularization/sattca/legacy`
- Source directory: `SaTTCA`
- Main entrypoints: `sattc.py`, `tta2d.py`, `tta3dCT.py`, `tta3dMRI.py`
- Role: Scale-aware test-time click adaptation with click-mask generation, entropy/click losses, and 2D/3D entrypoints.

- Method: TENT
- Package: `medseg_tta.methods.output_level_regularization.tent`
- Legacy root: `medseg_tta/methods/output_level_regularization/tent/legacy`
- Source directory: `tent`
- Main entrypoints: `tent.py`, `tent2d.py`, `tta2d.py`, `tta3d.py`, `tta3dCT.py`
- Role: Fully test-time entropy minimization with batch-normalization affine parameter updates.

### Prior Estimation

- Method: ProSFDA-2D
- Package: `medseg_tta.methods.prior_estimation.prosfda_2d`
- Legacy root: `medseg_tta/methods/prior_estimation/prosfda_2d/legacy`
- Source directory: `ProSFDA2D`
- Main entrypoints: `tta2d.py`, `prosfda/training/run_training.py`, `prosfda/inference/run_inference.py`
- Role: Prompt-learning source-free adaptation implementation with PLS/FAS components for 2D segmentation.

- Method: ProSFDA-3D
- Package: `medseg_tta.methods.prior_estimation.prosfda_3d`
- Legacy root: `medseg_tta/methods/prior_estimation/prosfda_3d/legacy`
- Source directory: `ProSFDA3D`
- Main entrypoints: `tta3dCT.py`, `prosfda/training/run_training.py`, `prosfda/inference/run_inference.py`
- Role: Local 3D extension of ProSFDA with prompt-aware UNet variants and CT TTA trainer.

- Method: ExploringTTA
- Package: `medseg_tta.methods.prior_estimation.exploring_tta`
- Legacy root: `medseg_tta/methods/prior_estimation/exploring_tta/legacy`
- Source directory: `ExploringTTA`
- Main entrypoints: `test_target_tta.py`, `tta3dCT.py`
- Role: Experiment harness for TENT, entropy-KL, histogram matching, and filter-inspection adaptation variants.

## Shared Structure

- `medseg_tta.registry` records paradigm metadata, method metadata, local availability, and package paths.
- `medseg_tta.cli` exposes grouped method listing, table listing, method details, paradigm listing, structure validation, and legacy entrypoint forwarding.
- `medseg_tta.validate` checks registry-to-filesystem consistency, package importability, legacy roots, and registered entrypoint files.
- `medseg_tta.common.legacy` keeps old wrapper scripts lightweight and defers optional dependency imports until real execution.
- `medseg_tta.models` contains sanitized common UNet and nnUNet backbones copied from the local TTA baseline implementation.

## Exclusions

- RSA is not copied, cleaned, validated, or wrapped in this release.
- Checkpoints, generated results, cached bytecode, local nested `.git` directories, and large binary assets are excluded.
