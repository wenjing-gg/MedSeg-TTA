# Code Analysis

The processed implementation is organized as a unified package with sanitized legacy method code under `medseg_tta/methods/<method>/legacy` and common model/backbone copies under `medseg_tta/models`.

## Included Method Families

### DG-TTA
- Package: `medseg_tta.methods.dg_tta`
- Source directory: `DG-TTA`
- Python files copied: 22
- Main entrypoints: `tta2d.py`, `tta3dCT.py`, `test_target_tta.py`
- Role: Domain-generalization style test-time adaptation with consistency regularization and spatial/intensity augmentation utilities.

### SaTTCA
- Package: `medseg_tta.methods.sattca`
- Source directory: `SaTTCA`
- Python files copied: 25
- Main entrypoints: `sattc.py`, `tta2d.py`, `tta3dCT.py`, `tta3dMRI.py`
- Role: Scale-aware test-time click adaptation with click-mask generation, entropy/click losses, and 2D/3D entrypoints.

### GraTa
- Package: `medseg_tta.methods.grata`
- Source directory: `GraTa`
- Python files copied: 31
- Main entrypoints: `tta2d.py`, `test_target_tta.py`, `GraTa-master/TTA.py`
- Role: Gradient-based test-time adaptation optimizer and 2D segmentation integration.

### GraTa-3D
- Package: `medseg_tta.methods.grata_3d`
- Source directory: `GraTa-3d`
- Python files copied: 32
- Main entrypoints: `tta3dCT.py`, `grata_3d.py`, `grata_wrapper.py`, `GraTa-master/TTA.py`
- Role: 3D adaptation wrapper around the GraTa optimizer for CT segmentation experiments.

### TestFit
- Package: `medseg_tta.methods.testfit`
- Source directory: `Testfit`
- Python files copied: 13
- Main entrypoints: `tta2d.py`, `tta3dCT.py`, `testfit.py`
- Role: Patch/window-level online adaptation using entropy minimization over sliding-window inference.

### TENT
- Package: `medseg_tta.methods.tent`
- Source directory: `tent`
- Python files copied: 34
- Main entrypoints: `tent.py`, `tent2d.py`, `tta2d.py`, `tta3d.py`, `tta3dCT.py`
- Role: Fully test-time entropy minimization with batch-normalization affine parameter updates.

### ProSFDA-2D
- Package: `medseg_tta.methods.prosfda_2d`
- Source directory: `ProSFDA2D`
- Python files copied: 43
- Main entrypoints: `tta2d.py`, `prosfda/training/run_training.py`, `prosfda/inference/run_inference.py`
- Role: Prompt-learning source-free adaptation implementation with PLS/FAS components for 2D segmentation.

### ProSFDA-3D
- Package: `medseg_tta.methods.prosfda_3d`
- Source directory: `ProSFDA3D`
- Python files copied: 32
- Main entrypoints: `tta3dCT.py`, `prosfda/training/run_training.py`, `prosfda/inference/run_inference.py`
- Role: Local 3D extension of ProSFDA with prompt-aware UNet variants and CT TTA trainer.

### ExploringTTA
- Package: `medseg_tta.methods.exploring_tta`
- Source directory: `ExploringTTA`
- Python files copied: 15
- Main entrypoints: `test_target_tta.py`, `tta3dCT.py`
- Role: Experiment harness for TENT, entropy-KL, histogram matching, and filter-inspection adaptation variants.

### SFDA-FSM
- Package: `medseg_tta.methods.sfda_fsm`
- Source directory: `SFDA-FSM`
- Python files copied: 28
- Main entrypoints: `tta2d.py`, `tta2d_inf.py`, `tta3d.py`, `tta3dCT.py`, `tools/train_adapt.py`, `tools/test.py`
- Role: Source-free domain adaptation with Fourier style mining, domain inversion, CDD, and CADC components.

## Shared Structure

- `medseg_tta.registry` records method metadata and local availability.
- `medseg_tta.cli` exposes method listing, method details, and legacy entrypoint forwarding.
- `medseg_tta.common.legacy` keeps old wrapper scripts lightweight and defers optional dependency imports until real execution.
- `medseg_tta.models` contains sanitized common UNet and nnUNet backbones copied from the local TTA baseline implementation.

## Exclusions

- RSA is not copied, cleaned, validated, or wrapped in this release.
- Checkpoints, generated results, cached bytecode, local nested `.git` directories, and large binary assets are excluded.
