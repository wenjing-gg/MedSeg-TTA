# Method Index

This index follows the representative MedSeg-TTA method table and groups local code by adaptation paradigm. RSA remains excluded from local code processing.

## Input-level Transformation

| Method | Original modality | Original dimension | Local status | Local dimensions | Package path |
| --- | --- | --- | --- | --- | --- |
| AIF-SFDA | OCT | 2D | not found locally | - | - |
| STDR | MRI | 2D | not found locally | - | - |
| RSA | MRI | 2D | skipped by request | - | - |
| SFDA-FSM | Endoscope | 2D | included | 2D, 3D | `medseg_tta.methods.input_level_transformation.sfda_fsm` |
| DL-TTA | PATH | 2D | not found locally | - | - |

## Feature-level Alignment

| Method | Original modality | Original dimension | Local status | Local dimensions | Package path |
| --- | --- | --- | --- | --- | --- |
| GraTa | OCT | 2D | included | 2D, 3D | `medseg_tta.methods.feature_level_alignment.grata` |
| UDA-MIMA | MRI/CT | 3D | not found locally | - | - |
| DeTTA | CT | 2D | not found locally | - | - |
| TestFit | CT/PATH | General | included | 2D, 3D | `medseg_tta.methods.feature_level_alignment.testfit` |
| DANN | MRI | 3D | not found locally | - | - |

## Output-level Regularization

| Method | Original modality | Original dimension | Local status | Local dimensions | Package path |
| --- | --- | --- | --- | --- | --- |
| SmaRT | MRI | 3D | not found locally | - | - |
| DG-TTA | MRI/CT | 3D | included | 2D, 3D | `medseg_tta.methods.output_level_regularization.dg_tta` |
| SaTTCA | CT | 3D | included | 2D, 3D | `medseg_tta.methods.output_level_regularization.sattca` |
| UPL-SFDA | CMR/MRI | General | not found locally | - | - |
| TENT | General Image | General | included | 2D, 3D | `medseg_tta.methods.output_level_regularization.tent` |

## Prior Estimation

| Method | Original modality | Original dimension | Local status | Local dimensions | Package path |
| --- | --- | --- | --- | --- | --- |
| ProSFDA | OCT | 2D | included | 2D, 3D | `medseg_tta.methods.prior_estimation.prosfda` |
| ExploringTTA | US | 3D | included | 3D | `medseg_tta.methods.prior_estimation.exploring_tta` |
| PASS | OCT | 2D | not found locally | - | - |
| VPTTA | OCT | 2D | not found locally | - | - |
| AdaMI | MRI/CT | 3D | not found locally | - | - |

## Entrypoints

### Input-level Transformation

- SFDA-FSM
  - `two_d`: `tta2d.py`, `tta2d_inf.py`, `tools/train_adapt.py`, `tools/test.py`
  - `three_d`: `tta3d.py`, `tta3dCT.py`

### Feature-level Alignment

- GraTa
  - `two_d`: `tta2d.py`, `test_target_tta.py`, `GraTa-master/TTA.py`
  - `three_d`: `tta3dCT.py`, `grata_3d.py`, `grata_wrapper.py`, `GraTa-master/TTA.py`
- TestFit
  - `two_d`: `tta2d.py`
  - `three_d`: `tta3dCT.py`, `testfit.py`

### Output-level Regularization

- DG-TTA
  - `two_d`: `tta2d.py`
  - `three_d`: `tta3dCT.py`, `test_target_tta.py`
- SaTTCA
  - `two_d`: `sattc.py`, `tta2d.py`
  - `three_d`: `tta3dCT.py`, `tta3dMRI.py`
- TENT
  - `two_d`: `tent2d.py`, `tta2d.py`
  - `three_d`: `tent.py`, `tta3d.py`, `tta3dCT.py`

### Prior Estimation

- ProSFDA
  - `two_d`: `tta2d.py`, `prosfda/training/run_training.py`, `prosfda/inference/run_inference.py`
  - `three_d`: `tta3dCT.py`, `prosfda/training/run_training.py`, `prosfda/inference/run_inference.py`
- ExploringTTA
  - `three_d`: `test_target_tta.py`, `tta3dCT.py`
