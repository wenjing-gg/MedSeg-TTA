# Method Index

This index follows the representative MedSeg-TTA method table and is grouped by the four adaptation paradigms in the table. RSA is intentionally excluded from code processing in this release.

## Input-level Transformation

| Method | Original modality | Original dimension | Local status | Package path |
| --- | --- | --- | --- | --- |
| AIF-SFDA | OCT | 2D | not found locally | - |
| STDR | MRI | 2D | not found locally | - |
| RSA | MRI | 2D | skipped by request | - |
| SFDA-FSM | Endoscope | 2D | included | medseg_tta.methods.input_level_transformation.sfda_fsm |
| DL-TTA | PATH | 2D | not found locally | - |

## Feature-level Alignment

| Method | Original modality | Original dimension | Local status | Package path |
| --- | --- | --- | --- | --- |
| GraTa | OCT | 2D | included | medseg_tta.methods.feature_level_alignment.grata, medseg_tta.methods.feature_level_alignment.grata_3d |
| UDA-MIMA | MRI/CT | 3D | not found locally | - |
| DeTTA | CT | 2D | not found locally | - |
| TestFit | CT/PATH | General | included | medseg_tta.methods.feature_level_alignment.testfit |
| DANN | MRI | 3D | not found locally | - |

## Output-level Regularization

| Method | Original modality | Original dimension | Local status | Package path |
| --- | --- | --- | --- | --- |
| SmaRT | MRI | 3D | not found locally | - |
| DG-TTA | MRI/CT | 3D | included | medseg_tta.methods.output_level_regularization.dg_tta |
| SaTTCA | CT | 3D | included | medseg_tta.methods.output_level_regularization.sattca |
| UPL-SFDA | CMR/MRI | General | not found locally | - |
| TENT | General Image | General | included | medseg_tta.methods.output_level_regularization.tent |

## Prior Estimation

| Method | Original modality | Original dimension | Local status | Package path |
| --- | --- | --- | --- | --- |
| ProSFDA | OCT | 2D | included | medseg_tta.methods.prior_estimation.prosfda_2d, medseg_tta.methods.prior_estimation.prosfda_3d |
| ExploringTTA | US | 3D | included | medseg_tta.methods.prior_estimation.exploring_tta |
| PASS | OCT | 2D | not found locally | - |
| VPTTA | OCT | 2D | not found locally | - |
| AdaMI | MRI/CT | 3D | not found locally | - |

## Entrypoints

### Input-level Transformation

- SFDA-FSM: tta2d.py, tta2d_inf.py, tta3d.py, tta3dCT.py, tools/train_adapt.py, tools/test.py

### Feature-level Alignment

- GraTa: tta2d.py, test_target_tta.py, GraTa-master/TTA.py
- GraTa-3D: tta3dCT.py, grata_3d.py, grata_wrapper.py, GraTa-master/TTA.py
- TestFit: tta2d.py, tta3dCT.py, testfit.py

### Output-level Regularization

- DG-TTA: tta2d.py, tta3dCT.py, test_target_tta.py
- SaTTCA: sattc.py, tta2d.py, tta3dCT.py, tta3dMRI.py
- TENT: tent.py, tent2d.py, tta2d.py, tta3d.py, tta3dCT.py

### Prior Estimation

- ProSFDA-2D: tta2d.py, prosfda/training/run_training.py, prosfda/inference/run_inference.py
- ProSFDA-3D: tta3dCT.py, prosfda/training/run_training.py, prosfda/inference/run_inference.py
- ExploringTTA: test_target_tta.py, tta3dCT.py
