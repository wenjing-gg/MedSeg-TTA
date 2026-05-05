# Method Index

This index follows the representative MedSeg-TTA method table. RSA is intentionally excluded from code processing in this release.

| Method | Paradigm | Original modality | Original dimension | Local status | Package path |
| --- | --- | --- | --- | --- | --- |
| AIF-SFDA | Input-level Transformation | OCT | 2D | not found locally | - |
| STDR | Input-level Transformation | MRI | 2D | not found locally | - |
| RSA | Input-level Transformation | MRI | 2D | skipped by request | - |
| SFDA-FSM | Input-level Transformation | Endoscope | 2D | included | medseg_tta.methods.sfda_fsm |
| DL-TTA | Input-level Transformation | PATH | 2D | not found locally | - |
| GraTa | Feature-level Alignment | OCT | 2D | included | medseg_tta.methods.grata, medseg_tta.methods.grata_3d |
| UDA-MIMA | Feature-level Alignment | MRI/CT | 3D | not found locally | - |
| DeTTA | Feature-level Alignment | CT | 2D | not found locally | - |
| TestFit | Feature-level Alignment | CT/PATH | General | included | medseg_tta.methods.testfit |
| DANN | Feature-level Alignment | MRI | 3D | not found locally | - |
| SmaRT | Output-level Regularization | MRI | 3D | not found locally | - |
| DG-TTA | Output-level Regularization | MRI/CT | 3D | included | medseg_tta.methods.dg_tta |
| SaTTCA | Output-level Regularization | CT | 3D | included | medseg_tta.methods.sattca |
| UPL-SFDA | Output-level Regularization | CMR/MRI | General | not found locally | - |
| TENT | Output-level Regularization | General Image | General | included | medseg_tta.methods.tent |
| ProSFDA | Prior Estimation | OCT | 2D | included | medseg_tta.methods.prosfda_2d, medseg_tta.methods.prosfda_3d |
| ExploringTTA | Prior Estimation | US | 3D | included | medseg_tta.methods.exploring_tta |
| PASS | Prior Estimation | OCT | 2D | not found locally | - |
| VPTTA | Prior Estimation | OCT | 2D | not found locally | - |
| AdaMI | Prior Estimation | MRI/CT | 3D | not found locally | - |

## Entrypoints

- DG-TTA: tta2d.py, tta3dCT.py, test_target_tta.py
- SaTTCA: sattc.py, tta2d.py, tta3dCT.py, tta3dMRI.py
- GraTa: tta2d.py, test_target_tta.py, GraTa-master/TTA.py
- GraTa-3D: tta3dCT.py, grata_3d.py, grata_wrapper.py, GraTa-master/TTA.py
- TestFit: tta2d.py, tta3dCT.py, testfit.py
- TENT: tent.py, tent2d.py, tta2d.py, tta3d.py, tta3dCT.py
- ProSFDA-2D: tta2d.py, prosfda/training/run_training.py, prosfda/inference/run_inference.py
- ProSFDA-3D: tta3dCT.py, prosfda/training/run_training.py, prosfda/inference/run_inference.py
- ExploringTTA: test_target_tta.py, tta3dCT.py
- SFDA-FSM: tta2d.py, tta2d_inf.py, tta3d.py, tta3dCT.py, tools/train_adapt.py, tools/test.py
