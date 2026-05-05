# Validation

Validation was rerun locally after unifying top-level method directories and moving dimension handling into `two_d` / `three_d` subtrees.

## Install

```bash
pip install -e .
pip install -e .[runtime]
```

The `runtime` extra covers the currently required runtime dependencies that were missing in the local environment during the original package release, including `monai`, `dynamic-network-architectures`, `medpy`, `opencv-python-headless`, and `SimpleITK`.

## Passed

```bash
python -m compileall medseg_tta DG-TTA GraTa ProSFDA SaTTCA Testfit tent SFDA-FSM ExploringTTA
python -m medseg_tta list-paradigms
python -m medseg_tta list-methods --flat
python -m medseg_tta list-methods --dimension 2d --flat
python -m medseg_tta list-methods --dimension 3d --flat
python -m medseg_tta show-method grata
python -m medseg_tta show-method prosfda --dimension 3d
python -m medseg_tta validate-structure
python -m medseg_tta run-legacy grata_3d tta3dCT.py --help
python GraTa/two_d/tta2d.py --help
python GraTa/three_d/tta3dCT.py --help
python ProSFDA/two_d/tta2d.py --help
python ProSFDA/three_d/tta3dCT.py --help
python SFDA-FSM/two_d/tools/test.py --help
python DG-TTA/two_d/tta2d.py --help
python DG-TTA/three_d/tta3dCT.py --help
```

## Runtime Smoke

After dependency installation, at least one 2D entrypoint and one 3D entrypoint should be executed without `--help`.

Recommended smoke commands:

```bash
python DG-TTA/two_d/tta2d.py --target_dir /tmp/missing_target --checkpoint_dir /tmp/medseg-tta-2d --model_path /tmp/missing.pth --gpu -1
python tent/three_d/tta3dCT.py --target_dir /tmp/missing_target --tent_results_dir /tmp/medseg-tta-3d --checkpoint /tmp/missing.pth --gpu -1
```

These commands are expected to complete argument parsing, import resolution, and result-directory setup before failing on the intentionally missing data or checkpoint path.

## Scope Notes

- Full inference success still depends on method-specific datasets and checkpoints being available locally.
- Some research scripts retain upstream hard-coded defaults; this refactor preserves algorithm code and focuses on structure, entrypoint routing, and import/runtime compatibility.
