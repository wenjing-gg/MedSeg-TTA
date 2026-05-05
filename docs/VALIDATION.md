# Validation

Validation was run locally with Python 3.12.2 after reorganizing the method package by adaptation paradigm.

## Passed

```bash
python - <<'PY'
from pathlib import Path
import ast
errors = []
for p in sorted(Path('.').rglob('*.py')):
    if '.git' in p.parts or '__pycache__' in p.parts:
        continue
    try:
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    except SyntaxError as e:
        errors.append((str(p), e.lineno, e.offset, e.msg))
print('checked', 341)
print('errors', len(errors))
PY
python -m compileall medseg_tta DG-TTA SaTTCA GraTa GraTa-3d Testfit tent ProSFDA2D ProSFDA3D ExploringTTA SFDA-FSM
python -m medseg_tta list-paradigms
python -m medseg_tta list-methods
python -m medseg_tta table
python -m medseg_tta show-method tent
python -m medseg_tta validate-structure
python -m medseg_tta run-legacy tent tta3dCT.py --help
python DG-TTA/tta2d.py --help
python SaTTCA/tta3dCT.py --help
python GraTa-3d/tta3dCT.py --help
python Testfit/tta2d.py --help
python tent/tta3dCT.py --help
python ProSFDA2D/tta2d.py --help
python ProSFDA3D/tta3dCT.py --help
python ExploringTTA/test_target_tta.py --help
python SFDA-FSM/tta2d.py --help
```

The generated Python files under `medseg_tta` and the top-level legacy wrapper directories were also checked for leading `#` comments and AST-detectable docstrings; both counts were zero.

## Runtime Scope

Full inference is not part of this validation pass because the current local environment lacks optional runtime dependencies such as `monai`, `SimpleITK`, and `cv2`, and complete datasets/checkpoints are not guaranteed for every method.
