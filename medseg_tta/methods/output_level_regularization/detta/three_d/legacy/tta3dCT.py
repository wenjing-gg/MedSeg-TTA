from __future__ import annotations

from medseg_tta.methods.output_level_regularization.detta.two_d.legacy.detta_core import main


if __name__ == "__main__":
    raise SystemExit(main(dimension="three_d", modality="ct"))
