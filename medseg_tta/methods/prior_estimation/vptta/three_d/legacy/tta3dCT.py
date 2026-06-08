from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VPTTA 3D CT entrypoint placeholder for the MedSeg-TTA comparison layout."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="shared initialized source checkpoint")
    parser.add_argument("--target_dir", type=str, required=True, help="3D CT target dataset root")
    parser.add_argument("--source_dataset", type=str, default="CT", help="source dataset identifier")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main() -> int:
    build_parser().parse_args()
    raise NotImplementedError(
        "The provided VPTTA release only includes 2D OPTIC/POLYP workflows. "
        "This 3D CT entrypoint is reserved for wiring a real 3D VPTTA implementation "
        "that reads the shared initialized checkpoint and performs target adaptation."
    )


if __name__ == "__main__":
    raise SystemExit(main())

