# A Large Scale Benchmark for Test Time Adaptation in Medical Image Segmentation

[![GitHub](https://img.shields.io/badge/GitHub-MedSeg--TTA-181717?logo=github)](https://github.com/wenjing-gg/MedSeg-TTA)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-orange)](https://wenjing-gg.github.io/MedSeg-TTA/)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20soon-lightgrey?logo=arxiv)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

MedSeg-TTA is a benchmark for test-time adaptation in medical image segmentation. This repository centers on the public leaderboard, the benchmark metadata behind it, and the currently available local method implementations organized by paradigm.

## Leaderboard

Live site:

```text
https://wenjing-gg.github.io/MedSeg-TTA/
```

The leaderboard is the main entry point for this repository. It provides:

- paradigm-level comparison across domain-shift regimes
- method-level ranking across seven medical imaging modalities
- modality-specific drilldown with Dice and HD95 views
- direct GitHub jumps into local method folders, including `two_d/` and `three_d/` paths where available

The web source lives in `site/`, the leaderboard data lives in `site/data/leaderboard.json`, and deployment is handled by `.github/workflows/deploy-pages.yml`.

## Benchmark Scope

- 7 modalities: MRI, CT, US, PATH, DER, OCT, CXR
- 4 paradigms:
  - `input_level_transformation`
  - `feature_level_alignment`
  - `output_level_regularization`
  - `prior_estimation`
- current local method roots:
  - `SFDA-FSM`
  - `GraTa`
  - `Testfit`
  - `DG-TTA`
  - `SaTTCA`
  - `tent`
  - `ProSFDA`
  - `ExploringTTA`

## Repository Layout

```text
MedSeg-TTA/
├── medseg_tta/
├── site/
├── feature_level_alignment/
│   ├── GraTa/
│   └── Testfit/
├── input_level_transformation/
│   └── SFDA-FSM/
├── output_level_regularization/
│   ├── DG-TTA/
│   ├── SaTTCA/
│   └── tent/
├── prior_estimation/
│   ├── ExploringTTA/
│   └── ProSFDA/
└── ASSETS.md
```

## Assets and Licensing

Dataset provenance, third-party code sources, redistribution notes, and license details are documented in [ASSETS.md](/Volumes/VVV/TTA/Code/MedSeg-TTA/ASSETS.md:1).

## Citation

If you find this project useful, please cite:

```bibtex
@article{MedSeg-TTA,
  title   = {MedSeg-TTA: Benchmarking Test-time Adaptation Methods for Domain Shift in Medical Image Segmentation},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2025}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
