# A Large Scale Benchmark for Test Time Adaptation in Medical Image Segmentation

[![GitHub](https://img.shields.io/badge/GitHub-MedSeg--TTA-181717?logo=github)](https://github.com/wenjing-gg/MedSeg-TTA)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-orange)](https://wenjing-gg.github.io/MedSeg-TTA/)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20soon-lightgrey?logo=arxiv)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

MedSeg-TTA is a benchmark for test-time adaptation in medical image segmentation. This repository centers on the public leaderboard, the benchmark figures behind it, and the currently available local method implementations organized by paradigm.

## Leaderboard

<h1 align="center">
  <a href="https://wenjing-gg.github.io/MedSeg-TTA/">MEDSEG-TTA</a>
</h1>

<p align="center">
  <a href="https://wenjing-gg.github.io/MedSeg-TTA/">👉 Click here to explore the full leaderboard in detail.</a>
</p>

## Benchmark Overview

The benchmark unifies medical TTA evaluation around a shared surface that connects source-target dataset pairs, paradigm-level comparisons, and local method code roots.

![Framework](fig/framework.png)

## Dataset Coverage

MedSeg-TTA covers seven modalities and multiple cross-domain source-target pairs spanning MRI, CT, US, PATH, DER, OCT, and CXR.

![Dataset Coverage](fig/dataset.png)

## Repository Layout

```text
MedSeg-TTA/
├── medseg_tta/
├── site/
├── feature_level_alignment/
│   ├── GraTa/
│   └── Testfit/
├── input_level_transformation/
│   ├── SFDA-FSM/
│   ├── DLTTA/
│   ├── STDR/
│   └── RSA/
├── output_level_regularization/
│   ├── DG-TTA/
│   ├── SaTTCA/
│   ├── UPL-SFDA/
│   └── tent/
├── prior_estimation/
│   ├── ExploringTTA/
│   ├── AdaMI/
│   ├── PASS/
│   └── ProSFDA/
└── ASSETS.md
```

## Assets and Licensing

Dataset provenance, third-party code sources, redistribution notes, and license details are documented in [ASSETS.md](ASSETS.md).

## Citation

If you find this project useful, please cite:

```bibtex
@article{anonymous2025medsegtta,
  title   = {A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation},
  author  = {Anonymous Authors},
  journal = {Anonymous preprint},
  year    = {2025}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
