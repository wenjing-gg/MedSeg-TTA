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
  <a href="https://wenjing-gg.github.io/MedSeg-TTA/">馃憠 Click here to explore the full leaderboard in detail.</a>
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
鈹溾攢鈹€ medseg_tta/
鈹溾攢鈹€ site/
鈹溾攢鈹€ feature_level_alignment/
鈹?  鈹溾攢鈹€ GraTa/
鈹?  鈹斺攢鈹€ Testfit/
鈹溾攢鈹€ input_level_transformation/
鈹?  鈹溾攢鈹€ SFDA-FSM/
鈹?  鈹斺攢鈹€ UPL-SFDA/
鈹溾攢鈹€ output_level_regularization/
鈹?  鈹溾攢鈹€ DG-TTA/
鈹?  鈹溾攢鈹€ SaTTCA/
鈹?  鈹斺攢鈹€ tent/
鈹溾攢鈹€ prior_estimation/
鈹?  鈹溾攢鈹€ ExploringTTA/
鈹?  鈹溾攢鈹€ AdaMI/
鈹?  鈹斺攢鈹€ ProSFDA/
鈹斺攢鈹€ ASSETS.md
```

## Assets and Licensing

Dataset provenance, third-party code sources, redistribution notes, and license details are documented in [ASSETS.md](ASSETS.md).

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
