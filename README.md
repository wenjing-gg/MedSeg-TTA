# A Large Scale Benchmark for Test Time Adaptation in Medical Image Segmentation

[![GitHub](https://img.shields.io/badge/GitHub-MedSeg-TTA-181717?logo=github)](https://github.com/wenjing-gg/MedSeg-TTA)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20soon-lightgrey?logo=arxiv)](#)
[![Modalities](https://img.shields.io/badge/Modalities-7-blue)](#)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Coming%20soon-orange)](#)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#)

MedSeg-TTA is an open benchmark for evaluating test-time adaptation (TTA) methods in medical image segmentation. It standardizes data, metrics, and protocols across modalities, organs, and tasks, enabling fair and reproducible comparisons under strict constraints (fixed backbone, no source-domain access, and no implicit leakage).

![Framework](fig/framework.png)

## Main Contributions

- Multi-modal and multi-center open-source dataset: We construct a dataset that covers tumor, organ, and lesion segmentation across seven imaging modalities, namely MRI, CT, US, PATH, DER, OCT, and CXR. The dataset employs standardized preprocessing and partitioning, faithfully reflecting distribution shifts across institutions, scanners, and populations, thereby providing a data foundation for the unified TTBA benchmark.
- Strong baselines and SOTA reproduction: Under a unified TTBA setting that fixes the backbone and forbids source-domain access as well as any implicit leakage, we systematically reproduce and validate twenty state-of-the-art TTA methods across four paradigms, delivering readily usable strong baselines and reproducible scripts. We also establish a public leaderboard that enables comparisons across modalities, organs, and tasks using region-consistency and structure-sensitive metrics such as Dice and HD95.
- Paradigm taxonomy and applicability lineage: We categorize TTA methods into four paradigms according to their locus of operation and, based on evaluations across modalities, organs, and tasks, construct lineage maps that highlight effective and ineffective regimes. This delineates applicability boundaries and provides practical guidance for future method selection.

![Dataset Coverage](fig/dataset.png)

## Source–Target Dataset Pairs and Downloads

After aligning class definitions between source and target domains, “Binary” tasks have a foreground–background split, while “4-Class” tasks include the background plus three anatomical classes. The Reprocess column uses ✓ to indicate standardized reprocessing and ✗ for original data.

| Modal | Dataset | Domain | Category | Quantity | Year | Reprocess | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MRI | BraTS-GLI2024 | Source | 4-Class | 1k–2k | 2024 | ✗ | [Link](https://www.synapse.org/Synapse:syn59059776) |
| MRI | BraTS-SSA | Target | 4-Class | 60 | 2023 | ✗ | [Link](https://www.synapse.org/#!Synapse:syn51514109) |
| CT | LiTS | Source | Binary | <0.5k | 2017 | ✓ | [Link](https://competitions.codalab.org/competitions/17094) |
| CT | 3D-IRCADB | Target | Binary | 20 | 2010 | ✓ | [Link](https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download) |
| Dermoscopy | ISIC-2017 | Source | Binary | 2k–3k | 2017 | ✗ | [Link](https://challenge.isic-archive.com/data/#2017) |
| Dermoscopy | PH² | Target | Binary | <0.5k | 2015 | ✗ | [Link](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) |
| Ultrasound | TN3K | Source | Binary | 2k–3k | 2021 | ✗ | [Link](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) |
| Ultrasound | DDTI | Target | Binary | 0.5k–1k | 2015 | ✗ | [Link](http://cimalab.intec.co/applications/thyroid/) |
| X-Ray | SZ-CXR | Source | Binary | 0.5k–1k | 2018 | ✗ | [Link](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen) |
| X-Ray | Montgomery | Target | Binary | <0.5k | 2021 | ✗ | [Link](https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip) |
| Fundus | RIGA+ (MES) | Source | Binary | <0.5k | 2021 | ✓ | [Link](https://github.com/mohaEs/RIGA-segmentation-masks/raw/main/RIGA_masks.zip) |
| Fundus | RIGA+ (MB) | Target | Binary | <0.5k | 2021 | ✓ | [Link](https://github.com/mohaEs/RIGA-segmentation-masks/raw/main/RIGA_masks.zip) |
| Histopathology | CRAG | Source | Binary | <0.5k | 2019 | ✓ | [Link](https://github.com/XiaoyuZHK/CRAG-Dataset_Aug_ToCOCO) |
| Histopathology | Glas | Target | Binary | <0.5k | 2017 | ✓ | [Link](https://academictorrents.com/details/208814dd113c2b0a242e74e832ccac28fcff74e5) |

![Paradigm Taxonomy](fig/paradigm.png)

## Getting Started

- Code and reproducible baselines are being prepared for public release. The repository already hosts standardized figures and dataset references; training and evaluation scripts will follow under a unified TTBA protocol.
- Planned components: baseline configs, evaluation toolkit (Dice/HD95 + region consistency), submission format, and a public leaderboard for cross-modality comparisons.

## Citation

If you find this project useful, please cite:

```
@article{MedSeg-TTA,
  title   = {MedSeg-TTA: Benchmarking Test-time Adaptation Methods for Domain Shift in Medical Image Segmentation},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2025}
}
```

## Links

- Repository: https://github.com/wenjing-gg/MedSeg-TTA
- Paper (arXiv): coming soon
- Leaderboard: coming soon

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` for guidelines (branching, Conventional Commits, CI checks). PRs should pass markdownlint and link checks.

## License

This project is released under the MIT License. See `LICENSE` for details.
