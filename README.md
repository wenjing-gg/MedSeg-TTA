# MedSegTTABoard: Benchmarking Test-time Adaptation Methods for Domain Shift in Medical Image Segmentation

MedSegTTABoard 旨在统一并复现医学图像分割领域的测试时自适应（TTA）方法评测，提供标准化的数据、脚本与可复现实验，以便在多模态、多器官、多任务下进行公平对比。

![Framework](fig/framework.png)

## 核心贡献（Main Contributions）

- **Multi-modal and multi-center open-source dataset:** We construct a dataset that covers tumor, organ, and lesion segmentation across seven imaging modalities, namely MRI, CT, US, PATH, DER, OCT, and CXR. The dataset employs standardized preprocessing and partitioning, faithfully reflecting distribution shifts across institutions, scanners, and populations, thereby providing a data foundation for the unified TTBA benchmark.
- **Strong baselines and SOTA reproduction:** Under a unified TTBA setting that fixes the backbone and forbids source-domain access as well as any implicit leakage, we systematically reproduce and validate twenty state-of-the-art TTA methods across four paradigms, delivering readily usable strong baselines and reproducible scripts. We also establish a public leaderboard that enables comparisons across modalities, organs, and tasks using region-consistency and structure-sensitive metrics such as Dice and HD95.
- **Paradigm taxonomy and applicability lineage:** We categorize TTA methods into four paradigms according to their locus of operation and, based on evaluations across modalities, organs, and tasks, construct lineage maps that highlight effective and ineffective regimes. This delineates applicability boundaries and provides practical guidance for future method selection.

![Dataset Coverage](fig/dataset.png)

## 数据集下载与配对表（Source–Target Dataset Pairs）

对齐源域与目标域的类别定义后，“Binary”为二分类前景-背景任务；“4-Class”为四类（含背景）。`Reprocess` 列使用 ✓ 表示标准化重处理，✗ 表示使用原始数据。

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

---

若你在 GitHub 查看本页未显示 PDF，请使用上面的 PNG 版本预览；原始 PDF 也已随仓库发布于 `fig/` 目录。

