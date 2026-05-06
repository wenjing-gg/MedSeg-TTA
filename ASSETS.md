# Existing Assets, Licenses, and Terms of Use

MedSeg-TTA builds on public medical image datasets and extends several published test-time adaptation implementations. This document records the provenance, access terms, redistribution status, and modification scope of existing assets used by the benchmark.

The MedSeg-TTA framework code is released under the license specified in this repository. Third-party method code adapted from upstream repositories may be subject to the original upstream licenses or terms listed below. Such components are not automatically relicensed under the main MedSeg-TTA repository license unless the upstream license permits it.

Original medical images and annotations are **not redistributed** in this repository. Users should obtain each dataset from its official source and comply with the corresponding license, data-use agreement, challenge terms, or institutional access policy.

## Summary

- We provide benchmark code, preprocessing/evaluation interfaces, method wrappers, configuration files, and dataset access instructions.
- We do not redistribute original medical images or annotations.
- We cite the original papers and dataset creators.
- We document upstream repositories and license status for third-party method code.
- For upstream repositories without an explicit license, we retain attribution and mark the license status as not explicitly stated.
- For upstream code with non-commercial terms, the corresponding component remains subject to the original non-commercial terms and is not relicensed under the main repository license.

---

## Public datasets

| Asset | Type | Used as | Official source / owner | Citation in paper | License / terms status | Redistribution in MedSeg-TTA | Notes |
|---|---|---|---|---|---|---|---|
| BraTS-GLI2024 | Dataset | MRI source domain | BraTS / Synapse | `de2024brats` | Official BraTS/Synapse data-use terms; users should follow the official access policy | Not redistributed; official link only | Used for source-domain brain tumor segmentation training |
| BraTS-SSA | Dataset | MRI target domain | BraTS-SSA / Synapse | `adewole2023brain` | Official Synapse/data-use terms; users should follow the official access policy | Not redistributed; official link only | Used as external target domain for MRI |
| LiTS | Dataset | CT source domain | LiTS challenge organizers | `bilic2023liver` | Official challenge/data-use terms | Not redistributed; official link only | Used for CT liver segmentation source training |
| 3D-IRCADB | Dataset | CT target domain | IRCAD | `soler20103d` | Dataset-specific terms from the official IRCAD distribution page | Not redistributed; official link only | Used as external CT liver segmentation target domain |
| ISIC-2017 | Dataset | DER source domain | ISIC Archive / ISIC Challenge | `8363547` | ISIC archive/challenge terms | Not redistributed; official link only | Used for melanoma lesion segmentation source training |
| PH2 | Dataset | DER target domain | PH2 dataset creators | `mendoncca2015ph2` | Dataset-specific terms from the official PH2 distribution page | Not redistributed; official link only | Used as external dermoscopy target domain |
| TN3K | Dataset | Ultrasound source domain | TN3K authors / official repository | `gong2021multi` | Repository/dataset-specific terms; users should check the official TN3K release | Not redistributed; official link only | Used for thyroid nodule segmentation source training |
| DDTI | Dataset | Ultrasound target domain | DDTI authors / CIMA Lab | `pedraza2015open` | Dataset-specific terms from the official DDTI/CIMA Lab page | Not redistributed; official link only | Used as external ultrasound target domain |
| SZ-CXR | Dataset | CXR source domain | Shenzhen CXR dataset / official or mirrored source | `stirenko2018chest` | Source/Kaggle or official distribution terms, depending on access route | Not redistributed; official link only | Used for lung-field segmentation source training |
| Montgomery CXR Set | Dataset | CXR target domain | NIH / Montgomery County / Open-i | `jaeger2014two` | NIH/Open-i collection terms | Not redistributed; official link only | Used as external CXR target domain |
| RIGA+ / MESSIDOR subsets | Dataset | Fundus/OCT-style source domain in the paper terminology | RIGA+ / MESSIDOR-related sources | `hu2022domain` | Dataset-specific terms from the official distribution page | Not redistributed; official link only | Used for optic structure segmentation source domain |
| RIGA+ / Magrabia and BinRushed subsets | Dataset | Fundus/OCT-style target domain in the paper terminology | RIGA+ / Magrabia / BinRushed sources | `hu2022domain` | Dataset-specific terms from the official distribution page | Not redistributed; official link only | Used as external fundus target domain |
| CRAG | Dataset | Pathology source domain | CRAG dataset authors / official repository | `graham2019mild` | Dataset/repository-specific terms | Not redistributed; official link only | Used for colorectal gland/tissue segmentation source domain |
| GlaS | Dataset | Pathology target domain | GlaS challenge organizers | `SIRINUKUNWATTANA2017489` | Challenge/dataset-specific terms | Not redistributed; official link only | Used as external pathology target domain |

---

## Third-party method code

MedSeg-TTA adapts, extends, or reimplements published TTA methods under a unified benchmark interface. The table below records the upstream source and license status for each method. Method names follow the names used in the MedSeg-TTA paper. When an upstream repository uses a different project or paper title, the repository link and citation context are recorded in the corresponding row.

| Method | Upstream repository | Upstream license / terms status | How used in MedSeg-TTA | Modifications in MedSeg-TTA | Notice |
|---|---|---|---|---|---|
| AIF-SFDA | `https://github.com/JingHuaMan/AIF-SFDA` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official implementation | Unified benchmark interface; standardized data loading; medical segmentation evaluation wrapper; cross-modality integration | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| STDR | `https://github.com/whq-xxh/SFADA-GTV-Seg` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the public implementation associated with Wang et al., IEEE TMI 2024 | Unified benchmark interface; standardized preprocessing/evaluation; benchmark integration; style/statistical alignment adaptation where needed | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| RSA | `https://github.com/zenghy96/Reliable-Source-Approximation` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official implementation | Unified benchmark interface; extension for benchmark protocol and dimensional settings where needed | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| SFDA-FSM | `https://github.com/CityU-AIM-Group/SFDA-FSM` | Apache-2.0 | Adapted from the official implementation | Unified interface; standardized preprocessing/evaluation; extension to benchmark modalities | Apache-2.0 license and attribution should be retained |
| DL-TTA | `https://github.com/med-air/DLTTA` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the public implementation | Unified interface; medical segmentation benchmark wrapper; dimensional/protocol adaptation | Attribution retained; upstream also acknowledges components such as Tent and ATTA |
| GraTa | `https://github.com/Chen-Ziyang/GraTa` | MIT | Adapted from the official implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | MIT license and attribution should be retained |
| UDA-MIMA | `https://github.com/huqian999/UDA-MIMA` | License not confirmed at the time of documentation | Adapted from the public implementation | Unified interface; benchmark wrapper; 2D/3D compatibility adjustments where needed | Attribution retained; verify upstream license before redistribution |
| DeTTA | `https://github.com/WenRuxue/DeTTA` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official/public implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| TestFit | `https://github.com/yizhezhang2000/TestFit` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the public core-code repository | Unified interface; standardized preprocessing/evaluation; benchmark integration | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| DANN | `https://github.com/fungtion/DANN` or reimplemented from Ganin et al. | MIT if using `fungtion/DANN`; otherwise reimplemented from the paper | Adapted or reimplemented as a feature-level alignment baseline | Unified segmentation interface; benchmark protocol integration | Retain MIT license if code from `fungtion/DANN` is used |
| SmaRT | `https://github.com/baiyou1234/SmaRT` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official/public implementation | Unified interface; benchmark wrapper; 2D/3D compatibility adjustments where needed | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| DG-TTA | `https://github.com/multimodallearning/DG-TTA` | MIT | Adapted from the official implementation | Unified interface; medical segmentation benchmark integration | MIT license and attribution should be retained |
| SaTTCA | `https://github.com/SplinterLi/SaTTCA` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| UPL-SFDA | `https://github.com/HiLab-git/UPL-SFDA` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| TENT | `https://github.com/DequanWang/tent` | License not confirmed at the time of documentation | Adapted from the official example code or reimplemented from the paper | Entropy minimization baseline adapted to segmentation and benchmark protocol | Attribution retained; verify upstream license before redistribution if source files are copied |
| ProSFDA | `https://github.com/ShishuaiHu/ProSFDA` | MIT | Adapted from the official/public implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | MIT license and attribution should be retained |
| ExploringTTA | `https://github.com/joshuaomolegan/TTA-for-3D-Fetal-Subcortical-Segmentation` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the public implementation | Unified interface; adaptation to benchmark tasks and protocol | Attribution retained; not claimed as newly licensed by MedSeg-TTA |
| PASS | `https://github.com/EndoluminalSurgicalVision-IMR/PASS` | License not explicitly stated in the upstream repository at the time of documentation | Adapted from the official implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | Attribution retained; PASS itself acknowledges upstream codebases such as ProSFDA, TTA, and VPTTA |
| VPTTA | `https://github.com/Chen-Ziyang/VPTTA` | MIT | Adapted from the official implementation | Unified interface; standardized preprocessing/evaluation; benchmark integration | MIT license and attribution should be retained |
| AdaMI | `https://github.com/mathilde-b/TTA` | Non-commercial research purposes only, according to the upstream repository documentation | Adapted from the official/public implementation | Unified interface; segmentation benchmark integration; dimensional adjustments where needed | This component remains subject to the upstream non-commercial research-use terms and is not relicensed under the main MedSeg-TTA license |

---

## Third-party code licensing notes

Some upstream repositories used by MedSeg-TTA do not explicitly provide a license file. In those cases, MedSeg-TTA records the upstream repository and retains attribution, but does not claim that those components are newly licensed under the MedSeg-TTA repository license.

If a third-party implementation has no explicit license and substantial source files are copied or modified, users and maintainers should consider one of the following:

1. obtain permission from the original authors;
2. replace the copied source files with a clean-room reimplementation based on the paper;
3. keep the component excluded from redistribution until license status is clarified.

Components with explicit permissive licenses, such as MIT or Apache-2.0, should retain the upstream copyright and license notices.

Components with non-commercial or research-only terms remain subject to those terms and are not relicensed under the main repository license.

---

## Software dependencies

MedSeg-TTA also depends on common scientific Python and deep learning libraries. Exact versions are specified in the repository dependency files, such as `pyproject.toml`, `requirements.txt`, or environment files.

Examples include:

| Dependency | Role | License / terms |
|---|---|---|
| Python | Programming language | Python Software Foundation License |
| PyTorch | Deep learning framework | PyTorch/BSD-style license; see official PyTorch license |
| NumPy | Numerical computing | BSD-style license |
| SciPy | Scientific computing | BSD-style license |
| SimpleITK | Medical image I/O and processing | Apache-2.0 |
| nibabel | Neuroimaging file I/O | MIT |
| scikit-image | Image processing | BSD-style license |
| OpenCV | Image processing | Apache-2.0 |
| MONAI | Medical imaging deep learning framework, if used | Apache-2.0 |

Users should refer to the dependency files and the upstream projects for exact license text.

---

## Redistribution policy

MedSeg-TTA does not redistribute original medical images or annotations. Dataset users must download datasets from the official sources and comply with the corresponding data-use terms.

MedSeg-TTA may include adapted method implementations and benchmark wrappers. Where upstream code has an explicit license, the corresponding license notice should be retained. Where upstream code has no explicit license, the component is documented as such and should not be interpreted as being relicensed under the main repository license.

---

## Citation and attribution

If you use MedSeg-TTA, please cite the MedSeg-TTA paper and the original papers for the datasets and baseline methods used in your experiments. The benchmark paper bibliography contains the citations for all datasets and methods listed above.

For any third-party component, users should also follow the citation instructions provided by the corresponding upstream repository or paper.

---

## Disclaimer

This document is provided for transparency and research reproducibility. It is not legal advice. Users are responsible for ensuring that their use of datasets, third-party code, and benchmark components complies with the applicable licenses, data-use agreements, and institutional policies.