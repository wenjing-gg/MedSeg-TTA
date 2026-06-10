# 🧠 SmaRT: Style-Modulated Robust Test-Time Adaptation for Cross-Domain Brain Tumor Segmentation

---

## 📖 Introduction
SmaRT (**S**tyle-**m**odulated **R**obust **T**est-time adaptation) is a source-free framework for brain tumor segmentation in MRI.  
It addresses **severe domain shifts** (e.g., low-field SSA scans, pediatric gliomas) by combining:
- 🎨 **Style-aware augmentation** with adaptive transformations.
- 🔄 **Dual-EMA momentum update** for stable pseudo-label refinement.
- 🧩 **Structural priors** (consistency, integrity, connectivity) for anatomical fidelity.

---

## 🧩 Framework Overview
<div align="center">
<img src="./imgs/model.pdf" width="90%">
</div>

---

---

## 📂 Dataset
We use three benchmark datasets:

- [BraTS-2024](https://www.synapse.org/#!Synapse:syn51156910/wiki/): 1350 glioma cases, with modalities T1, T1Gd, T2, FLAIR.  


- [BraTS-SSA](https://www.synapse.org/#!Synapse:syn51156910/wiki/): 60 glioma cases from Sub-Saharan Africa (low-field MRI).  
  - Contains motion artifacts, lower resolution, and SNR issues.  

- [BraTS-PED](https://www.synapse.org/#!Synapse:syn51156910/wiki/): 464 pediatric glioma cases.  
  

**Split principle**:  
- Training: majority of BraTS (source domain).  
- Testing: full SSA and PED datasets.  
- Validation: 20% of BraTS training set.

**Preprocessing**:  
- Resampling to **128×128×128** voxels.  
- Intensity normalization (per non-zero region).  
- Four modalities concatenated as input.

---

## ⚙️ Environment Setup
Recommended setup:
- **Python**: 3.8  
- **PyTorch**: 2.4.1+cu121  
- **CUDA**: ≥ 11.7  
- **GPU**: NVIDIA A800  

Install dependencies:
```bash
pip install -r requirements.txt