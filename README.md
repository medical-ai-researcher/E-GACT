# Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT)

<p align="center">
  <img src="architecture.jpg" width="95%" alt="E-GACT Architecture Diagram">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Computer%20Methods%20and%20Programs%20in%20Biomedicine%20%28Under%20Review%29-green.svg)](#)?style=flat-square

> **Official Code Repository for the paper:**  
> *"Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability"*  
> Submitted to **Computer Methods and Programs in Biomedicine** (Elsevier, IF=6.4, Q1).

---

## ⚡ Reproducibility

**Instructions:**

1. Open our official interactive Notebook via Google Colab:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medical-ai-researcher/E-GACT/blob/main/E_GACT_FINAL.ipynb)

2. Set Hardware Accelerator to **T4 GPU** (`Runtime → Change runtime type`).

3. Click **`Runtime → Run All`**.

Pre-computed checkpoints are loaded automatically from the `checkpoints/` folder — full pipeline runs in **~5 minutes** without retraining.

> **📝 Note:** Manuscript results (Tables 3–4) were produced on an RTX 4090 cluster with full 5-fold CV (seed=42). Checkpoints reproduce all reported figures and tables exactly.

---

## 📖 Overview

E-GACT addresses three clinical bottlenecks via a **dual-path architecture**:

1. **Path A — Tabular Transformer:** Lightweight intra-patient feature modelling (d=32, H=4, 2 layers).
2. **Path B — Raw-Feature GNN:** Strictly inductive k-NN graph (k=10) built on raw feature space via a novel **hybrid cosine–Jaccard similarity metric** (α=0.5) — independent of Path A, built once per fold via FAISS FlatIP/HNSW.
3. **Supervised Contrastive Learning (SCL):** Regularises against class imbalance (τ=0.07, λ=0.6).
4. **Dual-Layer Clinical Explainability:** Subgraph-frozen KernelSHAP + Neighbourhood Influence Score (NIS) + Case Retrieval.

**Key numbers:**
| Metric | Value |
|:---|:---|
| Parameters (NHANES) | 39,473 |
| Parameters (BRFSS) | 33,425 |
| Parameters (130-US) | ~150,000 |
| CPU Inference (N=250k) | 3.21 ms / patient (FP32) |
| INT8 Inference (N=250k) | 3.81 ms / patient |
| Model weights (INT8) | 90.9 KB |
| Complexity | O(N log N) |

---

## 📊 Results Summary

Evaluated via **5-fold stratified CV** against **8 baselines** across 3 primary clinical cohorts.

| Dataset | E-GACT AUC | Best Baseline | Status |
|:---|:---|:---|:---|
| **NHANES** (N=6,049) | **0.8397 ± 0.005** | CatBoost 0.8364 | ✅ Best across all metrics |
| **130-US** (N=101,766) | 0.6585 ± 0.007 | LightGBM 0.6777 | ✅ Best among deep learning |
| **BRFSS imbalanced** (N=70,000) | **0.8290 ± 0.004** | MLP 0.8296 | ✅ p=0.001 vs LightGBM |
| **BRFSS balanced** (N=50,000) | **0.8304 ± 0.003** | SAINT 0.8305 | ✅ Scalability benchmark |

---

## 📊 Datasets

| Dataset | Modality | N | F | Target | Pos. (%) |
|:---|:---|:---|:---|:---|:---|
| [NHANES (2017–2018)](https://wwwn.cdc.gov/nchs/nhanes/) | Clinical Lab | 6,049 | 84 | Early T2DM (HbA1c ≥6.5%) | 14.3 |
| [130-US Hospitals](https://archive.ics.uci.edu/dataset/296/) | EHR | 101,766 | 45 | 30-Day Readmission | 11.2 |
| [CDC BRFSS (2015)](https://www.cdc.gov/brfss/) | Population Survey | 70,000 / 50,000 | 21 | T2DM (self-reported) | 13.9 / 50.2 |

---

## 🔍 Dual-Layer XAI

**Layer 1 — Subgraph-Frozen KernelSHAP**  
Top NHANES predictors: Age, family history (MCQ300C), statin use, waist circumference.  
Top BRFSS predictors: GenHlth, BMI, Age, HighBP, HighChol.

**Layer 2 — Neighbourhood Influence Score + Case Retrieval**  
NIS(i) = ŷᵢ − mean(ŷ_neighbours). Case vignette: Δtopo=+0.055 elevates borderline patient (0.272→0.327) into High-Risk, triggering HbA1c referral.

---

## 📝 Citation

```bibtex
@article{egact2026,
  title   = {Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT)
             for Early Diabetes Risk Prediction: Bridging Algorithmic
             Topology and Clinical Explainability},
  author  = {İsa Ataş},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {under review}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
