<h1 align="center">
  Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT)
</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Artificial Intelligence in Medicine Submission (Under Review)-green.svg" alt="Paper"></a>
</p>

> **Official Code Repository for the paper:**  
> *"Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability"* (Submitted to Artificial Intelligence in Medicine - AIM).

---

## ⚡ Zero-Click Reproducibility (Reviewer Guide)

We deeply respect the time of academic peer-reviewers. To facilitate a seamless and transparent review process, we provide an automated, **"Zero-Click" Universal Data Pipeline**. 
- No Google Drive mounting required.
- No API keys, credentials, or manual dataset downloads needed.

**Instructions:**
1. Open our official interactive Notebook via Google Colab:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AroQ5a6Il4cyYqf21RW21rxwGuhp4wYB)
2. Ensure the Hardware Accelerator is set to **T4 GPU** (`Runtime -> Change runtime type`).
3. Click **`Runtime -> Run All`**.
4. The script will autonomously fetch the raw clinical cohorts, perform strictly inductive leakage-free graph construction, train the E-GACT architecture, compute Neighbourhood Influence Scores (NIS), and output the ROC-AUC benchmarks alongside high-resolution Explainable AI (XAI) figures.

> **📝 Editorial Note on Colab Constraints:** The results reported in the main manuscript (Tables 3 & 4) are derived from a rigorous 5-fold stratified cross-validation on a dedicated high-resource cluster (RTX 4090). To comfortably satisfy Colab's free-tier GPU time and RAM constraints (15GB), this reproducibility notebook utilises a single 80/20 stratified split and reduces the FAISS HNSW inference parameter to $M=16$ (instead of $M=32$). Results obtained here closely approximate, but may marginally differ from, the tabled 5-fold figures.

---

## 📖 Overview

Predicting Type 2 Diabetes Mellitus (T2DM) and associated clinical outcomes from tabular Electronic Health Records (EHR) is critical for early clinical intervention. However, current Deep Tabular Models chronically underperform against tree-based ensembles on noisy clinical datasets, and fail to scale on massive population-level cohorts due to $\mathcal{O}(N^2)$ attention complexities. Furthermore, standard algorithms evaluate patients as Independent and Identically Distributed (I.I.D.) instances, neglecting the fundamental clinical paradigm of **Case-Based Reasoning**.

**E-GACT** addresses these methodological bottlenecks by integrating:
1. **Lightweight Tabular Transformer:** For non-linear, intra-patient feature projection.
2. **Strictly Inductive $k$-NN Graph via EMA Buffer:** To capture inter-patient topological similarities (Case-Based Reasoning). An Exponential Moving Average (EMA) buffer prevents *representation drift* without incurring full-dataset forward passes, ensuring $\mathcal{O}(N \log N)$ architectural scalability.
3. **Supervised Contrastive Learning (SCL):** To actively organize the topological latent space and prevent over-smoothing against severe class imbalances.
4. **Dual-Layer Clinical Explainability:** Unifying Subgraph-Frozen SHAP (feature-level) and Neighbourhood Influence Scores (topology-level) to provide clinically actionable insights.

**Edge AI Compatibility:** With a highly compact footprint of only **0.45M learnable parameters**, E-GACT operates with $<45$ ms inference latency on standard microprocessors, qualifying the framework for zero-latency, privacy-preserving local Edge AI deployments.

---

## 🏗️ Architecture

<p align="center">
  <img src="E-GACT Architecture.jpg" width="95%" alt="E-GACT Architecture Diagram">
  <br><em>Figure 1: Overall workflow of the strictly inductive, leakage-free E-GACT architecture featuring EMA-based index refresh.</em>
</p>

---

## 📊 Benchmarked Datasets

To demonstrate algorithmic robustness and scalability across varying modalities, E-GACT is evaluated on three globally validated, open-access cohorts:

| Dataset | Modality | Size (Patients) | Focus Area | Target Prediction |
| :--- | :--- | :--- | :--- | :--- |
| **[NHANES (2017-2018)](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)** | Clinical Lab + Demographics | ~6,000 | Physiological Signals | T2DM (HbA1c $\geq$ 6.5) |
| **[130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)** | Electronic Health Records (EHR)| ~101,000 | Case-Based Reasoning | Readmission Risk |
| **[CDC BRFSS (2015)](https://www.cdc.gov/brfss/annual_data/annual_2015.html)** | Population Survey | 50,000* | Edge AI Scalability | T2DM |

*\*Note on BRFSS:* As stated in the manuscript, the 50,000-patient 1:1 balanced slice of the BRFSS dataset is used specifically to benchmark $\mathcal{O}(N \log N)$ computational scalability without the confounding effects of class imbalance. The predictive performance on the naturally imbalanced BRFSS cohort (14.9% positive rate) is fully reported in **Supplementary Table S1** of the paper.

---

## 🔍 Dual-Layer Clinical Explainability (XAI)

E-GACT strictly avoids the "black-box" paradigm. Computing feature attributions in a Graph Neural Network is inherently complex due to neighborhood contamination (Message Passing). We resolve this via a novel **Dual-Layer** approach:

1. **Layer 1: Subgraph Freezing SHAP (Feature-Level)**
   The pipeline automatically freezes the historical graph topology and isolates the target patient's input. It generates high-resolution SHAP summary plots highlighting exactly which physiological factors (e.g., BMI, Age, Glycohemoglobin) drove a specific patient into a high-risk category.
2. **Layer 2: Neighbourhood Influence Score (Topology-Level)**
   The pipeline calculates the NIS (Equation 5 in the paper) to explicitly quantify how much the patient's retrieved historical neighbourhood shifted their baseline risk prediction.

Additionally, **t-SNE Latent Space Visualizations** are generated to demonstrate how the Supervised Contrastive Loss mathematically forces diabetic and healthy patient profiles into distinct topological manifolds.

---
## 📝 Citation

If you find this codebase or methodology useful in your research, please consider citing our paper:
```bibtex
@article{egact_2026,
  title={Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability},
  author={Anonymous Authors},
  journal={Artificial Intelligence in Medicine - AIM (Submitted)},
  year={under review}
}
