<h1 align="center">
  Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT)
</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-IEEE_JBHI_Submission-green.svg" alt="Paper"></a>
</p>

> **Official Code Repository for the paper:**  
> *"Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability"* (Submitted to IEEE Journal of Biomedical and Health Informatics - JBHI).

---

## 📖 Overview

Predicting Type 2 Diabetes Mellitus (T2DM) and associated clinical outcomes from tabular Electronic Health Records (EHR) is critical for early clinical intervention. However, current Deep Tabular Models (e.g., TabNet, FT-Transformer) chronically underperform against tree-based ensembles on noisy clinical datasets, and fail to scale on massive population-level cohorts due to $\mathcal{O}(N^2)$ attention complexities. Furthermore, standard algorithms evaluate patients as Independent and Identically Distributed (I.I.D.) instances, neglecting the fundamental clinical paradigm of **Case-Based Reasoning**.

**E-GACT** addresses these methodological bottlenecks by integrating:
1. **Lightweight Tabular Transformer:** For non-linear, intra-patient feature projection.
2. **FAISS $k$-NN Graph Neural Network (GNN):** For inter-patient topological similarities (Case-Based Reasoning), dynamically constructed in $\mathcal{O}(N \log N)$ time.
3. **Supervised Contrastive Learning (SCL):** To actively organize the topological latent space and prevent over-smoothing against severe class imbalances.

**Edge AI Compatibility:** With a highly compact footprint of only **0.45M learnable parameters**, E-GACT operates with $<45$ ms inference latency on standard microprocessors. This qualifies the framework for zero-latency, privacy-preserving local Edge AI deployments in resource-constrained primary care environments.

---

## 🏗️ Architecture

<p align="center">
  <img src="E-GACT Architecture.jpg" width="95%" alt="E-GACT Architecture Diagram">
  <br><em>Figure 1: Overall workflow of the strictly inductive, leakage-free E-GACT architecture.</em>
</p>

---

## 📊 Benchmarked Datasets

To demonstrate algorithmic robustness and scalability across varying modalities, E-GACT is evaluated on three globally validated, open-access cohorts:

| Dataset | Modality | Size (Patients) | Focus Area | Target Prediction | E-GACT (ROC-AUC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NHANES (2017-2018)** | Clinical Lab + Demographics | ~6,000 | Physiological Signals | T2DM (HbA1c $\geq$ 6.5) | **0.816** |
| **130-US Hospitals** | Electronic Health Records (EHR)| ~101,000 | Case-Based Reasoning | Readmission Risk | **0.662** |
| **CDC BRFSS (2015)** | Population Survey | ~50,000* | Edge AI Scalability | T2DM (Imbalanced) | **0.832** |

*\*Note: To ensure stable reproducibility within standard free-tier Cloud environments (e.g., 12GB RAM instances) without memory overflow, the BRFSS cohort is sub-sampled to 50,000 instances for this repository's demonstration.*

---

## ⚡ Reproducibility Pipeline (Reviewer Guide)

To facilitate a seamless and transparent peer-review process, we provide an automated, **"Zero-Click" Universal Data Pipeline**. 
- No Google Drive mounting required.
- No API keys or credentials needed.
- No manual dataset downloads required.

**Instructions:**
1. Open our official interactive Notebook via Google Colab:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vxQ1wFkJnOshUxagfWYLANFJMttfIX-D)
2. Ensure the Hardware Accelerator is set to **T4 GPU** (`Runtime -> Change runtime type`).
3. Click **`Runtime -> Run All`**.
4. The script will autonomously fetch the raw `.XPT` and `.csv` files from academic mirrors, perform strictly inductive leakage-free graph construction, train the E-GACT architecture, and output the ROC-AUC benchmarks alongside high-resolution Explainable AI (XAI) figures.

---

## 🛠️ Local Installation & Usage

For researchers wishing to clone and run this framework on local workstations or Edge devices:

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the necessary dependencies:
```bash
pip install torch torch-geometric faiss-cpu scikit-learn pandas lightgbm ucimlrepo shap matplotlib seaborn
