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

## ⚡ Zero-Click Reproducibility (Reviewer Guide)

We deeply respect the time of academic peer-reviewers. To facilitate a seamless and transparent review process, we provide an automated, **"Zero-Click" Universal Data Pipeline**. 
- No Google Drive mounting required.
- No API keys, credentials, or manual dataset downloads needed.

**Instructions:**
1. Open our official interactive Notebook via Google Colab:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vxQ1wFkJnOshUxagfWYLANFJMttfIX-D)
2. Ensure the Hardware Accelerator is set to **T4 GPU** (`Runtime -> Change runtime type`).
3. Click **`Runtime -> Run All`**.
4. The script will autonomously fetch the raw clinical cohorts, perform strictly inductive leakage-free graph construction, train the E-GACT architecture, and output the ROC-AUC benchmarks alongside high-resolution Explainable AI (XAI) figures.

---

## 📖 Overview

Predicting Type 2 Diabetes Mellitus (T2DM) and associated clinical outcomes from tabular Electronic Health Records (EHR) is critical for early clinical intervention. However, current Deep Tabular Models chronically underperform against tree-based ensembles on noisy clinical datasets, and fail to scale on massive population-level cohorts due to $\mathcal{O}(N^2)$ attention complexities. Furthermore, standard algorithms evaluate patients as Independent and Identically Distributed (I.I.D.) instances, neglecting the fundamental clinical paradigm of **Case-Based Reasoning**.

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

## 🛠️ Local Installation & Usage

For researchers wishing to clone and run this framework on local workstations or Edge devices:

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the necessary dependencies:

```bash
pip install torch torch-geometric faiss-cpu scikit-learn pandas lightgbm ucimlrepo shap matplotlib seaborn
```
### 2. Using E-GACT on Custom Tabular Data
The modular PyTorch design allows for straightforward integration into custom clinical datasets:
```bash
import torch
import torch.nn.functional as F
from models.egact import EGACT, build_faiss_hnsw_graph

# 1. Initialize the E-GACT Model (0.45M Params)
model = EGACT(
    num_cont=15,          # Number of continuous clinical features
    cat_dims=[4, 2, 7],   # Vocabulary sizes of categorical features
    d_model=32,           # Latent dimension size
    n_heads=4             # Attention heads
)

# Dummy Patient Data (Batch of 256 patients)
x_cont = torch.randn(256, 15) 
x_cat = torch.randint(0, 2, (256, 3))

# 2. Intra-Patient Projection (Extract Latent Z)
z_embeddings = model.get_z_embeddings(x_cont, x_cat)

# 3. Inter-Patient Topology (Build Dynamic FAISS Graph in O(N log N))
edge_index = build_faiss_hnsw_graph(z_embeddings, k=5, is_inductive=False)

# 4. GNN Aggregation & Hybrid Prediction
# h: Neighborhood profile, z_scl: Contrastive projection, logits: Final predictions
h, z_scl, logits = model.gnn_forward(z_embeddings, edge_index)
```
---
## 🔍 Clinical Explainability (XAI)

E-GACT strictly avoids the "black-box" paradigm. However, computing SHAP values in a Graph Neural Network is inherently complex due to neighborhood contamination (Message Passing). To resolve this, we introduce the Subgraph Freezing Approach.
The provided pipeline autonomously generates and saves high-resolution PDF graphics for:
t-SNE Latent Space Visualizations: Demonstrating how the Supervised Contrastive Loss (SCL) mathematically forces diabetic (red) and healthy (green) patient profiles into distinct topological manifolds.
SHAP Feature Attributions: Highlighting exactly which physiological factors (e.g., BMI, Age, Prior Admissions) drove a specific patient into the high-risk category, providing clinically actionable insights.

---
## 📝 Citation

If you find this codebase or methodology useful in your research, please consider citing our paper:
```bash
@article{egact_2026,
  title={Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability},
  author={Anonymous Authors},
  journal={IEEE Journal of Biomedical and Health Informatics (Submitted)},
  year={2026}
}
```

