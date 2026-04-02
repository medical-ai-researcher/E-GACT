<h1 align="center">
  🚀 Resource-Efficient Graph-Aware Contrastive Transformer <br>
  (E-GACT)
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

Predicting Type 2 Diabetes Mellitus (T2DM) from tabular Electronic Health Records (EHR) is critical for early clinical intervention. However, current Deep Tabular Models (e.g., TabNet, FT-Transformer) chronically underperform against tree-based ensembles on noisy clinical datasets, and fail to scale on massive population-level cohorts due to $\mathcal{O}(N^2)$ attention complexities. Furthermore, they evaluate patients as Independent and Identically Distributed (I.I.D.) instances, completely ignoring the fundamental clinical practice of **Case-Based Reasoning**.

**E-GACT** solves these bottlenecks mathematically by integrating:
1. **Lightweight Tabular Transformer:** For non-linear, intra-patient feature projection.
2. **FAISS $k$-NN Graph Neural Network (GNN):** For inter-patient topological similarities (Case-Based Reasoning) dynamically constructed in $\mathcal{O}(N \log N)$ time.
3. **Supervised Contrastive Learning (SCL):** To actively organize the topological space and prevent over-smoothing against severe class imbalances.

🔥 **Edge AI Ready:** With only **0.45M learnable parameters**, E-GACT operates with $<45$ ms inference latency on standard clinical microprocessors, making it highly suitable for zero-latency, privacy-preserving local Edge AI deployments in primary care.

---

## 🏗️ Architecture

<p align="center">
  <img src="E-GACT Architecture.jpg" width="95%">
  <br><em>Fig 1: Overall workflow of the strictly inductive E-GACT architecture.</em>
</p>

---

## 📊 Benchmarked Datasets
To prove algorithmic robustness and scalability across varying modalities, E-GACT is evaluated on three globally validated, open-access cohorts:

| Dataset | Modality | Size (Patients) | Focus Area | Target Prediction | E-GACT (ROC-AUC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NHANES (2017-2018)** | Clinical Lab + Demographics | ~6,000 | Physiological Signals | T2DM (HbA1c $\geq$ 6.5) | **0.816** |
| **130-US Hospitals** | Electronic Health Records (EHR)| ~101,000 | Case-Based Reasoning | Readmission Risk | **0.662** |
| **CDC BRFSS (2015)** | Population Survey | ~50,000* | Edge AI Scalability | T2DM (Imbalanced) | **0.832** |

*\*Note: BRFSS is sub-sampled to 50k to ensure stable execution within standard free-tier Cloud environments (e.g., Google Colab 12GB RAM) without Out-Of-Memory crashes.*

---

## ⚡ Zero-Click Reproducibility (For Peer-Reviewers)

We deeply respect the time of academic reviewers. We have designed a **"Zero-Click" Universal Data Pipeline**. 
- ❌ No Google Drive mounting required.
- ❌ No Kaggle API keys or passwords required.
- ❌ No manual file downloads required.

**How to Test the Code:**
1. Open our official interactive Notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vxQ1wFkJnOshUxagfWYLANFJMttfIX-D)

3. Click **`Runtime -> Run All`**.
4. The code will automatically fetch the raw `.XPT` and `.csv` files from public academic mirrors, strictly perform inductive leakage-free graph constructions, train the E-GACT architecture, and output the ROC-AUC benchmarks shown in **Table 1** of our paper.

*(Note to Reviewers: Please ensure Hardware Accelerator is set to **T4 GPU** in Colab for optimal execution speed).*

---

## 🛠️ Local Installation & Usage

If you wish to clone and run this repository on your local workstation or Edge device:

### 1. Requirements
```bash
pip install torch torch-geometric faiss-cpu scikit-learn pandas lightgbm ucimlrepo shap
2. Using E-GACT on Your Own Custom Tabular Data
The modular PyTorch design allows seamless integration into your own clinical projects:
code
Python
import torch
import torch.nn.functional as F
from models.egact import EGACT, build_faiss_hnsw_graph

# Initialize Model (0.45M Params)
model = EGACT(
    num_cont=15,          # Number of continuous clinical features
    cat_dims=[4, 2, 7],   # Vocabulary sizes of categorical features
    d_model=32,           # Latent dimension size
    n_heads=4             # Attention heads
)

# Dummy Patient Data
x_cont = torch.randn(256, 15) # Batch of 256 patients
x_cat = torch.randint(0, 2, (256, 3))

# 1. Intra-Patient Projection (Extract Z)
z_embeddings = model.get_z_embeddings(x_cont, x_cat)

# 2. Inter-Patient Topology (Build Dynamic FAISS Graph in O(N log N))
edge_index = build_faiss_hnsw_graph(z_embeddings, k=5, is_inductive=False)

# 3. GNN Aggregation & Prediction
# h: Neighborhood profile, z_scl: Contrastive projection, logits: Final predictions
h, z_scl, logits = model.gnn_forward(z_embeddings, edge_index)
🧠 Clinical Explainability (XAI)
E-GACT moves beyond purely statistical black-boxes. To compute SHAP values in a Graph Neural Network without neighborhood contamination, we introduce the Subgraph Freezing Approach.
The repository's pipeline automatically outputs high-resolution PDF graphics for:
t-SNE Latent Space Visualizations: Demonstrating how SCL forces diabetic (red) and healthy (green) patients into distinct topological clusters.
SHAP Feature Attributions: Highlighting exactly which physiological factors (e.g., BMI, Age) pushed a specific patient into the high-risk category.
📝 Citation
If you find this code or methodology useful in your research, please consider citing our paper:
code
Bibtex
@article{egact_2026,
  title={Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability},
  author={Anonymous Authors},
  journal={IEEE Journal of Biomedical and Health Informatics (Submitted)},
  year={2026}
}
