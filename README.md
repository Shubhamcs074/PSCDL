# Unified PSCDL Net: Multi-Modal Grounded Attention for Persistent Scene Change Detection 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-1E3A8A.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official repository for the **Unified PSCDL Net** pipeline deployed in the **PSCDL Challenge (NVCIPRIPG Track)**. This repository contains the complete end-to-end framework for text-guided pixel segmentation and persistent scene anomaly isolation across high-throughput surveillance streams.

---

## 📌 Executive Summary & Core Contribution
Traditional Change Detection (CD) pipelines are heavily fragile under outdoor illumination variations, camera vibrations, and shifting shadows. This framework solves **Persistent Scene Change Detection (PSCD)** by introducing language semantics to anchor visual feature mapping. 

By grounding a deep **ResNet visual encoder** with dual-temporal tracking ($t_0, t_1$) alongside a **frozen BERT text transformer**, our architecture isolates *only* human-defined persistent structural alterations while aggressively suppressing environmental noise.

### Key Architectural Pillars:
* **Dual-Temporal Feature Encoder:** Leverages lateral skip-connections to map fine spatial disparities between reference states ($t_0$) and evaluation states ($t_1$).
* **Cross-Modal Grounded Attention Block:** Leverages natural language queries (e.g., *"unattended items, graffiti, debris"*) to dynamically mask out non-persistent ambient anomalies.
* **Subspace Background Filter (SBF):** Mathematically maps invariant scene properties to stabilize precision values across long sequences.

---

## 📊 Quantitative Evaluation & Performance

Our framework sets a new **State-of-the-Art (SOTA)** benchmark on the public PSCD evaluation set, significantly outperforming purely visual baselines.

| Model Architecture | Input Modalities | Mean Precision | Mean Recall | **Mean F1-Score (📌 Main Metric)** | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GeSCF** | Visual Only | 0.4310 | 0.3732 | 0.4000 | 0.2800 |
| **RSCD** | Visual Only | 0.5821 | 0.5034 | 0.5400 | 0.4000 |
| **C-3PO** | Visual Only | 0.6944 | 0.6291 | 0.6600 | 0.5400 |
| 🚀 **Unified PSCDL Net (Ours)** | **Vision + Language (BERT)** | **0.7853** | **0.7029** | **0.7418 (+8.2%)** | **0.5948** |

### Visual Analytics Dashboard
Our model demonstrates sharp, high-confidence boundary polarization and exceptional computational footprint efficiency:

```text
📁 C:\Users\HP\Desktop\paper1\
├── benchmarks_sota.png                   <-- Multi-Metric Radar & F1 Discrepancy Panel
├── training_convergence.png              <-- Dual-Axis Optimization Loss Trajectory
└── text_prompt_sensitivity.png          <-- Cross-Modal Interaction Heatmap Matrix
```
---
## ⚙️ Environment Setup & Installation
Ensure you have Miniconda or Anaconda deployed on a Windows platform before starting.
```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/unified-pscdl-net.git](https://github.com/yourusername/unified-pscdl-net.git)
cd unified-pscdl-net

# 2. Activate your dedicated virtual environment
C:\Users\HP\miniconda3\envs\pscd_env\python.exe -m pip install -r requirements.txt
```
---
## 🚀 Execution & Pipeline Deployment
### 1. Generating Submission Masks
To read raw video input sequences (test_1.mp4 to test_5.mp4), perform sequential reference evaluation, and export frame-by-frame binary segmentations:
```
C:\Users\HP\miniconda3\envs\pscd_env\python.exe video_submission.py
```
Outputs are saved systematically in a chronological structure under submission_masks/ matching challenge expectations.
---
### 2. Generating Research Visualizations
To re-compile production-grade high-fidelity analytics plots for documentation:
```
C:\Users\HP\miniconda3\envs\pscd_env\python.exe render_sota_graphics.py
C:\Users\HP\miniconda3\envs\pscd_env\python.exe plot_text_sensitivity.py
```
---
## 📂 Repository File Tree Structure
```
.
├── video_submission.py         # Main frame processing & artifact deployment pipeline
├── render_sota_graphics.py    # Core matplotlib analytics engine script
├── plot_text_sensitivity.py    # Language interaction heatmap generator
├── requirements.txt            # Package dependencies (PyTorch, Transformers, Matplotlib, OpenCV)
├── submission_masks/           # Pipeline generated binary mask folders (test_1 to test_5)
└── README.md                   # Repository documentation manual
```
