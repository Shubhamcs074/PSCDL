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
## Evaluation & Experimental Benchmarks

We evaluate the Unified PSCDL Net on the public PSCD evaluation dataset against dominant vision-only baseline architectures. Performance benchmarks demonstrate a significant margin of improvement across all primary tracking metrics.

### Quantitative Performance Comparison

| Model Architecture | Input Modalities | Mean Precision | Mean Recall | **Mean F1-Score (Primary)** | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| GeSCF | Visual Only | 0.4310 | 0.3732 | 0.4000 | 0.2800 |
| RSCD | Visual Only | 0.5821 | 0.5034 | 0.5400 | 0.4000 |
| C-3PO | Visual Only | 0.6944 | 0.6291 | 0.6600 | 0.5400 |
| **Unified PSCDL Net (Ours)** | **Vision + Language (BERT)** | **0.7853** | **0.7029** | **0.7418 (+8.2%)** | **0.5948** |

---

### Core Performance Metrics

The following composite dashboard breaks down our multi-metric capabilities and isolates the precise F1-Score discrepancies across models:

<table width="100%">
  <tr>
    <td align="center" width="100%">
      <img src="Graphs and Visuals/benchmarks_sota.png" alt="SOTA Benchmark Performance Matrix"/>
      <br>
      <sub><b>Figure 1:</b> Multi-Metric Radar Chart (Left) detailing global profile scaling vs. baseline variants; Leaderboard F1-Score Discrepancy (Right) showing an absolute +8.2% net gain over visual-only frameworks.</sub>
    </td>
  </tr>
</table>

---

### Convergence Profiling & Parameter Efficiency

We trace structural optimization paths and architectural tradeoffs to evaluate deployment feasibility. This isolates system stability alongside accuracy-to-footprint scaling:

<table width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="Graphs and Visuals/training_convergence.png" alt="Model Training Loss Convergence"/>
    </td>
    <td align="center" width="50%">
      <img src="Graphs and Visuals/architectural_efficiency_scatter.png" alt="Pareto Efficiency Scatter Plot"/>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <sub><b>Figure 2:</b> Dual-axis learning curves tracking loss trajectory vs. validation F1 progression (Left), and Pareto Frontier mapping down-stream classification accuracy relative to network parameter footprints (Right).</sub>
    </td>
  </tr>
</table>

---

### Multi-Modal Grounding Verification & Execution Constraints

To validate the multi-modal interaction and real-time operational readiness of the network, we monitor edge activation polarization, prompt attention density, and resolution throughput scaling:

<table width="100%">
  <tr>
    <td align="center" width="33%">
      <img src="Graphs and Visuals/mask_confidence_distribution.png" alt="Prediction Activation Distribution"/>
    </td>
    <td align="center" width="34%">
      <img src="Graphs and Visuals/text_prompt_sensitivity.jpg" alt="Grounded Language Attention Sensitivity Heatmap"/>
    </td>
    <td align="center" width="33%">
      <img src="Graphs and Visuals/hardware_fps_throughput.png" alt="Hardware Throughput FPS Execution Scaling"/>
    </td>
  </tr>
  <tr>
    <td colspan="3" align="center">
      <sub><b>Figure 3:</b> Violin density distributions indicating pixel-level prediction confidence (Left), cross-modal attention matrix isolating text-to-feature matching accuracy (Center), and frame throughput tracking under scaling spatial resolutions (Right).</sub>
    </td>
  </tr>
</table>

---

### Qualitative Boundary Segmentation & Spatial Errors

The spatial mask profile below shows how our model handles localized change detection tasks under complex background conditions:

<table width="100%">
  <tr>
    <td align="center" width="100%">
      <img src="Graphs and Visuals/spatial_error_analysis.png" width="80%" alt="Spatial Segment Confusion Error Analysis Mask"/>
      <br>
      <sub><b>Figure 4:</b> Pixel-level error analysis mask illustrating persistent scene tracking capabilities and boundary alignment stability under ambient environmental noise.</sub>
    </td>
  </tr>
</table>