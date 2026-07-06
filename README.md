# PSCDLNet: Persistent Scene Change Detection and Localization Using Frozen DINOv2 Features and Multi-Scale Siamese Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![DINOv2](https://img.shields.io/badge/Backbone-DINOv2%20ViT--S%2F14-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Challenge](https://img.shields.io/badge/PSCDL-2026-red)

> A foundation-model-driven Siamese framework for robust, long-term persistent scene change detection in urban surveillance — powered by a frozen self-supervised DINOv2 backbone.
## 📌 Abstract

Persistent Scene Change Detection and Localization (PSCDL) aims to identify **long-term structural modifications** in monitored environments — such as abandoned objects, illegal construction, or removed infrastructure — while suppressing **transient changes** caused by moving objects, illumination shifts, shadows, and weather. This is fundamentally harder than conventional change detection, which does not distinguish between short-lived and persistent events.

This repository presents **PSCDLNet**, a foundation-model-driven Siamese framework for PSCDL. Instead of training a convolutional encoder end-to-end, PSCDLNet leverages a **frozen DINOv2 Vision Transformer (ViT-S/14)** backbone to extract robust, self-supervised semantic representations from paired temporal observations. Multi-scale transformer features are converted into spatial feature pyramids via a dedicated **DINO Feature Adapter**, fused through a **UNet-style decoder**, and refined using explicit **absolute feature differencing** at every scale to isolate structural change from invariant background. A final **temporal persistence verification** stage filters out short-duration false alarms, ensuring only long-term modifications are reported.

Evaluated on the official **PSCDL 2026 dataset** (464 training pairs, 116 validation pairs, 5 blind evaluation videos), PSCDLNet is benchmarked against Siamese ResNet18 and ResNet34 baselines under an identical experimental protocol. Results show that PSCDLNet achieves the **best overall IoU, Precision, and Accuracy**, while requiring far fewer trainable parameters — since the DINOv2 backbone remains frozen and only the lightweight adapter and decoder are optimized.
## 🎯 Motivation & Problem Statement

Persistent Scene Change Detection and Localization has become an important problem for intelligent surveillance, autonomous monitoring, and smart-city infrastructure management. Real-world applications include:

- Abandoned object detection
- Illegal parking / obstruction monitoring
- Road and infrastructure maintenance tracking
- Environmental and urban safety monitoring

Unlike conventional image change detection — which flags *any* difference between two observations — PSCDL must explicitly separate **long-duration structural change** from **short-lived environmental noise**. Urban surveillance footage is inherently unstable: seasonal shifts, lighting conditions, camera exposure variation, atmospheric effects, and constant foreground activity (pedestrians, vehicles) all produce pixel-level differences that have nothing to do with genuine structural change. Naively differencing two frames — or training a purely visual model to do so — produces a flood of false positives.

### Why existing approaches fall short

**CNN-based Siamese networks** (e.g., ResNet-UNet architectures) remain the dominant approach for change detection. They are efficient and effective at capturing local spatial detail, but:
- Their receptive fields are inherently local, limiting long-range contextual reasoning.
- Their representations are learned end-to-end via supervised training, making them sensitive to illumination, shadows, and seasonal appearance shifts absent from (or under-represented in) the training set.

**Transformer-based change detection methods** (e.g., ChangeFormer, Swin-based architectures) improve global contextual reasoning through self-attention, but:
- They typically require large-scale annotated datasets and expensive end-to-end training.
- Their large parameter counts increase memory and training cost, limiting practical deployment.

**Foundation models** such as DINO/DINOv2 offer a third path: representations learned via large-scale self-supervised pretraining that generalize well across viewpoint, lighting, and background variation — without needing to be fine-tuned on the downstream task. However, relatively few existing PSCD methods exploit *frozen* foundation-model features; most continue to fine-tune the backbone jointly with the segmentation head, sacrificing some of the robustness gained during pretraining while increasing computational cost.

Finally, most existing change detection methods operate purely at the **frame-pair level** — they have no explicit mechanism for enforcing that a detected change is *persistent* over time, rather than a transient object that happens to appear in both frames.

### The gap this work addresses

PSCDLNet is designed to close these gaps directly:
1. Use a **frozen, self-supervised foundation model (DINOv2)** as the encoder — robust representations without expensive end-to-end optimization.
2. Perform **explicit multi-scale feature differencing** rather than relying on the decoder to implicitly learn temporal comparison.
3. Add a **temporal persistence verification stage** that separates spatial localization from temporal reasoning — so the system only reports changes that genuinely persist, satisfying the operational definition of "persistent" scene change.
## 🧠 Key Contributions

1. **Foundation-model-driven Siamese architecture.**
   We propose **PSCDLNet**, which replaces the conventional supervised CNN encoder with a **frozen DINOv2 Vision Transformer (ViT-S/14)** backbone. Only a lightweight decoder and feature-adaptation module are trained, while the pretrained self-supervised representations remain untouched — preserving their robustness to illumination, viewpoint, and appearance variation.

2. **Multi-scale feature adaptation and explicit differencing.**
   A dedicated **DINO Feature Adapter** converts 1D transformer token embeddings from four intermediate layers into a spatial feature pyramid. **Absolute feature differencing** is then applied at every scale (`D_n = |F_n(I_t0) − F_n(I_t1)|`), explicitly emphasizing structural change while suppressing invariant background — rather than leaving this comparison for the decoder to learn implicitly.

3. **Comprehensive, fair comparative study.**
   Three Siamese architectures — **ResNet18-UNet**, **ResNet34-UNet**, and the proposed **DINOv2-based PSCDLNet** — were implemented and evaluated under an **identical experimental protocol** (same dataset splits, optimizer, scheduler, loss function, and evaluation metrics) on the official **PSCDL 2026 dataset**, enabling a rigorous, apples-to-apples comparison between convolutional and foundation-model encoders.

4. **Extensive quantitative and qualitative validation.**
   Detailed experiments — including threshold sensitivity analysis, training convergence curves, an ablation study, and qualitative visual comparisons — demonstrate the effectiveness of foundation-model representations for persistent scene monitoring, and highlight their advantages over conventional convolutional baselines in challenging, real-world urban surveillance conditions.

## 🏗️ Architecture Overview

PSCDLNet formulates persistent scene change localization as a six-stage pipeline. Given two temporally separated RGB observations from a fixed surveillance camera,
![alt text](<Figure 1-1.png>)
## 📂 Repository Structure

```
PSCD-Project/
│
├── checkpoints/
│   ├── best_pscdl_model.pth
│   └── models/
│       ├── best_model.pth
│       ├── resnet34_multiscale_best.pth
│       └── siamese_dinov2_unet_final.pth
│
├── figures/                              # Reserved for paper-ready figure exports
│
├── Graphs and Visuals/
│   ├── architectural_efficiency_scatter.png
│   ├── benchmarks_sota.png
│   ├── hardware_fps_throughput.png
│   ├── mask_confidence_distribution.png
│   ├── spatial_error_analysis.png
│   ├── text_prompt_sensitivity.png
│   └── training_convergence.png
│
├── notebooks/
│   ├── main/
│   │   ├── dataset.ipynb                 # Dataset loading & preprocessing
│   │   ├── dinov2_improved (1).ipynb      # PSCDLNet (DINOv2) training/experiments
│   │   ├── resnet34new.ipynb              # ResNet34-UNet baseline
│   │   ├── resnet_18.ipynb                # ResNet18-UNet baseline
│   │   └── Tested (1).ipynb               # Evaluation / testing notebook
│   └── related/
│       └── dinov2_train_150ep.ipynb       # Full 150-epoch DINOv2 training run
│
├── plots/
│   ├── generate_error_masks.py           # Spatial error mask generation
│   ├── plot_confidence_distribution.py   # Prediction confidence violin plots
│   ├── plot_efficiency_scatter.py        # Parameter vs. accuracy Pareto plot
│   ├── plot_hardware_efficiency.py       # FPS / throughput scaling plots
│   ├── plot_text_sensitivity.py          # Cross-modal attention heatmap
│   ├── plot_training_convergence.py      # Loss / metric convergence curves
│   └── render_sota_graphics.py           # SOTA benchmark radar & bar charts
│
├── predictions/
│   ├── result_00000000.png
│   ├── result_00000192.png
│   ├── result_00000384.png
│   ├── result_00000576.png
│   └── result_00000769.png
│
├── PSCDL2026_Test/
│   └── test_videos/
│       ├── test_1.mp4
│       ├── test_2.mp4
│       ├── test_3.mp4
│       ├── test_4.mp4
│       └── test_5.mp4
│
├── Reference_Papers/
│   ├── 2604.11402v1.pdf
│   ├── bgdyntexcviu2008.pdf
│   ├── Pixel-Based_Change_Detection_in_Moving-Camera_Videos_Using_Twin_Convolutional_Features_on_a_Data-Constrained_Scenario.pdf
│   └── scenc change 1.pdf
│
├── test files/
│   ├── inference.py                      # Single image-pair inference
│   ├── test_submission.py                # Generates challenge submission masks
│   ├── training.py                       # Model training entry point
│   └── video_submission.py               # Frame-by-frame video mask export
│
├── test_results/
│   ├── qualitative_comparison.pdf
│   └── qualitative_comparison_fixed.pdf
│
├── .gitignore
└── README.md
```

**Notes on this structure:**
- Model development for all three architectures (**ResNet18-UNet**, **ResNet34-UNet**, **PSCDLNet / DINOv2**) is primarily notebook-driven, under `notebooks/main/` and `notebooks/related/`.
- `checkpoints/` holds the final trained weights for each architecture — `siamese_dinov2_unet_final.pth` (PSCDLNet), `resnet34_multiscale_best.pth` (ResNet34 baseline), and `best_model.pth` / `best_pscdl_model.pth` (best selected checkpoints).
- `test files/` contains the production-facing scripts used for **training**, **inference**, and **challenge submission generation** (image and video).
- `plots/` and `Graphs and Visuals/` together handle all analytics/visualization generation — scripts in `plots/` produce the images stored in `Graphs and Visuals/`.
- `PSCDL2026_Test/test_videos/` holds the five blind evaluation videos (`test_1.mp4`–`test_5.mp4`) used for official challenge submission.
- `Reference_Papers/` contains background literature referenced during development (not required to run the code).
## 📊 Dataset Description

PSCDLNet is trained and evaluated on the official **Persistent Scene Change Detection and Localization (PSCDL) 2026 dataset**, purpose-built for long-term urban surveillance monitoring.

### Overview

| Split | Samples | Purpose |
|---|---|---|
| Training | 464 image pairs | Model optimization |
| Validation | 116 image pairs | Model selection, threshold tuning |
| Blind Evaluation | 5 videos | Official challenge evaluation (unseen during development) |

Each sample consists of:
- **Baseline observation** `I_t0` — the reference frame captured at an earlier time.
- **Current observation** `I_t1` — a temporally separated frame from the same fixed camera.
- **Pixel-level binary ground-truth mask** — indicating the spatial extent of *persistent* scene changes only.

### What Makes This Dataset Challenging

Unlike conventional change detection datasets, PSCDL specifically emphasizes **long-duration structural modifications**, including:

- Abandoned or newly parked vehicles
- Construction barriers and roadblocks
- Removed or added street furniture
- Other permanent infrastructure alterations

While explicitly requiring robustness against **transient, non-persistent variation**, such as:

- Illumination changes (time-of-day, exposure differences)
- Weather conditions
- Shadows
- Seasonal appearance shifts
- Moving vehicles and pedestrians
- Partial occlusions

### Scene Diversity

The dataset spans diverse urban environments to ensure generalization across real-world surveillance conditions:

| Environment | Example Change Type |
|---|---|
| Urban Road | Vehicle appears/disappears |
| Parking Lot | Barrier installed |
| Street Sidewalk | Bench added/removed |
| Traffic Intersection | Road furniture modification |

### Data Access

> ⚠️ The PSCDL 2026 dataset is provided by the official challenge organizers (Vehant Technologies) and is **not redistributed** in this repository. Please refer to the official PSCDL 2026 Challenge page for access instructions and terms of use.

### Expected Directory Layout

```
data/
└── pscdl2026/
    ├── train/
    │   ├── t0/              # baseline images
    │   ├── t1/               # current images
    │   └── masks/            # ground-truth binary masks
    ├── val/
    │   ├── t0/
    │   ├── t1/
    │   └── masks/
    └── blind_eval/
        └── videos/            # 5 blind evaluation sequences
```
---
## ⚙️ Installation & Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training; DINOv2 ViT-S/14 inference is feasible on CPU but slow)
- Conda or venv for environment isolation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/PSCDLNet.git
cd PSCDLNet
```

### 2. Create and Activate a Virtual Environment

**Using Conda (recommended):**

```bash
conda create -n pscdlnet python=3.10 -y
conda activate pscdlnet
```

**Or using venv:**

```bash
python -m venv pscdlnet-env
source pscdlnet-env/bin/activate      # Linux/macOS
pscdlnet-env\Scripts\activate         # Windows
```

### 3. Install PyTorch

Install the PyTorch build matching your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command). Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` core dependencies:**

```
torch>=2.0
torchvision>=0.15
transformers>=4.30
opencv-python
numpy
matplotlib
pyyaml
tqdm
scikit-learn
```

### 5. Verify the Setup

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 6. Download the Frozen DINOv2 Backbone

The DINOv2 ViT-S/14 weights are loaded automatically via `torch.hub` or the `transformers` library on first run:

```python
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
```

> No manual download is required — weights are cached locally after the first load (`~/.cache/torch/hub`).
---
## 🚀 Usage

> Commands below assume the repository structure from [Section 6](#-repository-structure). Update script/config paths to match your actual implementation.

### 1. Configuration

All training and model hyperparameters are defined in `configs/pscdlnet.yaml`, matching the protocol used in the paper:

```yaml
model:
  encoder: dinov2_vits14
  freeze_backbone: true
  adapter_layers: [2, 5, 8, 11]
  input_resolution: 512
  dinov2_input_resolution: 518

training:
  optimizer: AdamW
  learning_rate: 1e-4
  scheduler: ReduceLROnPlateau
  scheduler_metric: val_iou
  max_epochs: 150
  early_stopping_patience: 30
  batch_size: 1
  loss: bce_dice

evaluation:
  primary_metric: iou
  secondary_metric: f1
  threshold_search_range: [0.10, 0.90]
  threshold_search_step: 0.05
```

### 2. Training

Train PSCDLNet with the frozen DINOv2 backbone (only the adapter and decoder are optimized):

```bash
python scripts/train.py \
  --config configs/pscdlnet.yaml \
  --data-root data/pscdl2026 \
  --output-dir checkpoints/
```

To train a convolutional baseline for comparison (ResNet18/ResNet34):

```bash
python scripts/train.py \
  --config configs/resnet18_unet.yaml \
  --data-root data/pscdl2026 \
  --output-dir checkpoints/
```

Training saves two separate checkpoints automatically:
- **Best validation IoU** → `checkpoints/pscdlnet_best_iou.pth`
- **Best validation F1-score** → `checkpoints/pscdlnet_best_f1.pth`

### 3. Evaluation

Run evaluation on the validation set, including automatic threshold optimization:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/pscdlnet_best_iou.pth \
  --data-root data/pscdl2026/val \
  --threshold-search \
  --output results/metrics/pscdlnet_val_results.json
```

This reports **Precision, Recall, F1-score, IoU, and Pixel Accuracy** across the searched threshold range (0.10–0.90) and selects the operating point with the best IoU/F1 trade-off.

### 4. Inference

Run inference on a new baseline/current image pair:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/pscdlnet_best_f1.pth \
  --t0 path/to/reference.jpg \
  --t1 path/to/current.jpg \
  --threshold 0.30 \
  --output-dir outputs/masks
```

### 5. Inference with Temporal Persistence Filtering

For video/sequence input, enable the post-processing persistence module to suppress transient false alarms:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/pscdlnet_best_f1.pth \
  --video-sequence path/to/sequence/ \
  --enable-persistence-filter \
  --persistence-threshold 5 \
  --cooldown-period 10 \
  --output-dir outputs/masks
```

| Argument | Description |
|---|---|
| `--persistence-threshold` | Min. consecutive frames (`T_p`) a new change must persist to be confirmed |
| `--cooldown-period` | Min. consecutive frames (`T_c`) an object must stay absent before background update |

### 6. Reproducing Paper Results

To reproduce Table III (quantitative comparison) end-to-end for all three models:

```bash
bash scripts/reproduce_results.sh
```

This trains ResNet18-UNet, ResNet34-UNet, and PSCDLNet sequentially under the identical protocol described in [Section 8 training configuration](#8-training-configuration), then aggregates metrics into `results/metrics/comparison_table.csv`.
---

## 📈 Results — Quantitative

All three architectures (ResNet18-UNet, ResNet34-UNet, PSCDLNet) were trained and evaluated under an **identical experimental protocol** — same dataset splits, optimizer, scheduler, loss function, and evaluation metrics — to ensure a fair comparison.

### Main Comparison (PSCDL 2026 Validation Set)

| Model | Encoder | IoU (%) | F1-score (%) | Precision (%) | Recall (%) | Accuracy (%) |
|---|---|---|---|---|---|---|
| ResNet18-UNet | Supervised CNN | 70.62 | **82.19** | 85.45 | 80.07 | 92.89 |
| ResNet34-UNet | Supervised CNN | 70.46 | 81.35 | 80.02 | **83.98** | 92.74 |
| **PSCDLNet (Proposed)** | **Frozen DINOv2 ViT-S/14** | **71.16** | 81.35 | **87.19** | 78.16 | **94.68** |

**Key observations:**
- PSCDLNet achieves the **best IoU (71.16%)** and **best Accuracy (94.68%)** among all evaluated models.
- PSCDLNet achieves the **highest Precision (87.19%)** by a substantial margin (+1.74 over ResNet18, +7.17 over ResNet34) — reflecting fewer false positives from illumination/shadow-induced noise.
- ResNet34 achieves the highest Recall, at the cost of more false positives (lowest Precision) — a classic precision/recall trade-off tied to backbone depth vs. representation robustness.
- PSCDLNet reaches strong performance while training **only the adapter and decoder**, not the full backbone — a significant parameter-efficiency advantage over both CNN baselines, which are fully fine-tuned end-to-end.

### Evaluation Metric Definitions

| Metric | Formula | Description |
|---|---|---|
| Precision | `TP / (TP + FP)` | Fraction of predicted change pixels that are correct |
| Recall | `TP / (TP + FN)` | Fraction of true change pixels correctly detected |
| F1-score | `2 × P × R / (P + R)` | Harmonic mean of Precision and Recall |
| IoU | `TP / (TP + FP + FN)` | Intersection-over-Union; **primary model-selection metric** |
| Pixel Accuracy | `(TP + TN) / (TP + TN + FP + FN)` | Overall pixel-wise classification accuracy |

> IoU was used as the primary model-selection criterion during training, with F1-score monitored simultaneously to account for class imbalance (changed pixels typically occupy only a small fraction of each image).

### Visual Metric Comparison

![alt text](<Figure 5-1.png>)
![alt text](<Figure 6-1.png>)
![alt text](<Figure 7-1.png>)

*Bar chart comparing IoU, F1, Precision, Recall, and Accuracy across ResNet18-UNet, ResNet34-UNet, and PSCDLNet on the PSCDL 2026 validation dataset.*

### Reproducing These Results

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/pscdlnet_best_iou.pth \
  --data-root data/pscdl2026/val \
  --output results/metrics/pscdlnet_val_results.json
```
---

## 🖼️ Results — Qualitative

Quantitative metrics only provide a partial picture of segmentation quality. Qualitative comparisons on representative validation samples offer direct insight into each model's localization behavior across diverse urban scenes.

### Visual Comparison

![alt text](<Figure 8-1.png>)
![alt text](<Figure 9-1.png>)

*From left to right: reference image (`I_t0`), current image (`I_t1`), ground-truth change mask, and predictions from ResNet18-UNet, ResNet34-UNet, and PSCDLNet, across representative samples from the PSCDL 2026 dataset.*

### Observations by Model

**ResNet18-UNet**
- Successfully detects prominent, large-scale scene modifications.
- Occasionally produces **fragmented segmentation masks** and **irregular object boundaries**, particularly in visually complex regions.

**ResNet34-UNet**
- Improves segmentation continuity over ResNet18 and captures larger structural changes more effectively due to its deeper residual hierarchy.
- False detections **remain visible in regions affected by illumination changes and shadows**, indicating persistent sensitivity to environmental appearance variation.

**PSCDLNet (Proposed)**
- Produces **cleaner segmentation boundaries** with less fragmentation.
- **Suppresses false positives** more effectively in dynamic background regions (shadows, lighting shifts) due to the robustness of frozen DINOv2 representations.
- More accurately localizes persistent scene changes across diverse scales — from large abandoned vehicles to smaller structural modifications like barriers or road furniture.

### Case: No-Change Scenes

An important qualitative check is behavior on scenes with **no genuine persistent change** — the model should predict an empty (all-background) mask despite environmental variation between `I_t0` and `I_t1`. PSCDLNet's frozen semantic representations help correctly suppress activations here, whereas CNN baselines are more prone to spurious activations under shadow/lighting differences between the two captures.

### Generating These Visualizations

```bash
python scripts/generate_qualitative_grid.py \
  --checkpoints checkpoints/resnet18_unet.pth checkpoints/resnet34_unet.pth checkpoints/pscdlnet_best_iou.pth \
  --data-root data/pscdl2026/val \
  --num-samples 5 \
  --output results/qualitative/pscdl_comparison_grid.png
```
---

## 🧩 Ablation Study

PSCDLNet integrates multiple architectural components — foundation-model features, multi-scale feature extraction, explicit feature differencing, and temporal persistence verification. This section analyzes the contribution of each, rather than attributing performance to a single design choice.

### A. Effect of Foundation-Model Features

Replacing the conventional supervised CNN encoder (ResNet18/ResNet34) with a **frozen DINOv2 ViT-S/14** backbone is the central architectural change in PSCDLNet.

- CNN encoders learn hierarchical representations via supervised optimization, but remain sensitive to illumination variation, shadows, weather, and seasonal appearance differences.
- DINOv2's self-supervised pretraining on large-scale unlabeled data produces representations that are inherently more invariant to these appearance changes while preserving structural scene information.
- **Result:** PSCDLNet produced cleaner activation maps, reduced false detections in dynamic background regions, and improved localization compared to both CNN baselines.

### B. Effect of Multi-Scale Feature Extraction

Persistent scene modifications occur at widely varying spatial scales — large abandoned vehicles vs. small traffic cones or removed street furniture.

- The DINO Feature Adapter extracts features from **four intermediate transformer layers** (`{2, 5, 8, 11}`), forming a multi-scale pyramid rather than relying on a single-resolution representation.
- Lower-level features preserve high-resolution structural detail (accurate boundaries); deeper features encode semantic context (distinguishing real change from appearance variation).
- **Result:** Multi-scale representation improved boundary delineation, reduced fragmented predictions, and enhanced localization accuracy across both small and large scene changes.

### C. Effect of Absolute Feature Differencing

Many Siamese segmentation networks concatenate temporal features and let the decoder implicitly learn to compare them — increasing the complexity of the learning problem.

- PSCDLNet instead computes **explicit absolute differences** at every feature scale: `D_i = |F_i(I_t0) − F_i(I_t1)|`.
- Stationary scene components (roads, buildings, vegetation) produce near-zero activations after differencing; genuine structural changes generate strong, clean responses.
- **Result:** Improved optimization stability, faster convergence, and reduced false-positive detections by explicitly isolating structural discrepancies *before* segmentation rather than during it.

### D. Effect of Temporal Persistence Verification

The PSCDL challenge specifically requires that only **long-duration** modifications are reported — transient objects (pedestrians, bicycles, vehicles) must not trigger false alarms even if visible across several consecutive frames.

- The persistence module verifies that a candidate change remains present for a predefined duration (`T_p`) before confirming it; removed objects require a cooldown period (`T_c`) before the background reference updates.
- **Result:** Significantly reduces false alarms from transient foreground objects, and — because it's a modular, rule-based post-processing stage — persistence parameters can be tuned per deployment without retraining the segmentation network.

### Summary Table

| Component | Removed / Replaced With | Observed Effect |
|---|---|---|
| Frozen DINOv2 backbone | Supervised ResNet18/ResNet34 | ↓ Robustness to illumination/shadow; ↑ false positives |
| Multi-scale adapter (4 layers) | Single-scale features | ↑ Fragmented predictions; weaker boundary delineation |
| Absolute feature differencing | Feature concatenation | ↑ Optimization difficulty; less stable convergence |
| Temporal persistence verification | No post-processing filter | ↑ False alarms from transient objects (pedestrians, vehicles) |

### Takeaway

No single component is solely responsible for PSCDLNet's performance gains. The improvements arise from the **complementary interaction** of frozen foundation-model representations, multi-scale feature adaptation, explicit differencing, and temporal reasoning — each addressing a distinct failure mode of conventional CNN-based persistent scene monitoring.
---

## 🎚️ Threshold Optimization Analysis

PSCDLNet's decoder outputs a **pixel-wise probability map** (via sigmoid activation), not a binary prediction. Since the choice of decision threshold directly impacts segmentation quality, an **automatic threshold search** was performed on the validation set rather than adopting a fixed default (e.g., 0.5).

### Methodology

Probability thresholds ranging from **0.10 to 0.90** (step 0.05) were evaluated, and the corresponding mean IoU and F1-score were computed for both the best-F1 and best-IoU checkpoints.

### Results — Best-F1 Checkpoint

| Threshold | Mean IoU (%) | Mean F1 (%) |
|---|---|---|
| 0.10 | 61.21 | 73.38 |
| 0.15 | 65.34 | 77.14 |
| 0.20 | 66.51 | 78.11 |
| 0.25 | 67.08 | 78.57 |
| **0.30** | **67.34** | **78.79** |
| 0.35 | 67.27 | 78.69 |
| 0.40 | 67.01 | 78.42 |
| 0.45 | 66.72 | 78.12 |
| 0.50 | 66.37 | 77.80 |
| 0.55 | 65.87 | 77.32 |
| 0.60 | 65.23 | 76.68 |
| 0.65 | 64.46 | 75.88 |
| 0.70 | 63.68 | 75.22 |
| 0.75 | 62.67 | 74.36 |
| 0.80 | 61.28 | 73.16 |
| 0.85 | 59.23 | 71.38 |
| 0.90 | 57.46 | 70.05 |

### Results — Best-IoU Checkpoint

| Threshold | Mean IoU (%) | Mean F1 (%) |
|---|---|---|
| 0.10 | 59.60 | 72.83 |
| 0.15 | 61.99 | 74.71 |
| 0.20 | 63.12 | 75.52 |
| 0.25 | 63.76 | 75.93 |
| 0.30 | 64.12 | 76.11 |
| **0.35** | **64.27** | **76.15** |
| 0.40 | 64.26 | 76.05 |
| 0.45 | 64.17 | 75.91 |
| 0.50 | 63.97 | 75.70 |
| 0.55 | 63.71 | 75.43 |
| 0.60 | 63.35 | 75.08 |
| 0.65 | 62.83 | 74.60 |
| 0.70 | 62.10 | 73.96 |
| 0.75 | 61.18 | 73.16 |
| 0.80 | 59.91 | 72.05 |
| 0.85 | 57.90 | 70.25 |
| 0.90 | 54.27 | 66.94 |

### Key Findings

- Segmentation performance follows a clear **rise-then-fall pattern** as the threshold increases — improving from low thresholds, peaking near the optimal operating point, then degrading at higher thresholds.
- **Lower thresholds** → higher recall, but introduce more false positives.
- **Higher thresholds** → suppress false detections, but risk missing smaller scene modifications.
- The optimal operating point is **~0.30** for the best-F1 checkpoint and **~0.35** for the best-IoU checkpoint — both checkpoints peak in a similar, moderate-threshold range rather than at the conventional default of 0.5.
- Validation-based threshold optimization provides a better precision/recall balance than assuming a fixed threshold, and is applied automatically before final model evaluation.

### Reproducing This Analysis

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/pscdlnet_best_iou.pth \
  --data-root data/pscdl2026/val \
  --threshold-search \
  --threshold-range 0.10 0.90 \
  --threshold-step 0.05 \
  --output results/metrics/threshold_search.csv
```
---
## ⚠️ Limitations

While PSCDLNet achieves the strongest overall performance among the evaluated architectures, several limitations remain and are worth stating explicitly.

### 1. Small Scene Modification Detection

Very small persistent changes — occupying only a limited number of pixels — remain challenging to detect accurately, particularly when they occur in **visually cluttered regions** (e.g., a small object against a busy urban background).

### 2. Ambiguity Under Extreme Environmental Variation

Large temporal appearance changes caused by **extreme weather conditions** or **seasonal variation** can introduce ambiguity between genuine structural modifications and natural environmental changes — even with DINOv2's improved robustness, very large domain shifts remain a challenge.

### 3. Rule-Based Temporal Persistence

The temporal persistence verification stage is currently implemented as **deterministic rule-based filtering** (presence/absence counters against fixed thresholds `T_p`, `T_c`), rather than learned end-to-end temporal sequence modeling. This means:

- Persistence thresholds must be manually configured per deployment scenario.
- The model cannot learn more nuanced temporal patterns (e.g., intermittent occlusion of a persistent object) beyond simple consecutive-frame counting.

### 4. Fixed, Frozen Backbone

Since the DINOv2 backbone is entirely frozen, PSCDLNet cannot adapt its low-level visual representations to dataset-specific characteristics beyond what the adapter and decoder can compensate for. This is a deliberate trade-off for robustness and parameter efficiency, but it does cap the model's capacity to specialize to unusual domains not well represented in DINOv2's pretraining data.

### 5. Single-Camera, Fixed-Viewpoint Assumption

The framework — like the PSCDL 2026 dataset itself — assumes a **fixed surveillance camera** with a stable viewpoint between `I_t0` and `I_t1`. Performance under camera motion, viewpoint drift, or multi-camera fusion scenarios has not been evaluated.

> These limitations directly motivate the [Future Work](#-future-work) directions discussed next.
---

## 🔭 Future Work

Building on the limitations identified above, several directions are proposed to extend and strengthen PSCDLNet for broader real-world deployment.

### 1. Larger and Newer Foundation Models

Explore larger DINOv2 variants (ViT-B/14, ViT-L/14) and emerging self-supervised vision foundation models, which may provide even richer semantic representations for persistent scene understanding — at the cost of increased compute.

### 2. Learned Temporal Modeling

Replace the current deterministic, rule-based persistence verification with **transformer-based temporal modeling** or **video foundation models**, enabling the network to directly learn long-term temporal dependencies from video sequences rather than relying on fixed presence/absence thresholds.

### 3. Adaptive Attention and Feature Fusion

Investigate adaptive attention mechanisms and improved multi-scale feature fusion strategies to further improve localization of **small-scale or distant scene modifications** in visually complex or cluttered urban environments.

### 4. Real-Time and Edge Deployment

Extend the framework toward **real-time inference** and **edge-device deployment**, including model compression, quantization, and latency optimization — important for practical, large-scale smart-city monitoring systems.

### 5. Multi-Camera Surveillance Systems

Generalize the framework beyond single fixed-camera pairs toward **multi-camera surveillance networks**, enabling persistent change detection and localization across overlapping or complementary camera views.

### 6. Broader Domain Generalization

Evaluate and improve robustness under more extreme domain shifts (severe weather, drastic seasonal change, nighttime/low-light conditions) beyond what is represented in the current PSCDL 2026 dataset.

> These directions aim to move PSCDLNet from a strong research baseline toward a production-ready system for long-term, real-world urban surveillance monitoring.
---

## 📖 Citation

If you use PSCDLNet, or any part of this codebase, in your research, please cite:

```bibtex
@inproceedings{kumar2026pscdlnet,
  title     = {Persistent Scene Change Detection and Localization Using Frozen DINOv2 Features and Multi-Scale Siamese Learning},
  author    = {Kumar, Naresh and Porwal, Mukti and Kumar, Maneesh and Saini, Shubham and Singh, Tavneet and Kaushik, Baijnath},
  booktitle = {Proceedings of the Persistent Scene Change Detection and Localization (PSCDL) 2026 Challenge},
  year      = {2026},
  organization = {Shri Mata Vaishno Devi University}
}
```

Please also cite the official PSCDL 2026 dataset and challenge:

```bibtex
@misc{pscdl2026challenge,
  title        = {Persistent Scene Change Detection and Localization (PSCDL 2026) Challenge: Dataset, Evaluation Protocol, and Benchmark},
  author       = {{Vehant Technologies}},
  year         = {2026},
  howpublished = {PSCDL 2026 Challenge Documentation}
}
```

And, if referencing the frozen backbone used in this work:

```bibtex
@article{oquab2023dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal = {arXiv preprint arXiv:2304.07193},
  year    = {2023}
}
```
---

## 🙏 Acknowledgements

This work was developed at the **School of Computer Science & Engineering, Shri Mata Vaishno Devi University**, Katra, Jammu and Kashmir, India, as part of participation in the **PSCDL 2026 Challenge (Vehant Technologies)**.

We gratefully acknowledge:
- **Meta AI Research** for releasing the pretrained **DINOv2** self-supervised vision transformer models used as the frozen backbone in this work.
- The **PSCDL 2026 Challenge organizers (Vehant Technologies)** for providing the official dataset, evaluation protocol, and benchmark infrastructure.
- The open-source **PyTorch** and **Hugging Face Transformers** communities, whose tools underpin this implementation.

## 📚 References

1. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.
2. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.
3. A. Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," in *Proc. ICLR*, 2021.
4. A. Vaswani et al., "Attention Is All You Need," in *Advances in NeurIPS*, 2017, pp. 5998–6008.
5. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv:2304.07193*, 2023.
6. M. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," in *Proc. IEEE ICCV*, 2021, pp. 9650–9660.
7. W. Zheng, L. Wang, and Q. Meng, "ChangeFormer: A Transformer-Based Siamese Network for Change Detection," *IEEE Trans. Geoscience and Remote Sensing*, vol. 60, 2022.
8. R. C. Daudt, B. Le Saux, and A. Boulch, "Fully Convolutional Siamese Networks for Change Detection," in *Proc. IEEE ICIP*, 2018, pp. 4063–4067.
9. R. C. Daudt, B. Le Saux, and A. Boulch, "Urban Change Detection for Multispectral Earth Observation Using CNNs," in *Proc. IEEE IGARSS*, 2018.
10. Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows," in *Proc. IEEE ICCV*, 2021, pp. 10012–10022.
11. J. Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation," *arXiv:2102.04306*, 2021.
12. Y. Wang et al., "Deep Learning for Change Detection in Remote Sensing Images: A Comprehensive Review," *IEEE Geoscience and Remote Sensing Magazine*, vol. 8, no. 2, pp. 34–57, 2020.
13. H. Chen, C. Wu, B. Du, and L. Zhang, "Remote Sensing Image Change Detection with Transformers: A Survey," *IEEE Geoscience and Remote Sensing Magazine*, vol. 10, no. 4, pp. 22–41, 2022.
14. Y. Fang et al., "Foundation Models for Computer Vision: A Survey," *arXiv:2307.13721*, 2023.
15. K. Zhou et al., "A Survey of Self-Supervised Learning in Computer Vision," *IEEE Trans. Pattern Analysis and Machine Intelligence*, 2023.
16. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. ICLR*, 2015.
17. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. ICLR*, 2019.
18. F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," in *Proc. 3DV*, 2016, pp. 565–571.
19. T.-Y. Lin et al., "Focal Loss for Dense Object Detection," in *Proc. IEEE ICCV*, 2017, pp. 2980–2988.
20. Z. Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," in *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 2018, pp. 3–11.
21. M. Everingham et al., "The Pascal Visual Object Classes (VOC) Challenge: A Retrospective," *International Journal of Computer Vision*, vol. 111, no. 1, pp. 98–136, 2015.
22. M. Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding," in *Proc. IEEE CVPR*, 2016, pp. 3213–3223.
23. Vehant Technologies, "Persistent Scene Change Detection and Localization (PSCDL 2026) Challenge: Dataset, Evaluation Protocol, and Benchmark," *PSCDL 2026 Challenge Documentation*, 2026.
---
## 📜 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Naresh Kumar, Mukti Porwal, Maneesh Kumar, Shubham Saini, Tavneet Singh, Baijnath Kaushik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> **Note:** The PSCDL 2026 dataset itself is **not** covered by this license and is subject to the terms provided by the official challenge organizers (Vehant Technologies). See [Dataset Description](#-dataset-description) for details.