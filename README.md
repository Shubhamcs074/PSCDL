# PSCDLNet: Persistent Scene Change Detection and Localization Using Frozen DINOv2 Features and Multi-Scale Siamese Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Challenge](https://img.shields.io/badge/PSCDL-2026-red)

Official repository for **PSCDLNet**, a foundation-model-driven Siamese framework for **Persistent Scene Change Detection and Localization (PSCDL)**, developed for the **PSCDL 2026 Challenge**.

---

## 📌 Overview

Persistent Scene Change Detection and Localization (PSCDL) targets **long-term structural modifications** in monitored environments — abandoned objects, removed objects, construction barriers, roadblocks, and other permanent alterations — while suppressing **transient changes** caused by pedestrians, moving vehicles, illumination shifts, shadows, and weather.

Conventional CNN-based Siamese change detection networks (e.g., ResNet18/ResNet34-UNet) are fast and effective at capturing local spatial detail, but their supervised convolutional representations remain sensitive to illumination variation, shadows, and seasonal appearance changes — often producing false alarms in real-world surveillance footage.

**PSCDLNet** addresses this by replacing the convolutional encoder with a **frozen, self-supervised DINOv2 Vision Transformer (ViT-S/14)** backbone. Only a lightweight adapter and decoder are trained, while the foundation model's generalized semantic representations are preserved untouched. This yields a framework that is simultaneously **more robust** to environmental noise and **more parameter-efficient** to train than fully supervised CNN baselines.

---

## 🧠 Key Contributions

- **Foundation-model-based Siamese framework** — a frozen DINOv2 ViT-S/14 backbone paired with a trainable UNet-style decoder for persistent scene change localization.
- **Multi-scale DINO Feature Adapter** — converts 1D transformer token embeddings from four intermediate layers into a spatial feature pyramid suitable for convolutional decoding.
- **Explicit absolute feature differencing** — computes `|F(I_t0) − F(I_t1)|` at every feature scale, suppressing invariant background and amplifying genuine structural change, instead of relying on the decoder to implicitly learn feature comparison.
- **Temporal persistence verification** — a deterministic post-processing stage that confirms a detected change only after it persists across consecutive observations (and enforces a cooldown period for removed objects), filtering out transient false alarms.
- **Comprehensive comparative study** — Siamese ResNet18-UNet and ResNet34-UNet baselines were trained and evaluated under an identical protocol on the official PSCDL 2026 dataset to fairly benchmark convolutional vs. foundation-model encoders.

---

## 🏗️ Architecture

The pipeline consists of six sequential stages:

1. **Image preprocessing & normalization** — resize to 512×512 (interpolated to 518×518 for DINOv2's patch-size requirements), ImageNet channel normalization.
2. **Frozen Siamese DINOv2 encoder** — a shared, frozen ViT-S/14 backbone processes both temporal images `I_t0` and `I_t1` independently, extracting intermediate representations from transformer blocks `{2, 5, 8, 11}`.
3. **DINO Feature Adapter** — patch tokens from each selected layer are reshaped into 2D spatial grids and refined via 1×1 projection → 3×3 conv → BatchNorm → ReLU, producing a multi-scale feature pyramid (1/4, 1/8, 1/16, 1/32 resolution).
4. **Absolute feature differencing** — `D_n = |F_n(I_t0) − F_n(I_t1)|` is computed at each of the four scales.
5. **UNet-style decoder** — progressively upsamples and fuses the differenced feature pyramid (coarse → fine) via skip connections, ending in a 1×1 convolution + sigmoid to produce a pixel-wise change probability map.
6. **Temporal persistence verification (post-processing)** — a presence/absence counter validates that detected or removed regions persist beyond configurable thresholds (`T_p`, `T_c`) before being reported, suppressing transient objects such as pedestrians and vehicles.

```
Input Pair (t0, t1)
   │
   ▼
Frozen DINOv2 ViT-S/14 Backbone (shared weights, no gradient)
   │  layers {2, 5, 8, 11}
   ▼
DINO Feature Adapter (trainable) → multi-scale spatial features
   │
   ▼
Absolute Feature Differencing  D_n = |F_n(t0) − F_n(t1)|
   │
   ▼
UNet Decoder (trainable) → Change Probability Map → Sigmoid
   │
   ▼
Binarization (optimal threshold)
   │
   ▼
Temporal Persistence Verification (post-processing)
   │
   ▼
Final Persistent Change Mask
```

---

## 📊 Results

### Quantitative Comparison (PSCDL 2026 Validation Set)

| Model | Encoder | IoU (%) | F1-score (%) | Precision (%) | Recall (%) | Accuracy (%) |
|---|---|---|---|---|---|---|
| ResNet18-UNet | Supervised CNN | 70.62 | 82.19 | 85.45 | 80.07 | 92.89 |
| ResNet34-UNet | Supervised CNN | 70.46 | 81.35 | 80.02 | 83.98 | 92.74 |
| **PSCDLNet (Proposed)** | **Frozen DINOv2 ViT-S/14** | **71.16** | 81.35 | **87.19** | 78.16 | **94.68** |

PSCDLNet achieves the best IoU and Accuracy overall, and the highest Precision by a wide margin — reflecting its ability to suppress false positives caused by illumination changes and background clutter, at a modest recall trade-off.

### Threshold Sensitivity

An automatic threshold search (0.10–0.90) was performed on the validation set for both the best-F1 and best-IoU checkpoints. Peak performance for PSCDLNet occurs around a threshold of **0.30–0.35**, balancing precision and recall before degrading at extreme thresholds.

### Training Behavior

- All three models (ResNet18-UNet, ResNet34-UNet, PSCDLNet) were trained under an identical protocol: AdamW optimizer, LR = 1e-4, `ReduceLROnPlateau` scheduler (on validation IoU), max 150 epochs, early stopping (patience = 30), batch size = 1, hybrid **BCE + Dice loss**.
- The CNN baselines converge in ~60 epochs with the entire network trainable.
- PSCDLNet converges smoothly despite updating **only the adapter and decoder** (the DINOv2 backbone stays frozen throughout), demonstrating strong sample efficiency from self-supervised pretraining.

---

## 📁 Dataset

**PSCDL 2026 Official Dataset**

- **464** training image pairs
- **116** validation image pairs
- **5** blind evaluation videos (official challenge evaluation only)
- Each sample: a baseline observation `I_t0`, a current observation `I_t1`, and a pixel-level binary ground-truth mask of persistent changes.
- Diverse urban scenes: roads, parking lots, sidewalks, traffic intersections — with illumination shifts, weather variation, shadows, seasonal differences, and partial occlusion.

> The dataset is provided by the official PSCDL 2026 Challenge organizers and is not redistributed in this repository. Please refer to the official challenge page for access instructions.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/PSCDLNet.git
cd PSCDLNet

# 2. Create and activate an environment
conda create -n pscdlnet python=3.10 -y
conda activate pscdlnet

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** PyTorch ≥ 2.0, `torchvision`, `transformers` (for DINOv2), OpenCV, NumPy, Matplotlib.

---

## 🚀 Usage

> Update the script names/paths below to match your actual repository layout.

### Training

```bash
python train.py \
  --encoder dinov2_vits14 \
  --freeze-backbone \
  --input-size 512 \
  --batch-size 1 \
  --lr 1e-4 \
  --epochs 150 \
  --early-stopping-patience 30 \
  --loss bce_dice
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/pscdlnet_best_iou.pth \
  --data-root data/pscdl2026/val \
  --threshold-search
```

### Inference on a Video / Image Pair Sequence

```bash
python infer.py \
  --checkpoint checkpoints/pscdlnet_best_f1.pth \
  --t0 path/to/reference.jpg \
  --t1 path/to/current.jpg \
  --output-dir outputs/masks \
  --enable-persistence-filter
```

---

## 📂 Repository Structure

```
.
├── models/
│   ├── dino_encoder.py         # Frozen DINOv2 ViT-S/14 wrapper
│   ├── feature_adapter.py      # Multi-scale DINO Feature Adapter
│   ├── decoder.py               # UNet-style multi-scale decoder
│   └── persistence_filter.py   # Temporal persistence verification module
├── datasets/
│   └── pscdl_dataset.py         # PSCDL 2026 dataset loader
├── train.py                     # Training entry point
├── evaluate.py                   # Validation metrics + threshold search
├── infer.py                      # Inference on new image pairs / sequences
├── configs/
│   └── pscdlnet.yaml             # Model & training configuration
├── requirements.txt
└── README.md
```

---

## 🧩 Ablation Highlights

| Component Removed / Replaced | Effect |
|---|---|
| DINOv2 → ResNet18/34 | Lower robustness to illumination/shadow variation; more false positives |
| Multi-scale adapter → single-scale features | Increased fragmented predictions, weaker boundary delineation |
| Absolute differencing → feature concatenation | Harder optimization; decoder must implicitly learn comparison |
| Temporal persistence filter removed | Transient objects (pedestrians, vehicles) trigger false alarms |

Each component contributes complementary gains — the best performance emerges from their combination rather than any single design choice.

---

## ⚠️ Limitations

- Very small scene modifications in visually cluttered regions remain difficult to detect precisely.
- Extreme weather or strong seasonal appearance shifts can be confused with genuine structural change.
- Temporal persistence reasoning currently uses deterministic rule-based filtering rather than learned end-to-end temporal modeling.

## 🔭 Future Work

- Explore larger DINO variants / newer self-supervised backbones.
- Replace rule-based persistence filtering with transformer-based temporal sequence modeling.
- Improve small-object and long-range localization via adaptive attention/fusion.
- Extend to real-time and edge-device deployment for multi-camera smart-city systems.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@inproceedings{pscdlnet2026,
  title     = {Persistent Scene Change Detection and Localization Using Frozen DINOv2 Features and Multi-Scale Siamese Learning},
  author    = {Kumar, Naresh and Porwal, Mukti and Kumar, Maneesh and Saini, Shubham and Singh, Tavneet and Kaushik, Baijnath},
  booktitle = {PSCDL 2026 Challenge},
  year      = {2026},
  institution = {Shri Mata Vaishno Devi University}
}
```

---

## 🙏 Acknowledgements

- **DINOv2** — Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- **PSCDL 2026 Challenge** organizers (Vehant Technologies) for the official dataset and evaluation protocol.

## 📜 License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
