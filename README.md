# 🔍 Clever Hans Hunter

[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)](#test-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](https://clever-hans-hunter-xce4d2b8huzukgsoxkw9br.streamlit.app/)

> A gamified computer-vision project for deep-learning courses.  
> **You are handed a classifier that achieves high validation accuracy (up to 90–95 %) but fails in the real world because it learned a planted shortcut — your job is to diagnose and fix it.**
>
> 🚀 **[Live Demo](https://clever-hans-hunter-xce4d2b8huzukgsoxkw9br.streamlit.app/)**

---

## What It Teaches

| Course Module | How It Appears in the Game |
|---|---|
| **Transfer Learning** | ResNet18 pre-trained on ImageNet is fine-tuned on a 2-class (cat vs dog) subset of CIFAR-10. Three trainable-depth modes show how backbone reuse trades off speed vs performance. |
| Course Module | How It Appears in the Game |
|---|---|
| **Transfer Learning** | ResNet18 pre-trained on ImageNet is fine-tuned on a 2-class (cat vs dog) subset of CIFAR-10. Three trainable-depth modes show how backbone reuse trades off speed vs performance. |
| **Data Augmentation** | Six augmentations can be toggled: ColorJitter, RandomErasing, RandomGrayscale, Gaussian Noise, etc., with continuous strength sliders. As our empirical audits show, standard augmentations alone leave substantial shortcut signals intact and are insufficient to break shortcut reliance. |
| **Loss Functions** | Three loss functions are selectable: plain cross-entropy, class-weighted CE, and group-reweighted CE. Group-reweighted loss is the primary effective lever for mitigating shortcuts. |
| **Metrics & Evaluation** | Validation accuracy looks deceptively high (≈80–95 %). Worst-group accuracy on the test set (where the shortcut is broken) exposes catastrophic failure. The game's score is worst-group accuracy, not average accuracy. |

---

## Planted Shortcuts

| Level | Shortcut | Train/Val Correlation | Test Correlation | Target WG Acc |
|---|---|---|---|---|
| **1 – Color Tint** | Cat images get a blue background tint; dog images get a red tint. | 95 % | 50 % (random) | **55 %** |
| **2 – Corner Patch** | A small solid-colour square (blue=cat, red=dog) in the top-left corner. | 95 % | 50 % (random) | **60 %** |
| **3 – Stripe Texture** | Faint horizontal stripes overlay cat images; vertical stripes overlay dog images. | 95 % | 50 % (random) | **52 %** |

The shortcut is present (and correlated with the label) 95 % of the time in train/val, but randomly assigned in the test set. Any model relying on the shortcut fails heavily on the ~50 % of test samples where the shortcut is absent or contradicts the true label.

> **Note on targets:** Target worst-group accuracies (55 %, 60 %, 52 %) were set after empirical measurement. The recommended fix (group-reweighted loss + standard augmentation) clears the L1 target (62.8% ≥ 55%), but misses the L2 target (52.5% < 60%) and L3 target (51.2% < 52%). In contrast, group-reweighted loss alone (`loss_only`) reaches all three targets (L1: 62.5%, L2: 68.0%, L3: 60.9%).

---

## Empirical Results

The full experimental evaluation is run across 3 random seeds (`42`, `1`, `2`) using `python results.py` (2 epochs per condition). Wilson 95% confidence intervals are computed per seed using each group's test size ($n \approx 250$) and reported as the unpooled range $[min\_lo, max\_hi]$ across seeds (counts are not pooled across seeds).

> **CI caveat:** All conditions share the same test set. Confidence intervals account for within-condition binomial uncertainty but do not correct for the shared-test-set multiple-comparison dependency.

### Measured Accuracy Table (Mean ± Std over Seeds 42, 1, 2)

| Level | Condition | Val Acc | Test Avg | WG Acc (95% CI) | G0 Cat (no SC) | G1 Cat (+SC) | G2 Dog (no SC) | G3 Dog (+SC) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Baseline (head only) | 0.89 ± 0.01 | 0.53 ± 0.01 | **0.09 ± 0.01** [0.05, 0.15] | 0.09 [0.05, 0.15] | 0.91 [0.86, 0.96] | 0.20 [0.14, 0.27] | 0.93 [0.87, 0.96] |
| **1** | Ctrl (last block, no aug) | 0.95 ± 0.00 | 0.59 ± 0.01 | **0.13 ± 0.01** [0.08, 0.18] | 0.13 [0.08, 0.18] | 0.97 [0.93, 0.99] | 0.30 [0.20, 0.41] | 0.99 [0.96, 1.00] |
| **1** | Aug Only | 0.94 ± 0.00 | 0.61 ± 0.01 | **0.20 ± 0.02** [0.14, 0.28] | 0.20 [0.14, 0.28] | 0.97 [0.93, 0.99] | 0.30 [0.22, 0.38] | 0.99 [0.96, 1.00] |
| **1** | Loss Only | 0.82 ± 0.01 | 0.74 ± 0.02 | **0.62 ± 0.04** [0.51, 0.71] | 0.66 [0.59, 0.72] | 0.81 [0.72, 0.89] | 0.66 [0.51, 0.78] | 0.84 [0.73, 0.92] |
| **1** | Recommended Fix | 0.81 ± 0.03 | 0.73 ± 0.00 | **0.63 ± 0.06** [0.48, 0.74] | 0.68 [0.60, 0.75] | 0.79 [0.62, 0.93] | 0.65 [0.48, 0.78] | 0.82 [0.73, 0.90] |
| **1** | Oracle (plain CE, no aug) | 0.80 ± 0.01 | 0.80 ± 0.01 | **0.77 ± 0.02** [0.69, 0.84] | 0.83 [0.73, 0.90] | 0.79 [0.72, 0.85] | 0.78 [0.69, 0.85] | 0.80 [0.73, 0.85] |
| **1** | Oracle (with fix aug) | 0.80 ± 0.01 | 0.80 ± 0.01 | **0.75 ± 0.02** [0.67, 0.82] | 0.83 [0.77, 0.88] | 0.81 [0.72, 0.88] | 0.77 [0.70, 0.82] | 0.78 [0.67, 0.86] |
| **1** | Strong Aug Only (train-only) | 0.60 ± 0.16 | 0.52 ± 0.02 | **0.07 ± 0.10** [0.00, 0.26] | 0.77 [0.25, 1.00] | 0.95 [0.84, 1.00] | 0.11 [0.01, 0.26] | 0.27 [0.00, 0.85] |
| **1** | Strong Aug + Loss (train-only) | 0.54 ± 0.05 | 0.52 ± 0.01 | **0.13 ± 0.12** [0.01, 0.35] | 0.57 [0.07, 0.99] | 0.62 [0.04, 0.99] | 0.43 [0.02, 0.98] | 0.44 [0.01, 0.95] |
| **1** | Pre-processing Only (train & eval) | 0.81 ± 0.00 | 0.76 ± 0.00 | **0.67 ± 0.01** [0.59, 0.74] | 0.73 [0.64, 0.81] | 0.85 [0.79, 0.89] | 0.67 [0.59, 0.74] | 0.79 [0.71, 0.85] |
| **1** | Pre-processing + Loss (train & eval) | 0.76 ± 0.02 | 0.74 ± 0.00 | **0.70 ± 0.01** [0.62, 0.77] | 0.72 [0.64, 0.80] | 0.75 [0.66, 0.82] | 0.72 [0.62, 0.79] | 0.76 [0.68, 0.84] |
| **2** | Baseline (head only) | 0.77 ± 0.00 | 0.61 ± 0.01 | **0.41 ± 0.01** [0.34, 0.48] | 0.41 [0.34, 0.48] | 0.83 [0.76, 0.87] | 0.43 [0.36, 0.51] | 0.79 [0.73, 0.85] |
| **2** | Ctrl (last block, no aug) | 0.95 ± 0.00 | 0.58 ± 0.01 | **0.12 ± 0.02** [0.07, 0.18] | 0.12 [0.07, 0.18] | 0.98 [0.93, 1.00] | 0.27 [0.16, 0.38] | 1.00 [0.97, 1.00] |
| **2** | Aug Only | 0.95 ± 0.00 | 0.58 ± 0.01 | **0.15 ± 0.03** [0.08, 0.23] | 0.18 [0.08, 0.31] | 0.99 [0.95, 1.00] | 0.19 [0.14, 0.25] | 0.99 [0.96, 1.00] |
| **2** | Loss Only | 0.79 ± 0.01 | 0.76 ± 0.01 | **0.68 ± 0.04** [0.58, 0.78] | 0.77 [0.65, 0.87] | 0.82 [0.72, 0.89] | 0.68 [0.58, 0.78] | 0.78 [0.69, 0.86] |
| **2** | Recommended Fix | 0.73 ± 0.01 | 0.72 ± 0.02 | **0.53 ± 0.06** [0.38, 0.64] | 0.84 [0.77, 0.90] | 0.85 [0.78, 0.91] | 0.53 [0.38, 0.64] | 0.66 [0.56, 0.73] |
| **2** | Oracle (plain CE, no aug) | 0.80 ± 0.00 | 0.81 ± 0.00 | **0.78 ± 0.01** [0.72, 0.84] | 0.84 [0.78, 0.88] | 0.80 [0.72, 0.86] | 0.79 [0.73, 0.85] | 0.81 [0.74, 0.88] |
| **2** | Oracle (with fix aug) | 0.77 ± 0.02 | 0.76 ± 0.01 | **0.63 ± 0.06** [0.49, 0.74] | 0.85 [0.75, 0.95] | 0.85 [0.74, 0.94] | 0.63 [0.49, 0.74] | 0.71 [0.56, 0.81] |
| **2** | Strong Aug Only (train-only) | 0.92 ± 0.01 | 0.71 ± 0.03 | **0.45 ± 0.05** [0.33, 0.57] | 0.52 [0.42, 0.67] | 0.95 [0.91, 0.98] | 0.45 [0.33, 0.57] | 0.94 [0.90, 0.98] |
| **2** | Strong Aug + Loss (train-only) | 0.77 ± 0.01 | 0.77 ± 0.00 | **0.68 ± 0.06** [0.55, 0.80] | 0.82 [0.74, 0.89] | 0.82 [0.69, 0.92] | 0.69 [0.55, 0.80] | 0.75 [0.65, 0.84] |
| **2** | Pre-processing Only (train & eval) | 0.80 ± 0.00 | 0.81 ± 0.01 | **0.78 ± 0.01** [0.71, 0.84] | 0.83 [0.76, 0.88] | 0.81 [0.71, 0.89] | 0.79 [0.71, 0.84] | 0.83 [0.77, 0.89] |
| **2** | Pre-processing + Loss (train & eval) | 0.77 ± 0.00 | 0.77 ± 0.01 | **0.73 ± 0.01** [0.66, 0.79] | 0.80 [0.68, 0.88] | 0.77 [0.69, 0.85] | 0.74 [0.66, 0.81] | 0.78 [0.72, 0.83] |
| **3** | Baseline (head only) | 0.79 ± 0.00 | 0.60 ± 0.01 | **0.36 ± 0.03** [0.27, 0.45] | 0.36 [0.27, 0.45] | 0.82 [0.73, 0.87] | 0.42 [0.33, 0.49] | 0.82 [0.74, 0.89] |
| **3** | Ctrl (last block, no aug) | 0.95 ± 0.00 | 0.58 ± 0.00 | **0.14 ± 0.00** [0.10, 0.19] | 0.14 [0.10, 0.19] | 0.99 [0.95, 1.00] | 0.25 [0.16, 0.33] | 0.99 [0.95, 1.00] |
| **3** | Aug Only | 0.92 ± 0.00 | 0.59 ± 0.01 | **0.12 ± 0.02** [0.07, 0.19] | 0.12 [0.07, 0.19] | 0.87 [0.81, 0.92] | 0.40 [0.33, 0.49] | 0.98 [0.95, 0.99] |
| **3** | Loss Only | 0.80 ± 0.01 | 0.73 ± 0.01 | **0.61 ± 0.03** [0.51, 0.69] | 0.62 [0.55, 0.69] | 0.82 [0.72, 0.89] | 0.64 [0.51, 0.74] | 0.84 [0.77, 0.89] |
| **3** | Recommended Fix | 0.77 ± 0.04 | 0.68 ± 0.00 | **0.51 ± 0.02** [0.42, 0.59] | 0.53 [0.42, 0.65] | 0.72 [0.56, 0.84] | 0.63 [0.47, 0.76] | 0.85 [0.76, 0.93] |
| **3** | Oracle (plain CE, no aug) | 0.79 ± 0.01 | 0.79 ± 0.01 | **0.75 ± 0.01** [0.68, 0.82] | 0.80 [0.73, 0.86] | 0.80 [0.72, 0.88] | 0.75 [0.68, 0.82] | 0.80 [0.73, 0.88] |
| **3** | Oracle (with fix aug) | 0.78 ± 0.01 | 0.74 ± 0.01 | **0.60 ± 0.04** [0.48, 0.70] | 0.82 [0.73, 0.87] | 0.82 [0.75, 0.88] | 0.60 [0.48, 0.70] | 0.71 [0.62, 0.80] |
| **3** | Strong Aug Only (train-only) | 0.94 ± 0.00 | 0.52 ± 0.01 | **0.04 ± 0.01** [0.01, 0.09] | 0.06 [0.01, 0.17] | 1.00 [0.97, 1.00] | 0.08 [0.03, 0.17] | 1.00 [0.97, 1.00] |
| **3** | Strong Aug + Loss (train-only) | 0.69 ± 0.11 | 0.64 ± 0.03 | **0.38 ± 0.12** [0.17, 0.58] | 0.55 [0.17, 0.91] | 0.74 [0.46, 0.90] | 0.60 [0.41, 0.85] | 0.66 [0.35, 0.92] |
| **3** | Pre-processing Only (train & eval) | 0.84 ± 0.01 | 0.69 ± 0.02 | **0.48 ± 0.04** [0.36, 0.58] | 0.50 [0.36, 0.61] | 0.84 [0.78, 0.88] | 0.53 [0.46, 0.62] | 0.89 [0.82, 0.94] |
| **3** | Pre-processing + Loss (train & eval) | 0.74 ± 0.00 | 0.72 ± 0.01 | **0.66 ± 0.03** [0.57, 0.75] | 0.67 [0.57, 0.77] | 0.77 [0.69, 0.84] | 0.68 [0.61, 0.75] | 0.76 [0.68, 0.84] |

*All conditions are evaluated across random seeds 42, 1, and 2 (2 epochs per condition). Group counts are ~250 test samples per group; 95% Wilson CIs are evaluated per seed and reported as unpooled ranges across seeds.*

### Augmentation Cue Survival Audit (200 Training Images)

| Level | Shortcut Signal | Before Aug | After Std Aug | Retention | After Strong Aug (Prob) | Retention |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Mean Blue-minus-Red gap (cat − dog) | 0.5395 | 0.5625 | **104.3 %** | 0.0000 ($p=1.0$) | **0.0 %** |
| **2** | Top-left 8×8 Mean Blue-minus-Red gap | 1.4240 | 1.2728 | **89.4 %** | 0.2160 ($p=0.85$) | **15.2 %** |
| **3** | Stripe-frequency FFT energy | 11 753.1 | 11 599.9 | **98.7 %** | 270.1 ($p=1.0$) | **2.3 %** |

Empirical cue audits clarify why standard augmentations are ineffective:
- In Level 1, standard ColorJitter leaves 104.3% of the background blue-vs-red gap intact (the cue survives completely).
- In Level 2, standard augmentations leave 89.4% of the corner color gap intact.
- In Level 3, standard noise retains 98.7% of FFT stripe energy.
- Strong targeted transforms destroy the cue (0% on L1, 15.2% on L2, 2.3% on L3). However, applying strong augmentation only at train time introduces a severe train/eval distribution mismatch.

### Key Findings & Honest Failure Modes

1. **Validation Accuracy is Deceptive**: Across all three levels, baseline models score **77–95 %** validation accuracy while catastrophically failing on counter-shortcut test groups (**9–41 % worst-group accuracy**). High validation accuracy directly masks shortcut reliance.
2. **Group-Reweighted Loss is the Effective Lever**: Group-reweighted loss alone (`loss_only`) reaches the target across all three levels: Level 1 (62.5% ≥ 55%), Level 2 (68.0% ≥ 60%), and Level 3 (60.9% ≥ 52%).
3. **Augmentation Alone Fails and Hurts the Fix on L2/L3**: Standard augmentation alone (`aug_only`) achieves only **12–20 %** worst-group accuracy. Adding standard augmentation to the loss (`Recommended Fix`) degrades worst-group accuracy on Level 2 (52.5% vs 68.0%) and Level 3 (51.2% vs 60.9%), causing the "fix" to **miss the Level 2 (60%) and Level 3 (52%) targets**.
4. **Train/Eval Mismatch in Train-Only Strong Augmentation**: When strong transforms are applied only at train time (`strong_aug_only`), the model experiences an out-of-distribution test shift. On Level 1, validation accuracy collapses (individual seed runs drop to 47.6%, which is below random chance), and worst-group accuracy falls to 7.0% on L1 and 4.0% on L3. When the exact same transforms are applied at BOTH train and eval as deterministic pre-processing (`pre_only`, `pre_loss`), distribution alignment is maintained, achieving **67–78 %** worst-group accuracy on Levels 1 and 2, and **48–66 %** on Level 3.
5. **Deeper Fine-Tuning Accelerates Shortcut Exploitation**: Fine-tuning deeper layers without reweighting (`ctrl_lb`) merely hastens shortcut memorization, dropping worst-group accuracy to 12–14% while pushing validation accuracy to 95%.
6. **Plain Oracle Upper Bound**: An oracle model trained on balanced data without augmentation (`oracle_plain`) establishes an upper bound of 75–78% worst-group accuracy. Adding standard augmentations to the oracle (`oracle (with fix aug)`) degrades performance across all levels (to 60–75%), confirming that augmentation at these strengths hurts rather than helps.

---

## Visual Diagnostics & Grad-CAM Analysis

### Level 1: Color Tint
![Level 1 Per-Group Accuracy](docs/level1_groups.png)
![Level 1 Baseline vs Fix](docs/level1_comparison.png)

*Grad-CAM Before Fix (Model focuses on tinted background; prediction wrong):*  
![Level 1 Grad-CAM Before](docs/level1_gradcam_before.png)

*Grad-CAM After Fix (Heatmap still shows shortcut reliance on background tint; prediction remains wrong):*  
![Level 1 Grad-CAM After](docs/level1_gradcam_after.png)

---

### Level 2: Corner Patch
![Level 2 Per-Group Accuracy](docs/level2_groups.png)
![Level 2 Baseline vs Fix](docs/level2_comparison.png)

*Grad-CAM Before Fix (Heatmap shows shortcut reliance / coarse activation; prediction wrong):*  
![Level 2 Grad-CAM Before](docs/level2_gradcam_before.png)

*Grad-CAM After Fix (Heatmap still shows shortcut reliance; prediction remains wrong):*  
![Level 2 Grad-CAM After](docs/level2_gradcam_after.png)

---

### Level 3: Stripe Texture
![Level 3 Per-Group Accuracy](docs/level3_groups.png)
![Level 3 Baseline vs Fix](docs/level3_comparison.png)

*Grad-CAM Before Fix (Heatmap focuses on stripe texture artifacts; prediction wrong):*  
![Level 3 Grad-CAM Before](docs/level3_gradcam_before.png)

*Grad-CAM After Fix (Heatmap still shows shortcut reliance on stripe texture; prediction remains wrong):*  
![Level 3 Grad-CAM After](docs/level3_gradcam_after.png)

---

## Limitations

- **Group Label Requirement**: The group-reweighted loss uses group labels (true class × shortcut presence). In unconstrained real-world settings, group labels are rarely available without explicit metadata or unsupervised discovery.
- **Synthetic Shortcuts**: The shortcuts (color tints, corner patches, and synthetic stripe grids) are artificially planted to provide clear pedagogical contrasts. Natural shortcuts (e.g. background grass, camera watermarks) are more subtle and multimodal.
- **Image Resolution**: CIFAR-10 images are upscaled to 64×64. Due to low native resolution, Grad-CAM heatmaps at `layer4` are coarse.
- **Binary Classification**: The demonstration uses a 2-class subset (cat vs dog) to ensure fast CPU-bound training runs under ~3 minutes.

---

## How to Run

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 3. Reproduce Experimental Results
To rerun the full matched-controls empirical evaluation and regenerate all figures:
```bash
python results.py
```

### 4. Run the Test Suite
```bash
python tests/test_basics.py
pytest tests/ -v
```

<a id="test-status"></a>
**Test Status**: All 33 automated unit and integration tests passing (`24/24` basic checks + `9/9` game loop tests).

---

## Memory Budget & Resource Profile

Measured on CPU via `python measure_memory.py` using `psutil` process RSS (uint8 dataset storage, from `results/memory.json`):

| Operation | Peak RSS |
|---|---|
| Import torch + streamlit | 222.9 MB |
| Dataset Load (Level 1) | **529.1 MB** |
| head_only training run | 304.5 MB |
| last_block training run | 504.0 MB |
| full training run | 516.2 MB |
| Grad-CAM | 314.8 MB |

Dataset tensors are stored as **uint8** (3×64×64 per image) and converted to float32 per batch in the collate function — peak RSS is from the initial CIFAR-10 array concatenation, not from holding float32 images.

---

## Player Controls Guide

### Fine-Tuning Depth
- `head_only`: Fast training, freezes all ResNet layers except the final linear classifier. Max 4 epochs in the app.
- `last_block`: Unfreezes `layer4` + FC layer. Allows high-level feature extraction adjustments. Max 2 epochs.
- `full`: Unfreezes all layers. Max 1 epoch.

### Augmentations
- **ColorJitter**: Randomizes brightness, contrast, saturation, and hue (counters Level 1).
- **RandomGrayscale**: Removes chromatic information (counters Level 1 & Level 2).
- **RandomResizedCrop**: Displaces and crops parts of the image (counters Level 2).
- **RandomErasing**: Randomly occludes image patches (counters Level 2).
- **Gaussian Noise**: Adds tensor noise to disrupt high-frequency patterns (counters Level 3).

---

## Repository Structure

```
clever_hans_hunter/
├── streamlit_app.py       – Streamlit UI with full game loop, diagnostics, and training
├── data.py                – CIFAR-10 dataset builder, shortcut injection, group IDs, LEVEL_TARGETS
├── model.py               – Pretrained ResNet18 wrapper with freeze modes
├── train.py               – Training engine, augmentations, loss functions, and metrics
├── gradcam.py             – Hook-based Grad-CAM implementation
├── results.py             – Multi-seed matched-controls evaluation with Wilson CIs
├── measure_memory.py      – RAM profiling script
├── requirements.txt       – Pinned dependencies with PyTorch CPU wheel index
├── LICENSE                – MIT License
├── .streamlit/
│   └── config.toml        – Headless configuration and telemetry settings
├── checkpoints/           – Pre-trained baseline checkpoints (level1/2/3_baseline.pt)
├── data/
│   └── cifar_catdog.npz   – Pre-built cat/dog subset (17.5 MB, seed=42)
├── weights/
│   └── resnet18_imagenet.pth – Cached ImageNet weights (44.7 MB, no download on cold start)
├── docs/                  – Generated evaluation figures and Grad-CAM comparisons
├── results/
│   ├── memory.json        – Memory profiling data
│   └── results.json       – Empirical metrics across all seeds and conditions
└── tests/
    ├── test_basics.py     – Sanity checks for data, shapes, and shortcuts
    └── test_game_loop.py  – Unit tests for game loop functions
```

---

## The Clever Hans Story

In the early 1900s, a horse named **Clever Hans** appeared to perform arithmetic by tapping his hoof. He succeeded in front of large crowds—until psychologist Oskar Pfungstadt demonstrated that Hans was not performing arithmetic at all. Instead, he was observing subtle, involuntary physical cues and tension changes from the human examiner.

Deep neural networks regularly exhibit Clever Hans behavior: when an unintended spurious feature correlates with the label in the training set, the network will exploit that shortcut rather than learning causal semantic features. **Worst-group accuracy on counter-shortcut distributions exposes this failure.**
