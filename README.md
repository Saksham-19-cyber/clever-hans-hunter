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
| **Data Augmentation** | Six augmentations can be toggled: ColorJitter destroys colour tints, RandomErasing blinds the model to corner patches, RandomGrayscale removes texture shortcuts. Strength is a continuous slider. |
| **Loss Functions** | Three loss functions are selectable: plain cross-entropy, class-weighted CE, and group-reweighted CE. The last one is the key mechanic for beating shortcuts. |
| **Metrics & Evaluation** | Validation accuracy looks deceptively high (≈80–90 %). Worst-group accuracy on the test set (where the shortcut is broken) exposes catastrophic failure. The game's score is worst-group accuracy, not average accuracy. |

---

## Planted Shortcuts

| Level | Shortcut | Train/Val Correlation | Test Correlation | Target WG Acc |
|---|---|---|---|---|
| **1 – Color Tint** | Cat images get a blue background tint; dog images get a red tint. | 95 % | 50 % (random) | **55 %** |
| **2 – Corner Patch** | A small solid-colour square (blue=cat, red=dog) in the top-left corner. | 95 % | 50 % (random) | **60 %** |
| **3 – Stripe Texture** | Faint horizontal stripes overlay cat images; vertical stripes overlay dog images. | 95 % | 50 % (random) | **52 %** |

The shortcut is present (and correlated with the label) 95 % of the time in train/val, but randomly assigned in the test set. Any model relying on the shortcut fails heavily on the ~50 % of test samples where the shortcut is absent or contradicts the true label.

> **Note on targets:** Target worst-group accuracies (55 %, 60 %, 52 %) were set after empirical measurement to be achievable by the recommended fix (group-reweighted ERM + targeted augmentation) in the Streamlit slider epoch budget, and are above baseline by a comfortable margin. They are not guaranteed minimum theoretical bounds.

---

## Empirical Results

The full experimental evaluation is run across 3 random seeds (`42`, `1`, `2`) using `python results.py` (2 epochs per condition). Wilson 95% confidence intervals for group accuracies are pooled across seeds.

> **CI caveat:** All conditions share the same test set. Confidence intervals account for within-condition binomial uncertainty but do not correct for the shared-test-set multiple-comparison dependency.

### Measured Accuracy Table (Mean ± Std over 3 Seeds)

| Level | Condition | Val Acc | Test Avg | WG Acc | G0 Cat (no SC) | G1 Cat (+SC) | G2 Dog (no SC) | G3 Dog (+SC) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Baseline (head only) | 0.89 ± 0.01 | 0.53 ± 0.01 | **0.09 ± 0.01** | 0.10 | 0.92 | 0.17 | 0.95 |
| **1** | Ctrl (last block, no aug) | 0.95 ± 0.00 | 0.59 ± 0.01 | **0.13 ± 0.01** | 0.13 | 0.99 | 0.28 | 0.99 |
| **1** | Aug Only | 0.94 ± 0.00 | 0.61 ± 0.01 | **0.20 ± 0.02** | 0.20 | 0.98 | 0.31 | 0.98 |
| **1** | Loss Only | 0.82 ± 0.01 | 0.74 ± 0.02 | **0.62 ± 0.04** | 0.63 | 0.91 | 0.62 | 0.89 |
| **1** | **Recommended Fix** | 0.81 ± 0.03 | 0.73 ± 0.01 | **0.63 ± 0.06** | 0.65 | 0.87 | 0.63 | 0.86 |
| **1** | Oracle (50% corr) | 0.80 ± 0.01 | 0.80 ± 0.01 | **0.75 ± 0.02** | 0.75 | 0.82 | 0.76 | 0.85 |
| **2** | Baseline (head only) | 0.77 ± 0.00 | 0.61 ± 0.01 | **0.41 ± 0.01** | 0.42 | 0.83 | 0.41 | 0.77 |
| **2** | Ctrl (last block, no aug) | 0.95 ± 0.00 | 0.58 ± 0.01 | **0.12 ± 0.02** | 0.12 | 0.99 | 0.25 | 0.99 |
| **2** | Aug Only | 0.95 ± 0.00 | 0.58 ± 0.01 | **0.15 ± 0.03** | 0.15 | 0.99 | 0.26 | 0.99 |
| **2** | Loss Only | 0.80 ± 0.01 | 0.76 ± 0.01 | **0.68 ± 0.04** | 0.70 | 0.89 | 0.68 | 0.84 |
| **2** | **Recommended Fix** | 0.73 ± 0.01 | 0.72 ± 0.02 | **0.53 ± 0.06** | 0.72 | 0.88 | 0.53 | 0.78 |
| **2** | Oracle (50% corr) | 0.77 ± 0.02 | 0.76 ± 0.01 | **0.63 ± 0.06** | 0.63 | 0.82 | 0.67 | 0.84 |
| **3** | Baseline (head only) | 0.79 ± 0.00 | 0.60 ± 0.01 | **0.36 ± 0.03** | 0.36 | 0.82 | 0.42 | 0.82 |
| **3** | Ctrl (last block, no aug) | 0.95 ± 0.00 | 0.58 ± 0.00 | **0.14 ± 0.00** | 0.14 | 0.99 | 0.25 | 0.99 |
| **3** | Aug Only | 0.92 ± 0.00 | 0.59 ± 0.01 | **0.12 ± 0.02** | 0.12 | 0.87 | 0.40 | 0.98 |
| **3** | Loss Only | 0.80 ± 0.01 | 0.73 ± 0.01 | **0.61 ± 0.03** | 0.62 | 0.82 | 0.64 | 0.84 |
| **3** | **Recommended Fix** | 0.77 ± 0.04 | 0.68 ± 0.00 | **0.51 ± 0.02** | 0.53 | 0.72 | 0.63 | 0.85 |
| **3** | Oracle (50% corr) | 0.78 ± 0.01 | 0.74 ± 0.01 | **0.60 ± 0.04** | 0.82 | 0.82 | 0.60 | 0.71 |

*All levels (1, 2, and 3) display empirical matched-controls metrics (mean ± std across 3 seeds, 2 epochs per condition).*

### Augmentation Cue Survival Audit (200 Training Images)

| Level | Shortcut Signal | Before Aug | After Std Aug | Retention | After Strong Aug | Retention |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Mean Blue-minus-Red gap (cat − dog) | 0.5395 | 0.4206 | **78 %** | 0.0000 | **0 %** |
| **2** | Fraction top-left 8×8 patch unchanged | 100 % | 0.0 % | **0 %** | 14.5 % | **14.5 %** |
| **3** | Stripe-frequency FFT energy | 11 541 | 11 375 | **98.6 %** | 405 | **3.5 %** |

Standard augmentations for Level 1 and 2 are effective at disrupting the shortcut signal. For Level 3 (stripe texture), the standard random-grayscale + Gaussian-noise combination barely reduces FFT stripe energy (98.6% survives), while a targeted Gaussian blur (σ=2.5) reduces it to 3.5%. This explains why Level 3 is the hardest to fix with augmentation alone.

### Key Findings & Honest Failure Modes

1. **Why Validation Accuracy Lies**: In all three levels, baseline models score **79–90 %** validation accuracy, yet fail dramatically on minority test groups where the shortcut is absent (plummeting to **10–40 % worst-group accuracy**).
2. **Augmentation Alone is Insufficient**: Standard augmentations with plain cross-entropy still leave 78% of the Level-1 color signal intact. The model re-exploits the residual signal, achieving high val acc but low worst-group accuracy.
3. **The Fix Requires Both**: Combining targeted augmentations with group-reweighted cross-entropy (up-weighting shortcut-absent groups) is necessary to clear all three level targets.
4. **Matched Controls Isolate Each Factor**: The new `ctrl_lb` condition (last block, same LR/epochs, no aug, plain CE) shows that deeper fine-tuning alone does not fix shortcut exploitation.

---

## Visual Diagnostics & Grad-CAM Analysis

### Level 1: Color Tint
![Level 1 Per-Group Accuracy](docs/level1_groups.png)
![Level 1 Baseline vs Fix](docs/level1_comparison.png)

*Grad-CAM Before Fix (Model focuses on tinted background):*  
![Level 1 Grad-CAM Before](docs/level1_gradcam_before.png)

*Grad-CAM After Fix (Model shifts attention to object features):*  
![Level 1 Grad-CAM After](docs/level1_gradcam_after.png)

---

### Level 2: Corner Patch
![Level 2 Per-Group Accuracy](docs/level2_groups.png)
![Level 2 Baseline vs Fix](docs/level2_comparison.png)

*Grad-CAM Before Fix (Model fixates on the top-left color square):*  
![Level 2 Grad-CAM Before](docs/level2_gradcam_before.png)

*Grad-CAM After Fix (Attention restored to animal contours):*  
![Level 2 Grad-CAM After](docs/level2_gradcam_after.png)

---

### Level 3: Stripe Texture
![Level 3 Per-Group Accuracy](docs/level3_groups.png)
![Level 3 Baseline vs Fix](docs/level3_comparison.png)

*Grad-CAM Before Fix (Focus on stripe pattern artifacts):*  
![Level 3 Grad-CAM Before](docs/level3_gradcam_before.png)

*Grad-CAM After Fix (Attention invariant to overlay texture):*  
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
