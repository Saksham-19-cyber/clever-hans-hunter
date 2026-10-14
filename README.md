# 🔍 Clever Hans Hunter

[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)](#test-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)](https://streamlit.io)

> A gamified computer-vision project for deep-learning courses.  
> **You are handed a classifier that achieves high validation accuracy (up to 90–95 %) but fails in the real world because it learned a planted shortcut — your job is to diagnose and fix it.**
>
> 🚀 **[Live Demo](TODO_STREAMLIT_URL)**

---

## What It Teaches

| Course Module | How It Appears in the Game |
|---|---|
| **Transfer Learning** | ResNet18 pre-trained on ImageNet is fine-tuned on a 2-class (cat vs dog) subset of CIFAR-10. Three trainable-depth modes show how backbone reuse trades off speed vs performance. |
| **Data Augmentation** | Six augmentations can be toggled: ColorJitter destroys colour tints, RandomErasing blinds the model to corner patches, RandomGrayscale removes texture shortcuts. Strength is a continuous slider. |
| **Loss Functions** | Three loss functions are selectable: plain cross-entropy, class-weighted CE, and group-reweighted CE (a form of distributionally robust optimisation). The last one is the key mechanic for beating shortcuts. |
| **Metrics & Evaluation** | Validation accuracy looks deceptively high (≈80–90 %). Worst-group accuracy on the test set (where the shortcut is broken) exposes catastrophic failure. The game's score is worst-group accuracy, not average accuracy. |

---

## Planted Shortcuts

| Level | Shortcut | Train/Val Correlation | Test Correlation | Target WG Acc |
|---|---|---|---|---|
| **1 – Color Tint** | Cat images get a blue background tint; dog images get a red tint. | 95 % | 50 % (random) | **55 %** |
| **2 – Corner Patch** | A small solid-colour square (blue=cat, red=dog) in the top-left corner. | 95 % | 50 % (random) | **60 %** |
| **3 – Stripe Texture** | Faint horizontal stripes overlay cat images; vertical stripes overlay dog images. | 95 % | 50 % (random) | **52 %** |

The shortcut is present (and correlated with the label) 95 % of the time in train/val, but randomly assigned in the test set. Any model relying on the shortcut fails heavily on the ~50 % of test samples where the shortcut is absent or contradicts the true label.

---

## Empirical Results

The full experimental evaluation is run across 3 random seeds (`42`, `1`, `2`) using `python results.py`.

### Measured Accuracy Table (Mean ± Std over 3 Seeds)

| Level | Condition | Val Acc (Misleading) | Test Avg Acc | Worst-Group Acc (Score) | Group 0 (Cat no SC) | Group 1 (Cat +SC) | Group 2 (Dog no SC) | Group 3 (Dog +SC) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Baseline | 0.90 ± 0.00 | 0.55 ± 0.00 | **0.10 ± 0.01** | 0.10 | 0.92 | 0.17 | 0.95 |
| **1** | **Recommended Fix** | 0.85 ± 0.01 | 0.75 ± 0.00 | **0.58 ± 0.01** | 0.63 | 0.87 | 0.61 | 0.86 |
| **1** | Augmentation Only | 0.95 ± 0.01 | 0.63 ± 0.00 | 0.22 ± 0.01 | 0.22 | 0.98 | 0.30 | 0.98 |
| **1** | Group Loss Only | 0.88 ± 0.01 | 0.74 ± 0.01 | 0.55 ± 0.01 | 0.55 | 0.91 | 0.58 | 0.89 |
| **2** | Baseline | 0.79 ± 0.00 | 0.61 ± 0.01 | **0.40 ± 0.02** | 0.42 | 0.83 | 0.40 | 0.77 |
| **2** | **Recommended Fix** | 0.81 ± 0.02 | 0.76 ± 0.00 | **0.63 ± 0.04** | 0.74 | 0.88 | 0.63 | 0.78 |
| **2** | Augmentation Only | 0.95 ± 0.00 | 0.61 ± 0.00 | 0.16 ± 0.04 | 0.16 | 0.99 | 0.27 | 0.99 |
| **2** | Group Loss Only | 0.85 ± 0.01 | 0.77 ± 0.01 | 0.65 ± 0.02 | 0.70 | 0.89 | 0.65 | 0.84 |
| **3** | Baseline | 0.81 ± 0.01 | 0.61 ± 0.01 | **0.34 ± 0.01** | 0.35 | 0.84 | 0.40 | 0.84 |
| **3** | **Recommended Fix** | 0.82 ± 0.01 | 0.72 ± 0.01 | **0.55 ± 0.04** | 0.63 | 0.88 | 0.56 | 0.79 |
| **3** | Augmentation Only | 0.93 ± 0.00 | 0.63 ± 0.01 | 0.17 ± 0.05 | 0.17 | 0.96 | 0.36 | 0.99 |
| **3** | Group Loss Only | 0.86 ± 0.01 | 0.73 ± 0.00 | 0.53 ± 0.01 | 0.54 | 0.89 | 0.57 | 0.90 |

### Key Findings & Honest Failure Modes

1. **Why Validation Accuracy Lies**: In all three levels, baseline models score **79–90 %** validation accuracy, yet fail dramatically on minority test groups where the shortcut is absent (plummeting to **10–40 % worst-group accuracy**).
2. **Augmentation Alone is Insufficient**: When applying augmentations with plain cross-entropy loss (`aug_only`), the model often fits the remaining signal and reaches **93–95 % validation accuracy**, but its worst-group accuracy drops to **16–22 %** because the optimization objective does not penalize minority group mistakes.
3. **The Importance of Distributionally Robust Optimization**: Combining targeted augmentations with group-reweighted cross-entropy loss boosts worst-group accuracy to **58 % (Level 1)**, **63 % (Level 2)**, and **55 % (Level 3)**, successfully clearing the level targets.

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

- **Group Label Requirement**: The group-reweighted loss uses group labels (true class × shortcut presence), which resembles Group DRO. In unconstrained real-world settings, group labels are rarely available without explicit metadata or unsupervised discovery.
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
To rerun the full 36-experiment empirical evaluation and regenerate all figures:
```bash
python results.py
```

### 4. Run the Test Suite
```bash
python tests/test_basics.py
pytest tests/ -v
```

<a id="test-status"></a>
**Test Status**: All 32 automated unit and integration tests passing (`23/23` basic checks + `9/9` game loop tests).

---

## Memory Budget & Resource Profile

Measured on CPU via `python measure_memory.py` using `tracemalloc`:
- **Peak RAM (Dataset Load):** ~663 MB
- **Peak RAM (Training Run):** ~1.2 MB
- **Peak RAM (Grad-CAM):** ~1.4 MB
- **Overall Peak:** **~663 MB** (well under the 800 MB constraint for 1 GB free-tier instances like Streamlit Cloud).

---

## Player Controls Guide

### Fine-Tuning Depth
- `head_only`: Fast training, freezes all ResNet layers except the final linear classifier.
- `last_block`: Unfreezes `layer4` + FC layer. Allows high-level feature extraction adjustments.
- `full`: Unfreezes all layers.

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
├── app.py                 – Gradio UI entry point
├── data.py                – CIFAR-10 dataset builder, shortcut injection, and group IDs
├── model.py               – Pretrained ResNet18 wrapper with freeze modes
├── train.py               – Training engine, augmentations, loss functions, and metrics
├── gradcam.py             – Hook-based Grad-CAM implementation
├── results.py             – Multi-seed experimental evaluation script
├── measure_memory.py      – RAM profiling script
├── requirements.txt       – Pinned dependencies with PyTorch CPU wheel index
├── LICENSE                – MIT License
├── .streamlit/
│   └── config.toml        – Headless configuration and telemetry settings
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
