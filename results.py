"""
results.py  –  Reproducible measurement script for Clever Hans Hunter.

For each level (1, 2, 3) and matched conditions:
  - baseline_head: head_only, lr 1e-3, plain CE, no aug (existing baseline)
  - ctrl_lb: last_block, lr 5e-4, plain CE, no aug (matched control)
  - aug_only: last_block, lr 5e-4, plain CE, standard aug
  - loss_only: last_block, lr 5e-4, group-reweighted CE, no aug
  - fix: last_block, lr 5e-4, group-reweighted CE, standard aug
  - oracle: same config as fix, trained with build_datasets(oracle=True)
  - strong_aug_only: last_block, lr 5e-4, plain CE, shortcut-specific strong aug
  - strong_aug_loss: last_block, lr 5e-4, group-reweighted CE, shortcut-specific strong aug

Reports val, test avg, per-group and worst-group, mean ± std over 3 seeds,
with Wilson 95% CIs for group accuracies.

Outputs:
  results/results.json          raw numbers
  docs/level{N}_groups.png      per-group bar charts
  docs/level{N}_comparison.png  baseline vs fix comparison
  docs/level{N}_gradcam_*.png   before/after Grad-CAM for one sample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))

from data import (
    CLASS_NAMES,
    DATA_ROOT,
    IMAGE_SIZE,
    LEVEL_TARGETS,
    build_datasets,
    get_normalize_transform,
    normalize_tensor,
)
from gradcam import _tensor_to_pil, make_gradcam_panel
from train import (
    TrainConfig,
    build_augmentation_transform,
    run_training,
    train_baseline,
)

RESULTS_DIR = Path("results")
DOCS_DIR = Path("docs")
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

GROUP_LABELS = ["Cat\n(no SC)", "Cat\n(+SC)", "Dog\n(no SC)", "Dog\n(+SC)"]

_BG = "#0f1117"
_CARD = "#1a1d27"
_ACC = "#6c63ff"
_GRN = "#4ade80"
_YEL = "#facc15"
_RED = "#f87171"
_CYN = "#38bdf8"
_PUR = "#c084fc"
_TXT = "#e2e8f0"
_BRD = "#2a2d3e"


def wilson_ci(k: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    """Calculate Wilson 95% score interval for binomial proportions."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p)) / n + (z**2) / (4 * (n**2)))) / denom
    return (max(0.0, float(center - margin)), min(1.0, float(center + margin)))


class BatchRandomGrayscale(nn.Module):
    """Strong aug for Level 1: 100% grayscale (destroys color tint shortcut)."""

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.transform = T.RandomGrayscale(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class EraseTopLeft(nn.Module):
    """Strong aug for Level 2: erases top-left corner with probability >= 0.8."""

    def __init__(self, p: float = 0.85, patch_size: int = 8):
        super().__init__()
        self.p = p
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.shape[0], device=x.device) < self.p
        x = x.clone()
        x[mask, :, : self.patch_size, : self.patch_size] = 0.0
        return x


class BatchGaussianBlur(nn.Module):
    """Strong aug for Level 3: Gaussian blur with sigma >= 2.0 (attenuates stripes)."""

    def __init__(self, kernel_size: int = 7, sigma: float = 2.5):
        super().__init__()
        self.transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


def get_strong_aug(level: int) -> nn.Module:
    """Return the shortcut-specific strong augmentation for each level."""
    if level == 1:
        return BatchRandomGrayscale(p=1.0)
    elif level == 2:
        return EraseTopLeft(p=0.85, patch_size=8)
    else:
        return BatchGaussianBlur(kernel_size=7, sigma=2.5)


class DeterministicGrayscale(nn.Module):
    """Deterministic preprocessing for Level 1: converts to 3-channel grayscale at train and eval."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        return gray.expand(-1, 3, -1, -1)


class DeterministicMaskCorner(nn.Module):
    """Deterministic preprocessing for Level 2: zeroes out top-left 8x8 corner at train and eval."""

    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, :, : self.patch_size, : self.patch_size] = 0.0
        return x


class DeterministicGaussianBlur(nn.Module):
    """Deterministic preprocessing for Level 3: Gaussian blur sigma=2.5 at train and eval."""

    def __init__(self, kernel_size: int = 7, sigma: float = 2.5):
        super().__init__()
        self.transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


def get_preproc_transform(level: int) -> nn.Module:
    """Return the deterministic cue-removing preprocessing transform for each level."""
    if level == 1:
        return DeterministicGrayscale()
    elif level == 2:
        return DeterministicMaskCorner(patch_size=8)
    else:
        return DeterministicGaussianBlur(kernel_size=7, sigma=2.5)


def _baseline_head_cfg(seed: int) -> TrainConfig:
    return TrainConfig(
        finetune_mode="head_only",
        optimizer="adam",
        learning_rate=1e-3,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=seed,
    )


def _ctrl_lb_cfg(seed: int) -> TrainConfig:
    """Matched control: last_block, lr 5e-4, plain CE, no augmentation."""
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=seed,
    )


def _aug_only_cfg(level: int, seed: int) -> TrainConfig:
    """Augmentation only, plain CE (ablation)."""
    base = dict(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=seed,
    )
    if level == 1:
        return TrainConfig(
            **base,
            use_color_jitter=True,
            use_random_gray=True,
            aug_strength=0.6,
        )
    elif level == 2:
        return TrainConfig(
            **base,
            use_random_erasing=True,
            use_random_crop=True,
            aug_strength=0.6,
        )
    else:
        return TrainConfig(
            **base,
            use_random_gray=True,
            use_gaussian_noise=True,
            aug_strength=0.6,
        )


def _loss_only_cfg(level: int, seed: int) -> TrainConfig:
    """Group-reweighted loss only, no augmentation (ablation)."""
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="group_reweighted_ce",
        seed=seed,
    )


def _fix_cfg(level: int, seed: int) -> TrainConfig:
    """README recommended fix per level (group-reweighted ERM + standard aug)."""
    base = dict(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="group_reweighted_ce",
        seed=seed,
    )
    if level == 1:
        return TrainConfig(
            **base,
            use_color_jitter=True,
            use_random_gray=True,
            aug_strength=0.6,
        )
    elif level == 2:
        return TrainConfig(
            **base,
            use_random_erasing=True,
            use_random_crop=True,
            aug_strength=0.6,
        )
    else:
        return TrainConfig(
            **base,
            use_random_gray=True,
            use_gaussian_noise=True,
            aug_strength=0.6,
        )


def _oracle_plain_cfg(seed: int) -> TrainConfig:
    """Fixed oracle condition: last_block, lr 5e-4, plain CE, NO augmentation, trained on oracle data."""
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=seed,
    )


def _oracle_cfg(level: int, seed: int) -> TrainConfig:
    """Oracle condition with fix augmentations, trained on shortcut-uncorrelated data."""
    return _fix_cfg(level, seed)


def _strong_aug_only_cfg(level: int, seed: int) -> TrainConfig:
    """Strong augmentation only, plain CE (uses knowledge of shortcut, train-only)."""
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=seed,
        custom_aug_transform=get_strong_aug(level),
    )


def _strong_aug_loss_cfg(level: int, seed: int) -> TrainConfig:
    """Strong augmentation + group-reweighted loss (uses knowledge of shortcut, train-only)."""
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="group_reweighted_ce",
        seed=seed,
        custom_aug_transform=get_strong_aug(level),
    )


def _pre_only_cfg(level: int, seed: int) -> TrainConfig:
    """Preprocessing ablation: cue-removing transform at BOTH train and eval, plain CE."""
    tf = get_preproc_transform(level)
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=seed,
        custom_aug_transform=tf,
        eval_transform=tf,
    )


def _pre_loss_cfg(level: int, seed: int) -> TrainConfig:
    """Preprocessing ablation: cue-removing transform at BOTH train and eval, group-reweighted CE."""
    tf = get_preproc_transform(level)
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="group_reweighted_ce",
        seed=seed,
        custom_aug_transform=tf,
        eval_transform=tf,
    )


CONDITIONS = {
    "baseline_head": _baseline_head_cfg,
    "baseline": _baseline_head_cfg,
    "ctrl_lb": _ctrl_lb_cfg,
    "aug_only": _aug_only_cfg,
    "loss_only": _loss_only_cfg,
    "fix": _fix_cfg,
    "oracle_plain": _oracle_plain_cfg,
    "oracle": _oracle_cfg,
    "strong_aug_only": _strong_aug_only_cfg,
    "strong_aug_loss": _strong_aug_loss_cfg,
    "pre_only": _pre_only_cfg,
    "pre_loss": _pre_loss_cfg,
}

TABLE_CONDITIONS = [
    ("baseline_head", "Baseline (head only)"),
    ("ctrl_lb", "Ctrl (last block, no aug)"),
    ("aug_only", "Aug Only"),
    ("loss_only", "Loss Only"),
    ("fix", "Recommended Fix"),
    ("oracle_plain", "Oracle (plain CE, no aug)"),
    ("oracle", "Oracle (with fix aug)"),
    ("strong_aug_only", "Strong Aug Only (train-only)"),
    ("strong_aug_loss", "Strong Aug + Loss (train-only)"),
    ("pre_only", "Pre-processing Only (train & eval)"),
    ("pre_loss", "Pre-processing + Loss (train & eval)"),
]


def audit_cue_survival(level: int, n_samples: int = 200, n_draws: int = 10) -> dict:
    """
    Measure on 200 augmented training images how often the planted shortcut cue survives,
    averaged over n_draws random augmentation draws.
    - Level 1: mean blue-minus-red channel gap per class.
    - Level 2: mean blue-minus-red channel gap in top-left 8x8 region per class.
    - Level 3: stripe-frequency energy before vs after.
    """
    tr_ds, _, _ = build_datasets(level, seed=42)
    cat_idxs = [i for i, l in enumerate(tr_ds.labels) if l == 0][: n_samples // 2]
    dog_idxs = [i for i, l in enumerate(tr_ds.labels) if l == 1][: n_samples // 2]
    idxs = cat_idxs + dog_idxs

    imgs = torch.stack([tr_ds[i][0].float() / 255.0 for i in idxs])
    lbls = torch.tensor([tr_ds[i][1] for i in idxs])

    std_cfg = _aug_only_cfg(level, 42)
    strong_aug = get_strong_aug(level)

    report = {"level": level, "n_samples": len(idxs), "n_draws": n_draws}

    if level == 1:
        cat_bmr_before = (imgs[lbls == 0][:, 2] - imgs[lbls == 0][:, 0]).mean().item()
        dog_bmr_before = (imgs[lbls == 1][:, 2] - imgs[lbls == 1][:, 0]).mean().item()
        gap_before = cat_bmr_before - dog_bmr_before

        std_gaps, strong_gaps = [], []
        for draw in range(n_draws):
            torch.manual_seed(42 + draw)
            std_aug = build_augmentation_transform(
                use_color_jitter=std_cfg.use_color_jitter,
                use_random_gray=std_cfg.use_random_gray,
                use_random_crop=std_cfg.use_random_crop,
                use_random_erasing=std_cfg.use_random_erasing,
                use_hflip=std_cfg.use_hflip,
                use_gaussian_noise=std_cfg.use_gaussian_noise,
                aug_strength=std_cfg.aug_strength,
                image_size=std_cfg.image_size,
            )
            aug_std = std_aug(imgs) if std_aug is not None else imgs.clone()
            cat_std = (aug_std[lbls == 0][:, 2] - aug_std[lbls == 0][:, 0]).mean().item()
            dog_std = (aug_std[lbls == 1][:, 2] - aug_std[lbls == 1][:, 0]).mean().item()
            std_gaps.append(cat_std - dog_std)

            torch.manual_seed(42 + draw)
            aug_strong = strong_aug(imgs)
            cat_str = (aug_strong[lbls == 0][:, 2] - aug_strong[lbls == 0][:, 0]).mean().item()
            dog_str = (aug_strong[lbls == 1][:, 2] - aug_strong[lbls == 1][:, 0]).mean().item()
            strong_gaps.append(cat_str - dog_str)

        gap_std = sum(std_gaps) / len(std_gaps)
        gap_strong = sum(strong_gaps) / len(strong_gaps)

        report.update({
            "metric": "mean_blue_minus_red_gap",
            "before": round(gap_before, 4),
            "after_std": round(gap_std, 4),
            "std_retention_pct": round((gap_std / gap_before) * 100, 1) if gap_before else 0.0,
            "after_strong": round(gap_strong, 4),
            "strong_retention_pct": round((gap_strong / gap_before) * 100, 1) if gap_before else 0.0,
            "details": f"B-R gap: {gap_before:.4f} -> {gap_std:.4f} ({gap_std/gap_before*100:.1f}%) -> {gap_strong:.4f} ({gap_strong/gap_before*100:.1f}%)",
        })
    elif level == 2:
        patch = imgs[:, :, :8, :8]
        cat_bmr_before = (patch[lbls == 0][:, 2] - patch[lbls == 0][:, 0]).mean().item()
        dog_bmr_before = (patch[lbls == 1][:, 2] - patch[lbls == 1][:, 0]).mean().item()
        gap_before = cat_bmr_before - dog_bmr_before

        std_gaps, strong_gaps = [], []
        for draw in range(n_draws):
            torch.manual_seed(42 + draw)
            std_aug = build_augmentation_transform(
                use_color_jitter=std_cfg.use_color_jitter,
                use_random_gray=std_cfg.use_random_gray,
                use_random_crop=std_cfg.use_random_crop,
                use_random_erasing=std_cfg.use_random_erasing,
                use_hflip=std_cfg.use_hflip,
                use_gaussian_noise=std_cfg.use_gaussian_noise,
                aug_strength=std_cfg.aug_strength,
                image_size=std_cfg.image_size,
            )
            aug_std = std_aug(imgs) if std_aug is not None else imgs.clone()
            p_std = aug_std[:, :, :8, :8]
            cat_std = (p_std[lbls == 0][:, 2] - p_std[lbls == 0][:, 0]).mean().item()
            dog_std = (p_std[lbls == 1][:, 2] - p_std[lbls == 1][:, 0]).mean().item()
            std_gaps.append(cat_std - dog_std)

            torch.manual_seed(42 + draw)
            aug_strong = strong_aug(imgs)
            p_str = aug_strong[:, :, :8, :8]
            cat_str = (p_str[lbls == 0][:, 2] - p_str[lbls == 0][:, 0]).mean().item()
            dog_str = (p_str[lbls == 1][:, 2] - p_str[lbls == 1][:, 0]).mean().item()
            strong_gaps.append(cat_str - dog_str)

        gap_std = sum(std_gaps) / len(std_gaps)
        gap_strong = sum(strong_gaps) / len(strong_gaps)

        report.update({
            "metric": "top_left_8x8_blue_minus_red_gap",
            "before": round(gap_before, 4),
            "after_std": round(gap_std, 4),
            "std_retention_pct": round((gap_std / gap_before) * 100, 1) if gap_before else 0.0,
            "after_strong": round(gap_strong, 4),
            "strong_retention_pct": round((gap_strong / gap_before) * 100, 1) if gap_before else 0.0,
            "details": f"Patch B-R gap: {gap_before:.4f} -> {gap_std:.4f} ({gap_std/gap_before*100:.1f}%) -> {gap_strong:.4f} ({gap_strong/gap_before*100:.1f}%)",
        })
    else:
        def stripe_energy(img_batch, lbl_batch):
            energies = []
            for img, lbl in zip(img_batch, lbl_batch):
                gray = img.mean(dim=0)
                fft = torch.fft.fft2(gray)
                mag2 = fft.abs() ** 2
                if lbl == 0:
                    e = mag2[16, 0].item() + mag2[-16, 0].item()
                else:
                    e = mag2[0, 16].item() + mag2[0, -16].item()
                energies.append(e)
            return sum(energies) / len(energies)

        e_before = stripe_energy(imgs, lbls)
        std_gaps, strong_gaps = [], []
        for draw in range(n_draws):
            torch.manual_seed(42 + draw)
            std_aug = build_augmentation_transform(
                use_color_jitter=std_cfg.use_color_jitter,
                use_random_gray=std_cfg.use_random_gray,
                use_random_crop=std_cfg.use_random_crop,
                use_random_erasing=std_cfg.use_random_erasing,
                use_hflip=std_cfg.use_hflip,
                use_gaussian_noise=std_cfg.use_gaussian_noise,
                aug_strength=std_cfg.aug_strength,
                image_size=std_cfg.image_size,
            )
            aug_std = std_aug(imgs) if std_aug is not None else imgs.clone()
            std_gaps.append(stripe_energy(aug_std, lbls))

            torch.manual_seed(42 + draw)
            aug_strong = strong_aug(imgs)
            strong_gaps.append(stripe_energy(aug_strong, lbls))

        e_std = sum(std_gaps) / len(std_gaps)
        e_strong = sum(strong_gaps) / len(strong_gaps)

        report.update({
            "metric": "stripe_frequency_energy_fft2",
            "before": round(e_before, 2),
            "after_std": round(e_std, 2),
            "std_retention_pct": round((e_std / e_before) * 100, 1) if e_before else 0.0,
            "after_strong": round(e_strong, 2),
            "strong_retention_pct": round((e_strong / e_before) * 100, 1) if e_before else 0.0,
            "details": f"Energy: {e_before:.1f} -> {e_std:.1f} ({e_std/e_before*100:.1f}%) -> {e_strong:.1f} ({e_strong/e_before*100:.1f}%)",
        })

    return report


def run_condition(
    level: int,
    condition: str,
    seeds: list,
    epochs_override: int | None = None,
) -> tuple[dict, object]:
    """Run condition over all seeds and return aggregated stats dict and first result."""
    is_oracle = condition in ("oracle", "oracle_plain")
    tr_ds, va_ds, te_ds = build_datasets(level, seed=42, oracle=is_oracle)

    rows = []
    first_result = None

    for seed in seeds:
        cfg_fn = CONDITIONS[condition]
        if condition in ("baseline", "baseline_head", "ctrl_lb", "oracle_plain"):
            cfg = cfg_fn(seed)
        else:
            cfg = cfg_fn(level, seed)

        if epochs_override is not None:
            cfg.epochs = epochs_override

        t0 = time.time()
        result = run_training(cfg, tr_ds, va_ds, te_ds)
        if first_result is None:
            first_result = result
        elapsed = time.time() - t0

        g_counts = {str(g): list(result.group_counts.get(g, (0, 0))) for g in range(4)}

        row = {
            "seed": seed,
            "val_acc": result.val_acc,
            "test_avg": result.test_avg_acc,
            "wg_acc": result.worst_group_acc,
            "group_accs": {str(g): result.group_accs.get(g, float("nan")) for g in range(4)},
            "group_counts": g_counts,
            "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"  [{condition:16s} seed={seed:2d}] wg={result.worst_group_acc:.3f} "
            f"val={result.val_acc:.3f} test_avg={result.test_avg_acc:.3f} elapsed={elapsed:.0f}s"
        )

    stats = build_stats_summary(level, condition, seeds, rows)
    return stats, first_result


def build_stats_summary(level: int, condition: str, seeds: list, rows: list) -> dict:
    """Compute summary statistics and unpooled Wilson CI ranges across seeds."""
    wg_vals = [r["wg_acc"] for r in rows]
    va_vals = [r["val_acc"] for r in rows]
    ta_vals = [r["test_avg"] for r in rows]

    group_summary = {}
    for g in range(4):
        accs = [r["group_accs"][str(g)] for r in rows if str(g) in r["group_accs"]]
        accs = [v for v in accs if not np.isnan(v)]

        cis_g = []
        for r in rows:
            if str(g) in r.get("group_counts", {}):
                k, n = r["group_counts"][str(g)]
                if n > 0:
                    cis_g.append(wilson_ci(k, n))

        if cis_g:
            ci_min_lo = min(c[0] for c in cis_g)
            ci_max_hi = max(c[1] for c in cis_g)
            ci_range = [round(ci_min_lo, 4), round(ci_max_hi, 4)]
        else:
            ci_range = [0.0, 0.0]

        group_summary[str(g)] = {
            "mean": float(np.mean(accs)) if accs else float("nan"),
            "std": float(np.std(accs)) if accs else 0.0,
            "ci_95": ci_range,
            "cis_per_seed": [[round(c[0], 4), round(c[1], 4)] for c in cis_g],
        }

    wg_cis = []
    for r in rows:
        min_g = None
        min_acc = 1.0
        for g in range(4):
            acc = r["group_accs"].get(str(g), float("nan"))
            if not np.isnan(acc) and acc <= min_acc:
                min_acc = acc
                min_g = str(g)
        if min_g is not None and min_g in r.get("group_counts", {}):
            k, n = r["group_counts"][min_g]
            if n > 0:
                wg_cis.append(wilson_ci(k, n))

    if wg_cis:
        wg_ci_range = [round(min(c[0] for c in wg_cis), 4), round(max(c[1] for c in wg_cis), 4)]
    else:
        wg_ci_range = [0.0, 0.0]

    return {
        "level": level,
        "condition": condition,
        "seeds": seeds,
        "rows": rows,
        "summary": {
            "val_acc": {"mean": float(np.mean(va_vals)), "std": float(np.std(va_vals))},
            "test_avg": {"mean": float(np.mean(ta_vals)), "std": float(np.std(ta_vals))},
            "wg_acc": {
                "mean": float(np.mean(wg_vals)),
                "std": float(np.std(wg_vals)),
                "ci_95": wg_ci_range,
                "cis_per_seed": [[round(c[0], 4), round(c[1], 4)] for c in wg_cis],
            },
            "group_accs": group_summary,
        },
    }


def _style(fig):
    fig.patch.set_facecolor(_BG)
    for ax in fig.axes:
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_TXT)
        ax.xaxis.label.set_color(_TXT)
        ax.yaxis.label.set_color(_TXT)
        ax.title.set_color(_TXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BRD)
    return fig


def plot_group_bars(stats: dict, level: int, path: Path):
    """Per-group accuracy bar chart with error bars for the matched controls."""
    conditions = [
        "baseline_head", "ctrl_lb", "aug_only", "loss_only", "fix",
        "oracle_plain", "pre_only", "pre_loss"
    ]
    cond_labels = [
        "Baseline (head)", "Ctrl (last block)", "Aug Only", "Loss Only", "Fix",
        "Oracle (plain)", "Pre Only", "Pre Loss"
    ]
    colors = [_RED, "#fb923c", _YEL, _ACC, _GRN, _CYN, "#a78bfa", "#f472b6"]
    target = LEVEL_TARGETS[level]

    x = np.arange(4)
    w = 0.10
    fig, ax = plt.subplots(figsize=(12, 4))

    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        key = f"L{level}_{cond}"
        if key not in stats and cond == "baseline_head":
            key = f"L{level}_baseline"
        if key not in stats:
            continue
        ga = stats[key]["summary"]["group_accs"]
        means = [ga[str(g)]["mean"] for g in range(4)]
        stds = [ga[str(g)]["std"] for g in range(4)]
        ax.bar(
            x + (i - 3.5) * w,
            means,
            w,
            label=label,
            color=color,
            alpha=0.85,
            yerr=stds,
            capsize=3,
            zorder=2,
        )

    ax.axhline(
        target,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=f"Target {target:.0%}",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Level {level} – Per-Group Accuracy (Matched Controls)")
    ax.legend(facecolor=_CARD, edgecolor=_BRD, labelcolor=_TXT, fontsize=8)
    ax.grid(axis="y", color=_BRD, linestyle="--", zorder=1)
    _style(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison(stats: dict, level: int, path: Path):
    """Bar chart comparing val acc / test avg / wg acc across conditions."""
    conditions = [
        "baseline_head", "ctrl_lb", "aug_only", "loss_only", "fix",
        "oracle_plain", "pre_only", "pre_loss"
    ]
    cond_labels = [
        "Baseline (head)", "Ctrl (LB)", "Aug Only", "Loss Only", "Fix",
        "Oracle (plain)", "Pre Only", "Pre Loss"
    ]
    colors = [_RED, "#fb923c", _YEL, _ACC, _GRN, _CYN, "#a78bfa", "#f472b6"]
    metrics = ["val_acc", "test_avg", "wg_acc"]
    m_labels = ["Val Acc", "Test Avg", "Worst-Group"]
    target = LEVEL_TARGETS[level]

    x = np.arange(len(metrics))
    w = 0.10
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        key = f"L{level}_{cond}"
        if key not in stats and cond == "baseline_head":
            key = f"L{level}_baseline"
        if key not in stats:
            continue
        s = stats[key]["summary"]
        means = [s[m]["mean"] for m in metrics]
        stds = [s[m]["std"] for m in metrics]
        ax.bar(
            x + (i - 3.5) * w,
            means,
            w,
            label=label,
            color=color,
            alpha=0.85,
            yerr=stds,
            capsize=3,
            zorder=2,
        )

    ax.axhline(
        target,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=f"WG target {target:.0%}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Level {level} – Matched Controls Comparison")
    ax.legend(facecolor=_CARD, edgecolor=_BRD, labelcolor=_TXT, fontsize=8)
    ax.grid(axis="y", color=_BRD, linestyle="--", zorder=1)
    _style(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def save_gradcam_examples(
    level: int,
    before_result,
    after_result,
    test_ds,
    docs_dir: Path,
):
    """Save Grad-CAM panels for one test sample (before / after fix)."""
    if before_result is None or before_result.model is None:
        return

    sample_idx = 0
    if before_result.suspicious:
        sample_idx = before_result.suspicious[0][0]
    if sample_idx >= len(test_ds):
        sample_idx = 0

    raw_img, true_lbl, group_id = test_ds[sample_idx]
    img_norm = normalize_tensor(raw_img.unsqueeze(0))

    for tag, result in [("before", before_result), ("after", after_result)]:
        if result is None or result.model is None:
            continue
        try:
            orig, heat, overlay, pred_cls, conf = make_gradcam_panel(
                result.model, img_norm, raw_img=raw_img
            )
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for ax, img, title in zip(axes, [orig, heat, overlay], ["Original", "Heatmap", "Overlay"]):
                ax.imshow(img)
                ax.set_title(title, color=_TXT)
                ax.axis("off")
            correct = "Correct" if pred_cls == true_lbl else "Wrong"
            fig.suptitle(
                f"Level {level} {tag.capitalize()} | "
                f"True:{CLASS_NAMES[true_lbl]} "
                f"Pred:{CLASS_NAMES[pred_cls]} ({conf:.0%}) {correct}",
                color=_TXT,
                fontsize=9,
            )
            fig.patch.set_facecolor(_BG)
            path = docs_dir / f"level{level}_gradcam_{tag}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")
        except Exception as e:
            print(f"  Grad-CAM {tag} error: {e}")


def print_table(all_stats: dict):
    print("\n" + "=" * 125)
    print(
        f"{'Level':>5} | {'Condition':>26} | {'Val Acc':>8} | {'Test Avg':>8} | "
        f"{'WG Acc (95% CI)':>18} | {'Group 0 (Cat no-SC)':>14} | {'Group 1 (Cat +SC)':>14} | "
        f"{'Group 2 (Dog no-SC)':>14} | {'Group 3 (Dog +SC)':>14}"
    )
    print("-" * 125)
    for level in [1, 2, 3]:
        for cond, label in TABLE_CONDITIONS:
            key = f"L{level}_{cond}"
            if key not in all_stats and cond == "baseline_head":
                key = f"L{level}_baseline"
            if key not in all_stats:
                continue
            s = all_stats[key]["summary"]
            ga = s["group_accs"]

            def fmt(m):
                return f"{m['mean']:.2f}±{m['std']:.2f}"

            def fmt_grp(g):
                info = ga.get(str(g), {})
                m = info.get("mean", float("nan"))
                ci = info.get("ci_95", [0.0, 0.0])
                if np.isnan(m):
                    return "      n/a     "
                return f"{m:.2f} [{ci[0]:.2f},{ci[1]:.2f}]"

            wg_ci = s["wg_acc"].get("ci_95", [0.0, 0.0])
            wg_str = f"{fmt(s['wg_acc'])} [{wg_ci[0]:.2f},{wg_ci[1]:.2f}]"

            print(
                f"  {level:>3}  | {label:>26} | {fmt(s['val_acc']):>8} | "
                f"{fmt(s['test_avg']):>8} | {wg_str:>18} | "
                f"{fmt_grp(0):>14} | {fmt_grp(1):>14} | "
                f"{fmt_grp(2):>14} | {fmt_grp(3):>14}"
            )
        print("-" * 125)
    print("=" * 125 + "\n")


def print_markdown_table(all_stats: dict):
    print("\n### Markdown Table for README.md:\n")
    print("| Level | Condition | Val Acc | Test Avg | WG Acc (95% CI) | G0 Cat (no SC) | G1 Cat (+SC) | G2 Dog (no SC) | G3 Dog (+SC) |")
    print("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for level in [1, 2, 3]:
        for cond, label in TABLE_CONDITIONS:
            key = f"L{level}_{cond}"
            if key not in all_stats and cond == "baseline_head":
                key = f"L{level}_baseline"
            if key not in all_stats:
                print(f"| **{level}** | {label} | — | — | — | — | — | — | — |")
                continue
            s = all_stats[key]["summary"]
            ga = s["group_accs"]
            val_str = f"{s['val_acc']['mean']:.2f} ± {s['val_acc']['std']:.2f}"
            tst_str = f"{s['test_avg']['mean']:.2f} ± {s['test_avg']['std']:.2f}"
            wg_ci = s["wg_acc"].get("ci_95", [0.0, 0.0])
            wg_str = f"**{s['wg_acc']['mean']:.2f} ± {s['wg_acc']['std']:.2f}** [{wg_ci[0]:.2f}, {wg_ci[1]:.2f}]"

            def fmt_g(gid):
                info = ga.get(str(gid), {})
                m = info.get("mean", float("nan"))
                ci = info.get("ci_95", [0.0, 0.0])
                if np.isnan(m):
                    return "—"
                return f"{m:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"

            print(f"| **{level}** | {label} | {val_str} | {tst_str} | {wg_str} | {fmt_g(0)} | {fmt_g(1)} | {fmt_g(2)} | {fmt_g(3)} |")
    print()


def check_targets(all_stats: dict) -> list:
    warnings = []
    for level in [1, 2, 3]:
        target = LEVEL_TARGETS[level]
        fix_key = f"L{level}_fix"
        if fix_key in all_stats:
            wg_mean = all_stats[fix_key]["summary"]["wg_acc"]["mean"]
            if wg_mean < target:
                warnings.append(
                    f"INFO  Level {level}: recommended fix achieves WG={wg_mean:.3f} "
                    f"< target {target:.2f} (misses target)."
                )
            else:
                warnings.append(
                    f"INFO  Level {level}: recommended fix achieves WG={wg_mean:.3f} "
                    f">= target {target:.2f} (clears target)."
                )
        loss_key = f"L{level}_loss_only"
        if loss_key in all_stats:
            wg_mean = all_stats[loss_key]["summary"]["wg_acc"]["mean"]
            if wg_mean >= target:
                warnings.append(
                    f"INFO  Level {level}: loss_only achieves WG={wg_mean:.3f} "
                    f">= target {target:.2f} (reaches target)."
                )
    return warnings


def save_level_results(level: int, level_stats: dict, cue_audit: dict):
    path = RESULTS_DIR / f"results_L{level}.json"
    data = dict(level_stats)
    if cue_audit:
        data["_cue_audit_reports"] = {str(level): cue_audit}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def merge_all_results() -> dict:
    merged = {}
    cue_audits = {}
    for lvl in [1, 2, 3]:
        p = RESULTS_DIR / f"results_L{lvl}.json"
        if p.exists():
            try:
                with open(p, "r") as f:
                    d = json.load(f)
                if "_cue_audit_reports" in d:
                    cue_audits.update(d.pop("_cue_audit_reports"))
                for k, v in d.items():
                    if isinstance(v, dict) and "rows" in v and "seeds" in v:
                        v["summary"] = build_stats_summary(
                            v.get("level", lvl),
                            v.get("condition", k),
                            v.get("seeds", []),
                            v["rows"],
                        )["summary"]
                merged.update(d)
                print(f"Loaded {p} ({len(d)} conditions)")
            except Exception as e:
                print(f"Error loading {p}: {e}")
    merged["_cue_audit_reports"] = cue_audits
    main_path = RESULTS_DIR / "results.json"
    with open(main_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged all per-level results into {main_path} ({len(merged)} total keys)")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for all conditions (useful for quick tests)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        type=str,
        default=None,
        help="Run only specific conditions",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conditions that already have all requested seeds in results_L{level}.json",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results_L*.json into results.json and generate table without running training",
    )
    args = parser.parse_args()

    if args.merge:
        merged = merge_all_results()
        for lvl in [1, 2, 3]:
            if any(k.startswith(f"L{lvl}_") for k in merged):
                plot_group_bars(merged, lvl, DOCS_DIR / f"level{lvl}_groups.png")
                plot_comparison(merged, lvl, DOCS_DIR / f"level{lvl}_comparison.png")
        print_table(merged)
        print_markdown_table(merged)
        for w in check_targets(merged):
            print(w)
        return

    all_conditions = [
        "baseline_head",
        "ctrl_lb",
        "aug_only",
        "loss_only",
        "fix",
        "oracle_plain",
        "oracle",
        "strong_aug_only",
        "strong_aug_loss",
        "pre_only",
        "pre_loss",
    ]
    conditions_to_run = args.conditions if args.conditions else all_conditions

    for level in args.levels:
        level_file = RESULTS_DIR / f"results_L{level}.json"
        level_stats = {}
        level_cue_audit = {}
        if level_file.exists():
            try:
                with open(level_file, "r") as f:
                    level_stats = json.load(f)
                if "_cue_audit_reports" in level_stats:
                    level_cue_audit = level_stats.pop("_cue_audit_reports").get(str(level), {})
                print(f"Loaded existing level {level} results from {level_file} ({len(level_stats)} keys)")
            except Exception as e:
                print(f"Warning: could not load existing {level_file}: {e}")

        print("\n" + "=" * 60)
        print(f"  AUGMENTATION CUE SURVIVAL AUDIT: LEVEL {level} (200 SAMPLES)")
        print("=" * 60)
        cue_report = audit_cue_survival(level, n_samples=200)
        level_cue_audit = cue_report
        print(f"Level {level} Cue Audit: {cue_report['details']}")

        print(f"\n{'='*60}")
        print(f"  LEVEL {level} MATCHED CONTROLS & EXPERIMENTS")
        print(f"{'='*60}")

        tr_ds, va_ds, te_ds = build_datasets(level, seed=42)
        base_r = None
        fix_r = None

        for cond in conditions_to_run:
            key = f"L{level}_{cond}"
            if args.skip_existing and key in level_stats:
                existing_seeds = set(level_stats[key].get("seeds", []))
                if set(args.seeds).issubset(existing_seeds):
                    print(f"\n  -- {cond} (already present in {level_file.name}, skipping) --")
                    continue

            print(f"\n  -- {cond} --")
            stats, res0 = run_condition(
                level, cond, args.seeds, epochs_override=args.epochs
            )
            level_stats[key] = stats
            if cond == "baseline_head":
                level_stats[f"L{level}_baseline"] = stats
                base_r = res0
            elif cond == "fix":
                fix_r = res0

            save_level_results(level, level_stats, level_cue_audit)
            print(f"  [Checkpoint saved to {level_file}]")

        plot_group_bars(level_stats, level, DOCS_DIR / f"level{level}_groups.png")
        plot_comparison(level_stats, level, DOCS_DIR / f"level{level}_comparison.png")
        if base_r is not None and fix_r is not None:
            save_gradcam_examples(level, base_r, fix_r, te_ds, DOCS_DIR)

    merged = merge_all_results()
    print_table(merged)
    print_markdown_table(merged)

    warnings = check_targets(merged)
    if warnings:
        print("\n".join(warnings))


if __name__ == "__main__":
    main()

