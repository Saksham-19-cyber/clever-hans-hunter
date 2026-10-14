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


def _oracle_cfg(level: int, seed: int) -> TrainConfig:
    """Oracle condition: same config as fix, trained on shortcut-uncorrelated data."""
    return _fix_cfg(level, seed)


def _strong_aug_only_cfg(level: int, seed: int) -> TrainConfig:
    """Strong augmentation only, plain CE (uses knowledge of shortcut)."""
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
    """Strong augmentation + group-reweighted loss (uses knowledge of shortcut)."""
    return TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="group_reweighted_ce",
        seed=seed,
        custom_aug_transform=get_strong_aug(level),
    )


CONDITIONS = {
    "baseline_head": _baseline_head_cfg,
    "baseline": _baseline_head_cfg,
    "ctrl_lb": _ctrl_lb_cfg,
    "aug_only": _aug_only_cfg,
    "loss_only": _loss_only_cfg,
    "fix": _fix_cfg,
    "oracle": _oracle_cfg,
    "strong_aug_only": _strong_aug_only_cfg,
    "strong_aug_loss": _strong_aug_loss_cfg,
}


def audit_cue_survival(level: int, n_samples: int = 200) -> dict:
    """
    Measure on 200 augmented training images how often the planted shortcut cue survives.
    - Level 1: mean blue-minus-red channel gap per class.
    - Level 2: fraction of images whose top-left 8x8 patch is unchanged.
    - Level 3: stripe-frequency energy before vs after.
    """
    tr_ds, _, _ = build_datasets(level, seed=42)
    cat_idxs = [i for i, l in enumerate(tr_ds.labels) if l == 0][: n_samples // 2]
    dog_idxs = [i for i, l in enumerate(tr_ds.labels) if l == 1][: n_samples // 2]
    idxs = cat_idxs + dog_idxs

    imgs = torch.stack([tr_ds[i][0].float() / 255.0 for i in idxs])
    lbls = torch.tensor([tr_ds[i][1] for i in idxs])

    torch.manual_seed(42)
    std_aug_fn = _aug_only_cfg(level, 42)
    std_aug = build_augmentation_transform(
        use_color_jitter=std_aug_fn.use_color_jitter,
        use_random_gray=std_aug_fn.use_random_gray,
        use_random_crop=std_aug_fn.use_random_crop,
        use_random_erasing=std_aug_fn.use_random_erasing,
        use_hflip=std_aug_fn.use_hflip,
        use_gaussian_noise=std_aug_fn.use_gaussian_noise,
        aug_strength=std_aug_fn.aug_strength,
        image_size=std_aug_fn.image_size,
    )
    aug_std = std_aug(imgs) if std_aug is not None else imgs.clone()

    torch.manual_seed(42)
    strong_aug = get_strong_aug(level)
    aug_strong = strong_aug(imgs)

    report = {"level": level, "n_samples": len(idxs)}

    if level == 1:
        cat_bmr_before = (imgs[lbls == 0][:, 2] - imgs[lbls == 0][:, 0]).mean().item()
        dog_bmr_before = (imgs[lbls == 1][:, 2] - imgs[lbls == 1][:, 0]).mean().item()
        gap_before = cat_bmr_before - dog_bmr_before

        cat_bmr_std = (aug_std[lbls == 0][:, 2] - aug_std[lbls == 0][:, 0]).mean().item()
        dog_bmr_std = (aug_std[lbls == 1][:, 2] - aug_std[lbls == 1][:, 0]).mean().item()
        gap_std = cat_bmr_std - dog_bmr_std

        cat_bmr_strong = (aug_strong[lbls == 0][:, 2] - aug_strong[lbls == 0][:, 0]).mean().item()
        dog_bmr_strong = (aug_strong[lbls == 1][:, 2] - aug_strong[lbls == 1][:, 0]).mean().item()
        gap_strong = cat_bmr_strong - dog_bmr_strong

        report.update({
            "metric": "mean_blue_minus_red_gap",
            "before": round(gap_before, 4),
            "after_std": round(gap_std, 4),
            "std_retention_pct": round((gap_std / gap_before) * 100, 1) if gap_before else 0.0,
            "after_strong": round(gap_strong, 4),
            "strong_retention_pct": round((gap_strong / gap_before) * 100, 1) if gap_before else 0.0,
            "details": f"Cat B-R: {cat_bmr_before:+.3f} -> {cat_bmr_std:+.3f} -> {cat_bmr_strong:+.3f} | Dog B-R: {dog_bmr_before:+.3f} -> {dog_bmr_std:+.3f} -> {dog_bmr_strong:+.3f}",
        })
    elif level == 2:
        unchanged_before = 1.000
        unchanged_std = sum(
            torch.allclose(imgs[i, :, :8, :8], aug_std[i, :, :8, :8], atol=1e-3)
            for i in range(len(idxs))
        ) / len(idxs)
        unchanged_strong = sum(
            torch.allclose(imgs[i, :, :8, :8], aug_strong[i, :, :8, :8], atol=1e-3)
            for i in range(len(idxs))
        ) / len(idxs)

        report.update({
            "metric": "fraction_top_left_8x8_unchanged",
            "before": unchanged_before,
            "after_std": round(unchanged_std, 3),
            "after_strong": round(unchanged_strong, 3),
            "details": f"Patch unchanged: 100% -> {unchanged_std*100:.1f}% (std aug) -> {unchanged_strong*100:.1f}% (strong aug p=0.85 erase)",
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
        e_std = stripe_energy(aug_std, lbls)
        e_strong = stripe_energy(aug_strong, lbls)

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
    is_oracle = condition == "oracle"
    tr_ds, va_ds, te_ds = build_datasets(level, seed=42, oracle=is_oracle)

    rows = []
    first_result = None

    for seed in seeds:
        cfg_fn = CONDITIONS[condition]
        if condition in ("baseline", "baseline_head", "ctrl_lb"):
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

    wg_vals = [r["wg_acc"] for r in rows]
    va_vals = [r["val_acc"] for r in rows]
    ta_vals = [r["test_avg"] for r in rows]

    group_summary = {}
    for g in range(4):
        accs = [r["group_accs"][str(g)] for r in rows if str(g) in r["group_accs"]]
        accs = [v for v in accs if not np.isnan(v)]
        k_tot = sum(r["group_counts"][str(g)][0] for r in rows if str(g) in r["group_counts"])
        n_tot = sum(r["group_counts"][str(g)][1] for r in rows if str(g) in r["group_counts"])
        ci_lo, ci_hi = wilson_ci(k_tot, n_tot)
        group_summary[str(g)] = {
            "mean": float(np.mean(accs)) if accs else float("nan"),
            "std": float(np.std(accs)) if accs else 0.0,
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "k_total": k_tot,
            "n_total": n_tot,
        }

    stats = {
        "level": level,
        "condition": condition,
        "seeds": seeds,
        "rows": rows,
        "summary": {
            "val_acc": {"mean": float(np.mean(va_vals)), "std": float(np.std(va_vals))},
            "test_avg": {"mean": float(np.mean(ta_vals)), "std": float(np.std(ta_vals))},
            "wg_acc": {"mean": float(np.mean(wg_vals)), "std": float(np.std(wg_vals))},
            "group_accs": group_summary,
        },
    }
    return stats, first_result


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
    conditions = ["baseline_head", "ctrl_lb", "aug_only", "loss_only", "fix", "oracle"]
    cond_labels = ["Baseline (head)", "Ctrl (last block)", "Aug Only", "Loss Only", "Fix", "Oracle"]
    colors = [_RED, "#fb923c", _YEL, _ACC, _GRN, _CYN]
    target = LEVEL_TARGETS[level]

    x = np.arange(4)
    w = 0.12
    fig, ax = plt.subplots(figsize=(11, 4))

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
            x + (i - 2.5) * w,
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
    conditions = ["baseline_head", "ctrl_lb", "aug_only", "loss_only", "fix", "oracle"]
    cond_labels = ["Baseline (head)", "Ctrl (LB)", "Aug Only", "Loss Only", "Fix", "Oracle"]
    colors = [_RED, "#fb923c", _YEL, _ACC, _GRN, _CYN]
    metrics = ["val_acc", "test_avg", "wg_acc"]
    m_labels = ["Val Acc", "Test Avg", "Worst-Group"]
    target = LEVEL_TARGETS[level]

    x = np.arange(len(metrics))
    w = 0.12
    fig, ax = plt.subplots(figsize=(9, 3.8))

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
            x + (i - 2.5) * w,
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
    print("\n" + "=" * 115)
    print(
        f"{'Level':>5} | {'Condition':>18} | {'Val Acc':>8} | {'Test Avg':>8} | "
        f"{'WG Acc':>8} | {'Group 0 (Cat no-SC)':>14} | {'Group 1 (Cat +SC)':>14} | "
        f"{'Group 2 (Dog no-SC)':>14} | {'Group 3 (Dog +SC)':>14}"
    )
    print("-" * 115)
    conditions = [
        "baseline_head",
        "ctrl_lb",
        "aug_only",
        "loss_only",
        "fix",
        "oracle",
        "strong_aug_only",
        "strong_aug_loss",
    ]
    for level in [1, 2, 3]:
        for cond in conditions:
            key = f"L{level}_{cond}"
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

            print(
                f"  {level:>3}  | {cond:>18} | {fmt(s['val_acc']):>8} | "
                f"{fmt(s['test_avg']):>8} | {fmt(s['wg_acc']):>8} | "
                f"{fmt_grp(0):>14} | {fmt_grp(1):>14} | "
                f"{fmt_grp(2):>14} | {fmt_grp(3):>14}"
            )
        print("-" * 115)
    print("=" * 115 + "\n")


def check_targets(all_stats: dict) -> list:
    warnings = []
    for level in [1, 2, 3]:
        target = LEVEL_TARGETS[level]
        fix_key = f"L{level}_fix"
        if fix_key not in all_stats:
            continue
        wg_mean = all_stats[fix_key]["summary"]["wg_acc"]["mean"]
        if wg_mean < target:
            warnings.append(
                f"WARN  Level {level}: recommended fix achieves WG={wg_mean:.3f} "
                f"< target {target:.2f}. "
                f"Consider loosening the shortcut strength or reducing the target."
            )
    return warnings


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
    args = parser.parse_args()

    all_stats = {}
    cue_audit_reports = {}
    fix_results_by_level = {}
    base_results_by_level = {}

    conditions = [
        "baseline_head",
        "ctrl_lb",
        "aug_only",
        "loss_only",
        "fix",
        "oracle",
        "strong_aug_only",
        "strong_aug_loss",
    ]

    print("\n" + "=" * 60)
    print("  STEP 4: AUGMENTATION CUE SURVIVAL AUDIT (200 SAMPLES)")
    print("=" * 60)
    for level in args.levels:
        report = audit_cue_survival(level, n_samples=200)
        cue_audit_reports[level] = report
        print(f"Level {level} Cue Audit: {report['details']}")

    for level in args.levels:
        print(f"\n{'='*60}")
        print(f"  LEVEL {level} MATCHED CONTROLS & EXPERIMENTS")
        print(f"{'='*60}")

        tr_ds, va_ds, te_ds = build_datasets(level, seed=42)

        for cond in conditions:
            print(f"\n  -- {cond} --")
            stats, res0 = run_condition(
                level, cond, args.seeds, epochs_override=args.epochs
            )
            key = f"L{level}_{cond}"
            all_stats[key] = stats
            if cond == "baseline_head":
                all_stats[f"L{level}_baseline"] = stats
                base_results_by_level[level] = res0
            elif cond == "fix":
                fix_results_by_level[level] = res0

        base_r = base_results_by_level[level]
        fix_r = fix_results_by_level[level]

        plot_group_bars(all_stats, level, DOCS_DIR / f"level{level}_groups.png")
        plot_comparison(all_stats, level, DOCS_DIR / f"level{level}_comparison.png")
        save_gradcam_examples(level, base_r, fix_r, te_ds, DOCS_DIR)

    all_stats["_cue_audit_reports"] = cue_audit_reports

    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved results to {json_path}")

    print_table(all_stats)

    warnings = check_targets(all_stats)
    if warnings:
        print("\n".join(warnings))
    else:
        print("All level targets are reachable with the documented fix.")


if __name__ == "__main__":
    main()
