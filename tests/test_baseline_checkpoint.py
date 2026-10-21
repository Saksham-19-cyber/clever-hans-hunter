import json
import pytest
import torch
import numpy as np
from pathlib import Path

from data import build_datasets, make_dataloaders
from model import build_model
from train import TrainConfig, run_training, evaluate, _compute_group_accs
from streamlit_app import cached_baseline, compute_test_fingerprint

RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_seed42_baseline_row(level: int):
    results_path = RESULTS_DIR / f"results_L{level}.json"
    with open(results_path, "r") as f:
        data = json.load(f)
    cond = data[f"L{level}_baseline_head"]
    row_42 = next(r for r in cond["rows"] if r["seed"] == 42)
    return row_42


@pytest.mark.parametrize("level", [1, 2, 3])
def test_cached_baseline_matches_results_json(level: int):
    expected = get_seed42_baseline_row(level)
    res = cached_baseline(level)

    assert abs(res.val_acc - expected["val_acc"]) < 1e-4, (
        f"L{level} val mismatch: got {res.val_acc}, expected {expected['val_acc']}"
    )
    assert abs(res.test_avg_acc - expected["test_avg"]) < 1e-4, (
        f"L{level} test mismatch: got {res.test_avg_acc}, expected {expected['test_avg']}"
    )
    assert abs(res.worst_group_acc - expected["wg_acc"]) < 1e-4, (
        f"L{level} worst-group mismatch: got {res.worst_group_acc}, expected {expected['wg_acc']}"
    )


def test_baseline_train_save_load_roundtrip(tmp_path):
    level = 1
    expected = get_seed42_baseline_row(level)
    tr, va, te = build_datasets(level, seed=42)
    cur_fp = compute_test_fingerprint(te)

    cfg = TrainConfig(
        finetune_mode="head_only",
        optimizer="adam",
        learning_rate=1e-3,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=42,
    )
    res = run_training(cfg, tr, va, te)
    model = res.model

    # Save fc_state_dict and BatchNorm buffers
    bn_buffers = {
        k: v for k, v in model.state_dict().items()
        if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    ckpt_path = tmp_path / f"level{level}_test_baseline.pt"
    torch.save({
        "level": level,
        "test_fingerprint": cur_fp,
        "fc_state_dict": model.fc.state_dict(),
        "bn_buffers": bn_buffers,
        "history": res.history,
    }, ckpt_path)

    # Load into fresh ImageNet ResNet18
    fresh_model = build_model(finetune_mode="head_only", pretrained=True)
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_dict = {}
    for k, v in loaded_ckpt["fc_state_dict"].items():
        load_dict[k if k.startswith("fc.") else f"fc.{k}"] = v
    load_dict.update(loaded_ckpt["bn_buffers"])

    missing, unexpected = fresh_model.load_state_dict(load_dict, strict=False)
    expected_keys = [
        k for k in fresh_model.state_dict().keys()
        if k.startswith("fc.") or k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    ]
    unexpected_missing = [k for k in expected_keys if k in missing]

    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    assert len(unexpected_missing) == 0, f"Missing expected keys: {unexpected_missing}"

    # Evaluate
    fresh_model.eval()
    _, val_loader, test_loader = make_dataloaders(tr, va, te, batch_size=64)
    val_acc, *_ = evaluate(fresh_model, val_loader, "cpu")
    test_acc, test_preds, test_labels, test_groups, _ = evaluate(fresh_model, test_loader, "cpu")

    group_accs = _compute_group_accs(test_preds, test_labels, test_groups)
    valid_accs = [v for v in group_accs.values() if not np.isnan(v)]
    worst_group = min(valid_accs) if valid_accs else 0.0

    assert abs(val_acc - expected["val_acc"]) < 1e-4, f"Val acc {val_acc} != {expected['val_acc']}"
    assert abs(test_acc - expected["test_avg"]) < 1e-4, f"Test acc {test_acc} != {expected['test_avg']}"
    assert abs(worst_group - expected["wg_acc"]) < 1e-4, f"WG acc {worst_group} != {expected['wg_acc']}"
