"""
run_extra_rows.py - Runs head_only + group_reweighted_ce across 3 levels, seeds [42, 1, 2], and epochs [2, 4],
timed with torch.set_num_threads(2).
"""
import json
import time
from pathlib import Path
import torch
from data import build_datasets, LEVEL_TARGETS
from train import TrainConfig, run_training

torch.set_num_threads(2)

SEEDS = [42, 1, 2]
EPOCHS_LIST = [2, 4]
LEVELS = [1, 2, 3]

results = []
out_file = Path("results/extra_rows_timed.json")
out_file.parent.mkdir(exist_ok=True)

print("Starting extra rows runs with torch.set_num_threads(2)...", flush=True)

for level in LEVELS:
    target = LEVEL_TARGETS[level]
    print(f"\n--- Loading datasets for Level {level} (Target WG: {target:.0%}) ---", flush=True)
    tr_ds, va_ds, te_ds = build_datasets(level, seed=42)
    
    for epochs in EPOCHS_LIST:
        for seed in SEEDS:
            cfg = TrainConfig(
                finetune_mode="head_only",
                optimizer="adam",
                learning_rate=1e-3,
                epochs=epochs,
                loss_fn_name="group_reweighted_ce",
                seed=seed,
            )
            t0 = time.time()
            res = run_training(cfg, tr_ds, va_ds, te_ds)
            elapsed = time.time() - t0
            
            cleared = res.worst_group_acc >= target
            entry = {
                "level": level,
                "seed": seed,
                "epochs": epochs,
                "finetune_mode": "head_only",
                "loss": "group_reweighted_ce",
                "val_acc": round(res.val_acc, 4),
                "test_avg_acc": round(res.test_avg_acc, 4),
                "worst_group_acc": round(res.worst_group_acc, 4),
                "target": target,
                "cleared": cleared,
                "elapsed_seconds": round(elapsed, 2),
                "group_accs": {str(g): round(acc, 4) for g, acc in res.group_accs.items()},
            }
            results.append(entry)
            print(
                f"[L{level} | ep={epochs} | seed={seed}] WG={res.worst_group_acc:.1%} "
                f"(target {target:.0%}, {'CLEARED' if cleared else 'FAILED'}) | "
                f"val={res.val_acc:.1%} test_avg={res.test_avg_acc:.1%} | {elapsed:.2f}s",
                flush=True
            )
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

print("\nAll extra rows completed successfully!", flush=True)
