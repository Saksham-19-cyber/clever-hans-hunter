"""
benchmark_timing.py
Measures median and range of seconds per run for:
- head_only (2 epochs)
- head_only (4 epochs)
- last_block (2 epochs)
- full (1 epoch)
with torch.set_num_threads(2), 3 runs each, nothing else running.
"""
import time
import json
import numpy as np
import torch
from pathlib import Path
from data import build_datasets
from train import TrainConfig, run_training

torch.set_num_threads(2)

RUNS = [
    ("head_only_2ep", "head_only", 2, 1e-3),
    ("head_only_4ep", "head_only", 4, 1e-3),
    ("last_block_2ep", "last_block", 2, 5e-4),
    ("full_1ep", "full", 1, 1e-4),
]

SEEDS = [42, 1, 2]

def main():
    print("Loading Level 1 datasets for timing benchmark...")
    tr_ds, va_ds, te_ds = build_datasets(1, seed=42)
    results = {}

    for name, mode, epochs, lr in RUNS:
        times = []
        val_accs = []
        wg_accs = []
        print(f"\n--- Timing {name}: mode={mode}, epochs={epochs}, lr={lr} (3 runs) ---")
        for seed in SEEDS:
            cfg = TrainConfig(
                finetune_mode=mode,
                optimizer="adam",
                learning_rate=lr,
                epochs=epochs,
                loss_fn_name="cross_entropy",
                seed=seed,
            )
            t0 = time.perf_counter()
            res = run_training(cfg, tr_ds, va_ds, te_ds)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            val_accs.append(res.val_acc)
            wg_accs.append(res.worst_group_acc)
            print(f"  [seed {seed}] elapsed: {elapsed:.2f}s | val={res.val_acc:.3f} | wg={res.worst_group_acc:.3f}")

        median_t = float(np.median(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
        results[name] = {
            "mode": mode,
            "epochs": epochs,
            "times": [round(t, 2) for t in times],
            "median_seconds": round(median_t, 2),
            "range_seconds": [round(min_t, 2), round(max_t, 2)],
            "val_accs": [round(v, 4) for v in val_accs],
            "wg_accs": [round(w, 4) for w in wg_accs],
        }
        print(f"==> {name}: median={median_t:.2f}s, range=[{min_t:.2f}s, {max_t:.2f}s]")

    out_file = Path("results/timing_benchmark_measured.json")
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved timing results to {out_file}")

if __name__ == "__main__":
    main()
