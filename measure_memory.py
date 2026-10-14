from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent))


class PeakRSSMeter:
    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.peak_bytes = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = None

    def _monitor(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            time.sleep(0.01)

    def start(self):
        self._stop.clear()
        self.peak_bytes = self.proc.memory_info().rss
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        final_rss = self.proc.memory_info().rss
        if final_rss > self.peak_bytes:
            self.peak_bytes = final_rss
        return round(self.peak_bytes / (1024 * 1024), 1)


def measure():
    results = {}
    proc = psutil.Process(os.getpid())

    import streamlit
    import torch
    base_rss = round(proc.memory_info().rss / (1024 * 1024), 1)
    results["import_torch_streamlit_MB"] = base_rss
    print(f"[1] Post-import RSS: {base_rss:.1f} MB")

    from data import build_datasets, get_normalize_transform, IMAGE_SIZE
    meter = PeakRSSMeter()
    gc.collect()
    meter.start()
    tr_ds, va_ds, te_ds = build_datasets(1, seed=42)
    peak_ds = meter.stop()
    results["dataset_load_peak_MB"] = peak_ds
    print(f"[2] Dataset load peak RSS: {peak_ds:.1f} MB")

    from train import TrainConfig, run_training
    cfg = TrainConfig(
        finetune_mode="head_only",
        optimizer="adam",
        learning_rate=1e-3,
        epochs=1,
        loss_fn_name="cross_entropy",
        seed=42,
    )
    meter = PeakRSSMeter()
    gc.collect()
    meter.start()
    train_res = run_training(cfg, tr_ds, va_ds, te_ds)
    peak_train = meter.stop()
    results["training_run_peak_MB"] = peak_train
    print(f"[3] Training run peak RSS: {peak_train:.1f} MB")

    from gradcam import make_gradcam_panel
    norm = get_normalize_transform()
    raw_img = te_ds[0][0]
    img_n = norm(raw_img.unsqueeze(0))
    meter = PeakRSSMeter()
    gc.collect()
    meter.start()
    make_gradcam_panel(train_res.model, img_n)
    peak_gradcam = meter.stop()
    results["gradcam_peak_MB"] = peak_gradcam
    print(f"[4] Grad-CAM peak RSS: {peak_gradcam:.1f} MB")

    overall_peak = max(base_rss, peak_ds, peak_train, peak_gradcam)
    results["overall_peak_MB"] = overall_peak
    print(f"\nOverall peak process RSS: {overall_peak:.1f} MB (Budget: 800 MB)")

    budget = 800
    if overall_peak > budget:
        print(f"\nWARNING: Peak RSS exceeds {budget} MB budget.")
    else:
        print(f"\nOK: Peak RSS is within {budget} MB budget.")

    Path("results").mkdir(exist_ok=True)
    with open("results/memory.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/memory.json")
    return results


if __name__ == "__main__":
    measure()
