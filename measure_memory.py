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
        self.peak_bytes = max(self.peak_bytes, self.proc.memory_info().rss)
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def get_peak_mb(self) -> float:
        current = self.proc.memory_info().rss
        if current > self.peak_bytes:
            self.peak_bytes = current
        return round(self.peak_bytes / (1024 * 1024), 1)

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.get_peak_mb()


def measure():
    results = {}
    proc = psutil.Process(os.getpid())

    # 1. Post-import RSS
    import streamlit
    import torch
    import numpy as np

    base_rss = round(proc.memory_info().rss / (1024 * 1024), 1)
    results["import_torch_streamlit_MB"] = base_rss
    print(f"[1] Post-import RSS: {base_rss:.1f} MB")

    from streamlit_app import cached_datasets, cached_baseline
    from train import TrainConfig, run_training
    from gradcam import make_gradcam_panel
    from data import get_normalize_transform

    meter = PeakRSSMeter()
    meter.start()

    # 2. Visit Level 1: datasets and baseline
    tr1, va1, te1 = cached_datasets(1)
    base1 = cached_baseline(1)
    peak_l1 = meter.get_peak_mb()
    results["visit_level_1_peak_MB"] = peak_l1
    print(f"[2] Level 1 loaded (datasets + baseline): peak RSS = {peak_l1:.1f} MB")

    # 3. Visit Level 2: datasets and baseline
    tr2, va2, te2 = cached_datasets(2)
    base2 = cached_baseline(2)
    peak_l2 = meter.get_peak_mb()
    results["visit_level_2_peak_MB"] = peak_l2
    print(f"[3] Level 2 loaded (datasets + baseline): peak RSS = {peak_l2:.1f} MB")

    # 4. Visit Level 3: datasets and baseline
    tr3, va3, te3 = cached_datasets(3)
    base3 = cached_baseline(3)
    peak_l3 = meter.get_peak_mb()
    results["visit_level_3_peak_MB"] = peak_l3
    print(f"[4] Level 3 loaded (datasets + baseline): peak RSS = {peak_l3:.1f} MB")

    # 5. One last_block 2-epoch training run
    print("[5] Running one last_block 2-epoch training run...")
    cfg_last = TrainConfig(
        finetune_mode="last_block",
        optimizer="adam",
        learning_rate=5e-4,
        epochs=2,
        loss_fn_name="cross_entropy",
        seed=42,
    )
    res_last = run_training(cfg_last, tr3, va3, te3)
    peak_train = meter.get_peak_mb()
    results["last_block_2ep_peak_MB"] = peak_train
    print(f"    last_block 2-epoch training finished: peak RSS = {peak_train:.1f} MB")

    # 6. One Grad-CAM panel computation
    print("[6] Running one Grad-CAM panel generation...")
    norm = get_normalize_transform()
    raw_img = te3[0][0]
    if raw_img.dtype == torch.uint8:
        raw_img = raw_img.float() / 255.0
    img_n = norm(raw_img.unsqueeze(0))
    make_gradcam_panel(res_last.model, img_n)
    peak_gradcam = meter.get_peak_mb()
    results["gradcam_peak_MB"] = peak_gradcam
    print(f"    Grad-CAM finished: peak RSS = {peak_gradcam:.1f} MB")

    # 7. Memory held by one trained TrainResult (model + history)
    param_bytes = sum(p.nelement() * p.element_size() for p in res_last.model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in res_last.model.buffers())
    model_bytes = param_bytes + buffer_bytes
    history_bytes = sys.getsizeof(res_last.history) + sum(sys.getsizeof(h) for h in res_last.history)
    cm_bytes = res_last.confusion_matrix.nbytes
    total_held_bytes = model_bytes + history_bytes + cm_bytes
    total_held_mb = round(total_held_bytes / (1024 * 1024), 2)
    results["train_result_held_memory_MB"] = total_held_mb
    print(f"[7] Memory held by one TrainResult (model + history): {total_held_mb:.2f} MB ({total_held_bytes} bytes)")

    overall_peak = meter.stop()
    results["overall_peak_MB"] = overall_peak
    print(f"\nOverall single-process peak RSS: {overall_peak:.1f} MB (Budget: 800 MB)")

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
