"""
measure_memory.py  –  Peak-RAM measurement for Clever Hans Hunter.

Measures:
  1. Peak RAM after loading Level 1 datasets
  2. Peak RAM after one training run (head_only, 1 epoch)
  3. Peak RAM after one Grad-CAM call

Run:
    python measure_memory.py

Reports numbers in MB.  Streamlit Cloud limit is ~1 GB.
"""
from __future__ import annotations 

import gc 
import sys 
import tracemalloc 
from pathlib import Path 

sys .path .insert (0 ,str (Path (__file__ ).parent ))


def mb (n_bytes :int )->float :
    return n_bytes /1024 /1024 


def measure ():
    results ={}


    gc .collect ()
    tracemalloc .start ()
    from data import build_datasets ,get_normalize_transform ,IMAGE_SIZE 
    tr_ds ,va_ds ,te_ds =build_datasets (1 ,seed =42 )
    _ ,peak1 =tracemalloc .get_traced_memory ()
    tracemalloc .stop ()
    results ["dataset_load_MB"]=round (mb (peak1 ),1 )
    print (f"[1] Dataset load peak: {mb (peak1 ):.1f} MB")


    gc .collect ()
    tracemalloc .start ()
    from train import TrainConfig ,run_training 
    cfg =TrainConfig (
    finetune_mode ="head_only",optimizer ="adam",
    learning_rate =1e-3 ,epochs =1 ,
    loss_fn_name ="cross_entropy",seed =42 ,
    )
    result =run_training (cfg ,tr_ds ,va_ds ,te_ds )
    _ ,peak2 =tracemalloc .get_traced_memory ()
    tracemalloc .stop ()
    results ["training_run_MB"]=round (mb (peak2 ),1 )
    print (f"[2] Training run peak: {mb (peak2 ):.1f} MB")


    gc .collect ()
    tracemalloc .start ()
    from gradcam import make_gradcam_panel 
    norm =get_normalize_transform ()
    raw_img =te_ds [0 ][0 ]
    img_n =norm (raw_img .unsqueeze (0 ))
    make_gradcam_panel (result .model ,img_n )
    _ ,peak3 =tracemalloc .get_traced_memory ()
    tracemalloc .stop ()
    results ["gradcam_MB"]=round (mb (peak3 ),1 )
    print (f"[3] Grad-CAM peak:     {mb (peak3 ):.1f} MB")

    total_peak =max (peak1 ,peak2 ,peak3 )
    results ["overall_peak_MB"]=round (mb (total_peak ),1 )
    print (f"\nOverall peak: {mb (total_peak ):.1f} MB  (Streamlit Cloud budget: ~1000 MB)")

    budget =800 
    if mb (total_peak )>budget :
        print (f"\nWARNING: Peak exceeds {budget } MB budget.")
        print ("Recommended reduction: lower N_TRAIN/N_VAL/N_TEST in data.py or use IMAGE_SIZE=48.")
    else :
        print (f"\nOK: Well within {budget } MB budget.")

    import json 
    Path ("results").mkdir (exist_ok =True )
    with open ("results/memory.json","w")as f :
        json .dump (results ,f ,indent =2 )
    print ("Saved results/memory.json")
    return results 


if __name__ =="__main__":
    measure ()
