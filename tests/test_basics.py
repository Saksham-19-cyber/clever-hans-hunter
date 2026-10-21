"""
tests/test_basics.py  –  Automated sanity checks for Clever Hans Hunter.

Run with:
    python -m pytest tests/test_basics.py -v
or:
    python tests/test_basics.py
"""

import sys 
import os 
sys .path .insert (0 ,os .path .join (os .path .dirname (__file__ ),".."))

import pytest 
import torch 
import numpy as np 

from data import (
build_datasets ,apply_shortcut ,make_group_id ,
shortcut_correlation ,SHORTCUT_CORR_TRAIN ,SHORTCUT_CORR_TEST ,
)
from model import build_model ,FINETUNE_MODES 
from gradcam import GradCAM ,make_gradcam_panel 
from data import get_normalize_transform ,normalize_tensor 


@pytest .mark .parametrize ("level",[1 ,2 ,3 ])
def test_shortcut_correlation_train (level ):
    """Shortcut correlation in train/val should be close to 95 %."""
    tr ,va ,te =build_datasets (level ,seed =42 )
    corr =shortcut_correlation (tr )
    assert abs (corr -SHORTCUT_CORR_TRAIN )<0.05 ,(
    f"Level {level }: train shortcut-corr={corr :.3f}, expected ~{SHORTCUT_CORR_TRAIN }"
    )


@pytest .mark .parametrize ("level",[1 ,2 ,3 ])
def test_shortcut_correlation_test (level ):
    """Shortcut correlation in test should be close to 50 %."""
    tr ,va ,te =build_datasets (level ,seed =42 )
    corr =shortcut_correlation (te )
    assert abs (corr -SHORTCUT_CORR_TEST )<0.07 ,(
    f"Level {level }: test shortcut-corr={corr :.3f}, expected ~{SHORTCUT_CORR_TEST }"
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_test_fingerprint_matches_pinned(level):
    """Test set pixel and metadata fingerprint must match data/fingerprints.json exactly."""
    import json
    from pathlib import Path
    from streamlit_app import compute_test_fingerprint
    fp_path = Path(__file__).parent.parent / "data" / "fingerprints.json"
    with open(fp_path, "r") as f:
        pinned = json.load(f)
    tr, va, te = build_datasets(level, seed=42)
    cur_fp = compute_test_fingerprint(te)
    assert cur_fp == pinned[str(level)], (
        f"Level {level} fingerprint mismatch: got {cur_fp}, expected {pinned[str(level)]}"
    )


def test_group_id_values ():
    """All 4 group ids are present in the training set."""
    tr ,_ ,_ =build_datasets (1 ,seed =42 )
    unique_groups =set (tr .group_ids )
    assert unique_groups =={0 ,1 ,2 ,3 },f"Expected groups {{0,1,2,3}}, got {unique_groups }"


def test_group_id_formula ():
    """Group id formula: 2*cls + shortcut_present."""
    for cls in [0 ,1 ]:
        for sp in [True ,False ]:
            gid =make_group_id (cls ,sp )
            assert gid ==2 *cls +int (sp ),f"Wrong gid for cls={cls } sp={sp }: {gid }"


@pytest .mark .parametrize ("level",[1 ,2 ,3 ])
@pytest .mark .parametrize ("cls",[0 ,1 ])
def test_apply_shortcut_shape_range (level ,cls ):
    """apply_shortcut must return a tensor in [0,1] with unchanged spatial dims."""
    img =torch .rand (3 ,64 ,64 )
    out ,sp =apply_shortcut (img ,cls ,level ,correlated =True )
    assert out .shape ==img .shape ,f"Shape mismatch: {out .shape } vs {img .shape }"
    assert out .min ()>=0.0 and out .max ()<=1.0 ,(
    f"Value out of range: min={out .min ():.3f} max={out .max ():.3f}"
    )
    assert isinstance (sp ,bool ),f"shortcut_present should be bool, got {type (sp )}"


@pytest .mark .parametrize ("mode",FINETUNE_MODES )
def test_model_output_shape (mode ):
    """Model should output (batch, 2) logits."""
    m =build_model (num_classes =2 ,finetune_mode =mode ,pretrained =False )
    x =torch .randn (4 ,3 ,64 ,64 )
    with torch .no_grad ():
        out =m (x )
    assert out .shape ==(4 ,2 ),f"Expected (4,2) logits, got {out .shape }"


@pytest .mark .parametrize ("mode",FINETUNE_MODES )
def test_finetune_modes_freeze (mode ):
    """head_only should have far fewer trainable params than full."""
    from model import count_trainable 
    m =build_model (finetune_mode =mode ,pretrained =False )
    n =count_trainable (m )
    if mode =="head_only":
        assert n <10_000 ,f"head_only has too many trainable params: {n }"
    elif mode =="full":
        assert n >1_000_000 ,f"full mode has too few trainable params: {n }"


def test_gradcam_output_shape ():
    """Grad-CAM heatmap should be 2D with the same H×W as the input."""
    model =build_model (pretrained =False ,finetune_mode ="full")
    model .eval ()
    normalize =get_normalize_transform ()

    img_raw =torch .rand (3 ,64 ,64 )
    img_norm =normalize (img_raw .unsqueeze (0 ))

    gcam =GradCAM (model )
    try :
        cam ,overlay_pil ,pred_cls ,conf =gcam .explain (img_norm )
    finally :
        gcam .remove_hooks ()


    assert cam .ndim ==2 ,f"Expected 2D heatmap, got shape {cam .shape }"
    assert cam .min ()>=0.0 and cam .max ()<=1.0 +1e-6 ,(
    f"CAM out of range: min={cam .min ():.3f} max={cam .max ():.3f}"
    )
    assert 0 <=pred_cls <=1 ,f"pred_cls={pred_cls } out of range"
    assert 0.0 <=conf <=1.0 ,f"confidence={conf } out of range"


def test_gradcam_with_uint8_raw_input ():
    model =build_model (pretrained =False ,finetune_mode ="full")
    model .eval ()
    tr ,va ,te =build_datasets (1 ,seed =42 )
    raw_img ,*_ =te [0 ]
    assert raw_img .dtype ==torch .uint8 
    img_norm =normalize_tensor (raw_img .unsqueeze (0 ))
    orig_pil ,heat_pil ,ov_pil ,pred_cls ,conf =make_gradcam_panel (model ,img_norm ,raw_img =raw_img )
    assert orig_pil .size ==(64 ,64 )
    assert ov_pil .size ==(64 ,64 )
    assert 0 <=pred_cls <=1 

    orig2 ,heat2 ,ov2 ,pred2 ,conf2 =make_gradcam_panel (model ,raw_img )
    assert orig2 .size ==(64 ,64 )


def test_dataset_sizes ():
    """Train/val/test should have the expected number of samples."""
    from data import N_TRAIN ,N_VAL ,N_TEST 
    tr ,va ,te =build_datasets (1 ,seed =42 )
    assert len (tr )==2 *N_TRAIN ,f"Train size {len (tr )} != {2 *N_TRAIN }"
    assert len (va )==2 *N_VAL ,f"Val size {len (va )} != {2 *N_VAL }"
    assert len (te )==2 *N_TEST ,f"Test size {len (te )} != {2 *N_TEST }"


def test_dataloader_shapes ():
    from data import make_dataloaders ,IMAGE_SIZE 
    tr ,va ,te =build_datasets (1 ,seed =42 )
    tr_dl ,va_dl ,te_dl =make_dataloaders (tr ,va ,te ,batch_size =16 )
    imgs ,labels ,groups =next (iter (tr_dl ))
    assert imgs .shape ==(16 ,3 ,IMAGE_SIZE ,IMAGE_SIZE ),f"Image shape {imgs .shape }"
    assert labels .shape ==(16 ,),f"Labels shape {labels .shape }"
    assert groups .shape ==(16 ,),f"Groups shape {groups .shape }"


if __name__ =="__main__":

    import traceback 

    tests =[
    lambda :[test_shortcut_correlation_train (l )for l in [1 ,2 ,3 ]],
    lambda :[test_shortcut_correlation_test (l )for l in [1 ,2 ,3 ]],
    test_group_id_values ,
    test_group_id_formula ,
    lambda :[test_apply_shortcut_shape_range (l ,c )for l in [1 ,2 ,3 ]for c in [0 ,1 ]],
    lambda :[test_model_output_shape (m )for m in FINETUNE_MODES ],
    lambda :[test_finetune_modes_freeze (m )for m in FINETUNE_MODES ],
    test_gradcam_output_shape ,
    test_dataset_sizes ,
    test_dataloader_shapes ,
    ]

    names =[
    "Shortcut corr (train)",
    "Shortcut corr (test)",
    "Group ids present",
    "Group id formula",
    "apply_shortcut shape/range",
    "Model output shape",
    "Fine-tune freeze modes",
    "Grad-CAM shape",
    "Dataset sizes",
    "DataLoader shapes",
    ]

    passed =failed =0 
    for name ,fn in zip (names ,tests ):
        try :
            fn ()
            print (f"  [PASS] {name }")
            passed +=1 
        except Exception as e :
            print (f"  [FAIL] {name } -> {e }")
            traceback .print_exc ()
            failed +=1 

    print (f"\n{passed } passed / {failed } failed")
    sys .exit (0 if failed ==0 else 1 )

