"""
results.py  –  Reproducible measurement script for Clever Hans Hunter.

For each level (1,2,3) and each condition (baseline, recommended fix,
augmentation-only, group-loss-only) it runs 3 seeds and reports:
  - val accuracy
  - test average accuracy
  - per-group accuracies (4 groups)
  - worst-group accuracy (mean ± std over seeds)

Outputs
  results/results.json          raw numbers
  docs/level{N}_groups.png      per-group bar charts
  docs/level{N}_comparison.png  baseline vs fix comparison
  docs/level{N}_gradcam_*.png   before/after Grad-CAM for one sample

Usage
  python results.py [--levels 1 2 3] [--seeds 42 1 2] [--epochs 3]
"""

from __future__ import annotations 

import argparse 
import json 
import os 
import sys 
import time 
from pathlib import Path 

if hasattr (sys .stdout ,"reconfigure"):
    try :
        sys .stdout .reconfigure (encoding ="utf-8")
    except Exception :
        pass 

import matplotlib 

matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
import numpy as np 
import torch 

sys .path .insert (0 ,str (Path (__file__ ).parent ))

from data import build_datasets ,get_normalize_transform ,CLASS_NAMES 
from gradcam import make_gradcam_panel ,_tensor_to_pil 
from train import TrainConfig ,run_training ,train_baseline 

RESULTS_DIR =Path ("results")
DOCS_DIR =Path ("docs")
RESULTS_DIR .mkdir (exist_ok =True )
DOCS_DIR .mkdir (exist_ok =True )

LEVEL_TARGETS ={1 :0.55 ,2 :0.60 ,3 :0.52 }

GROUP_LABELS =["Cat\n(no SC)","Cat\n(+SC)","Dog\n(no SC)","Dog\n(+SC)"]

_BG ="#0f1117";_CARD ="#1a1d27";_ACC ="#6c63ff"
_GRN ="#4ade80";_YEL ="#facc15";_RED ="#f87171"
_TXT ="#e2e8f0";_BRD ="#2a2d3e"


def _baseline_cfg (seed :int )->TrainConfig :
    return TrainConfig (
    finetune_mode ="head_only",optimizer ="adam",
    learning_rate =1e-3 ,epochs =3 ,
    loss_fn_name ="cross_entropy",seed =seed ,
    )


def _fix_cfg (level :int ,seed :int )->TrainConfig :
    """README recommended fix per level."""
    base =dict (finetune_mode ="last_block",optimizer ="adam",
    learning_rate =5e-4 ,epochs =3 ,
    loss_fn_name ="group_reweighted_ce",seed =seed )
    if level ==1 :
        return TrainConfig (**base ,
        use_color_jitter =True ,use_random_gray =True ,
        aug_strength =0.6 )
    elif level ==2 :
        return TrainConfig (**base ,
        use_random_erasing =True ,use_random_crop =True ,
        aug_strength =0.6 )
    else :
        return TrainConfig (**base ,
        use_random_gray =True ,use_gaussian_noise =True ,
        aug_strength =0.6 )


def _aug_only_cfg (level :int ,seed :int )->TrainConfig :
    """Augmentation only, plain CE (ablation)."""
    base =dict (finetune_mode ="last_block",optimizer ="adam",
    learning_rate =5e-4 ,epochs =3 ,
    loss_fn_name ="cross_entropy",seed =seed )
    if level ==1 :
        return TrainConfig (**base ,
        use_color_jitter =True ,use_random_gray =True ,
        aug_strength =0.6 )
    elif level ==2 :
        return TrainConfig (**base ,
        use_random_erasing =True ,use_random_crop =True ,
        aug_strength =0.6 )
    else :
        return TrainConfig (**base ,
        use_random_gray =True ,use_gaussian_noise =True ,
        aug_strength =0.6 )


def _loss_only_cfg (level :int ,seed :int )->TrainConfig :
    """Group-reweighted loss only, no augmentation (ablation)."""
    return TrainConfig (
    finetune_mode ="last_block",optimizer ="adam",
    learning_rate =5e-4 ,epochs =3 ,
    loss_fn_name ="group_reweighted_ce",seed =seed ,
    )


CONDITIONS ={
"baseline":_baseline_cfg ,
"fix":_fix_cfg ,
"aug_only":_aug_only_cfg ,
"loss_only":_loss_only_cfg ,
}


def run_condition (level :int ,condition :str ,seeds :list ,
epochs_override :int |None =None )->tuple [dict ,object ]:
    """Run condition over all seeds and return aggregated stats dict and first result."""
    tr_ds ,va_ds ,te_ds =build_datasets (level ,seed =42 )

    rows =[]
    first_result =None 
    for seed in seeds :
        if condition in ("fix","aug_only","loss_only"):
            cfg_fn =CONDITIONS [condition ]
            cfg =cfg_fn (level ,seed )if condition !="baseline"else cfg_fn (seed )
        else :
            cfg =CONDITIONS [condition ](seed )

        if epochs_override is not None :
            cfg .epochs =epochs_override 

        t0 =time .time ()
        result =run_training (cfg ,tr_ds ,va_ds ,te_ds )
        if first_result is None :
            first_result =result 
        elapsed =time .time ()-t0 

        row ={
        "seed":seed ,
        "val_acc":result .val_acc ,
        "test_avg":result .test_avg_acc ,
        "wg_acc":result .worst_group_acc ,
        "group_accs":{str (g ):v for g ,v in result .group_accs .items ()},
        "elapsed_s":round (elapsed ,1 ),
        }
        rows .append (row )
        print (f"  [{condition } seed={seed }] wg={result .worst_group_acc :.3f} "
        f"val={result .val_acc :.3f} elapsed={elapsed :.0f}s")


    wg_vals =[r ["wg_acc"]for r in rows ]
    va_vals =[r ["val_acc"]for r in rows ]
    ta_vals =[r ["test_avg"]for r in rows ]

    group_means ={}
    for g in range (4 ):
        vals =[r ["group_accs"][str (g )]for r in rows if str (g )in r ["group_accs"]]
        vals =[v for v in vals if not (v !=v )]
        group_means [str (g )]={"mean":float (np .mean (vals )),"std":float (np .std (vals ))}if vals else {"mean":float ("nan"),"std":0.0 }

    stats ={
    "level":level ,
    "condition":condition ,
    "seeds":seeds ,
    "rows":rows ,
    "summary":{
    "val_acc":{"mean":float (np .mean (va_vals )),"std":float (np .std (va_vals ))},
    "test_avg":{"mean":float (np .mean (ta_vals )),"std":float (np .std (ta_vals ))},
    "wg_acc":{"mean":float (np .mean (wg_vals )),"std":float (np .std (wg_vals ))},
    "group_accs":group_means ,
    },
    }
    return stats ,first_result 


def _style (fig ):
    fig .patch .set_facecolor (_BG )
    for ax in fig .axes :
        ax .set_facecolor (_CARD )
        ax .tick_params (colors =_TXT )
        ax .xaxis .label .set_color (_TXT )
        ax .yaxis .label .set_color (_TXT )
        ax .title .set_color (_TXT )
        for sp in ax .spines .values ():
            sp .set_edgecolor (_BRD )
    return fig 


def plot_group_bars (stats :dict ,level :int ,path :Path ):
    """Per-group accuracy bar chart with error bars for the 4 conditions."""
    conditions =["baseline","fix","aug_only","loss_only"]
    cond_labels =["Baseline","Recommended Fix","Aug Only","Loss Only"]
    colors =[_RED ,_GRN ,_YEL ,_ACC ]
    target =LEVEL_TARGETS [level ]

    x =np .arange (4 )
    w =0.18 
    fig ,ax =plt .subplots (figsize =(10 ,4 ))

    for i ,(cond ,label ,color )in enumerate (zip (conditions ,cond_labels ,colors )):
        key =f"L{level }_{cond }"
        if key not in stats :
            continue 
        ga =stats [key ]["summary"]["group_accs"]
        means =[ga [str (g )]["mean"]for g in range (4 )]
        stds =[ga [str (g )]["std"]for g in range (4 )]
        ax .bar (x +(i -1.5 )*w ,means ,w ,label =label ,color =color ,
        alpha =0.85 ,yerr =stds ,capsize =3 ,zorder =2 )

    ax .axhline (target ,color ="white",linestyle ="--",linewidth =1.2 ,
    label =f"Target {target :.0%}",zorder =3 )
    ax .set_xticks (x );ax .set_xticklabels (GROUP_LABELS )
    ax .set_ylim (0 ,1.1 );ax .set_ylabel ("Accuracy");ax .set_title (f"Level {level } – Per-Group Accuracy")
    ax .legend (facecolor =_CARD ,edgecolor =_BRD ,labelcolor =_TXT ,fontsize =8 )
    ax .grid (axis ="y",color =_BRD ,linestyle ="--",zorder =1 )
    _style (fig );fig .tight_layout ()
    fig .savefig (path ,dpi =130 ,bbox_inches ="tight")
    plt .close (fig )
    print (f"  Saved {path }")


def plot_comparison (stats :dict ,level :int ,path :Path ):
    """Bar chart comparing val acc / test avg / wg acc across conditions."""
    conditions =["baseline","fix","aug_only","loss_only"]
    cond_labels =["Baseline","Fix","Aug Only","Loss Only"]
    colors =[_RED ,_GRN ,_YEL ,_ACC ]
    metrics =["val_acc","test_avg","wg_acc"]
    m_labels =["Val Acc","Test Avg","Worst-Group"]
    target =LEVEL_TARGETS [level ]

    x =np .arange (len (metrics ))
    w =0.18 
    fig ,ax =plt .subplots (figsize =(8 ,3.5 ))

    for i ,(cond ,label ,color )in enumerate (zip (conditions ,cond_labels ,colors )):
        key =f"L{level }_{cond }"
        if key not in stats :
            continue 
        s =stats [key ]["summary"]
        means =[s [m ]["mean"]for m in metrics ]
        stds =[s [m ]["std"]for m in metrics ]
        ax .bar (x +(i -1.5 )*w ,means ,w ,label =label ,color =color ,
        alpha =0.85 ,yerr =stds ,capsize =3 ,zorder =2 )

    ax .axhline (target ,color ="white",linestyle ="--",linewidth =1.2 ,
    label =f"WG target {target :.0%}")
    ax .set_xticks (x );ax .set_xticklabels (m_labels )
    ax .set_ylim (0 ,1.12 );ax .set_ylabel ("Accuracy")
    ax .set_title (f"Level {level } – Baseline vs Fixes")
    ax .legend (facecolor =_CARD ,edgecolor =_BRD ,labelcolor =_TXT ,fontsize =8 )
    ax .grid (axis ="y",color =_BRD ,linestyle ="--",zorder =1 )
    _style (fig );fig .tight_layout ()
    fig .savefig (path ,dpi =130 ,bbox_inches ="tight")
    plt .close (fig )
    print (f"  Saved {path }")


def save_gradcam_examples (level :int ,before_result ,after_result ,
test_ds ,docs_dir :Path ):
    """Save Grad-CAM panels for one test sample (before / after fix)."""
    norm =get_normalize_transform ()

    if before_result .suspicious :
        sample_idx =before_result .suspicious [0 ][0 ]
    else :
        sample_idx =0 

    if sample_idx >=len (test_ds ):
        sample_idx =0 

    raw_img ,true_lbl ,group_id =test_ds [sample_idx ]
    img_norm =norm (raw_img .unsqueeze (0 ))

    for tag ,result in [("before",before_result ),("after",after_result )]:
        if result is None or result .model is None :
            continue 
        try :
            orig ,heat ,overlay ,pred_cls ,conf =make_gradcam_panel (
            result .model ,img_norm 
            )
            fig ,axes =plt .subplots (1 ,3 ,figsize =(9 ,3 ))
            for ax ,img ,title in zip (
            axes ,
            [orig ,heat ,overlay ],
            ["Original","Heatmap","Overlay"],
            ):
                ax .imshow (img )
                ax .set_title (title ,color =_TXT )
                ax .axis ("off")
            correct ="Correct"if pred_cls ==true_lbl else "Wrong"
            fig .suptitle (
            f"Level {level } {tag .capitalize ()} | "
            f"True:{CLASS_NAMES [true_lbl ]} "
            f"Pred:{CLASS_NAMES [pred_cls ]} ({conf :.0%}) {correct }",
            color =_TXT ,fontsize =9 ,
            )
            fig .patch .set_facecolor (_BG )
            path =docs_dir /f"level{level }_gradcam_{tag }.png"
            fig .savefig (path ,dpi =130 ,bbox_inches ="tight")
            plt .close (fig )
            print (f"  Saved {path }")
        except Exception as e :
            print (f"  Grad-CAM {tag } error: {e }")


def print_table (all_stats :dict ):
    print ("\n"+"="*88 )
    print (f"{'Level':>5} | {'Condition':>14} | {'Val':>6} | {'TestAvg':>7} | "
    f"{'WG':>6} | Group0 | Group1 | Group2 | Group3")
    print ("-"*88 )
    for level in [1 ,2 ,3 ]:
        for cond in ["baseline","fix","aug_only","loss_only"]:
            key =f"L{level }_{cond }"
            if key not in all_stats :
                continue 
            s =all_stats [key ]["summary"]
            ga =s ["group_accs"]

            def fmt (m ):return f"{m ['mean']:.2f}±{m ['std']:.2f}"
            def fmtg (g ):return f"{ga [str (g )]['mean']:.2f}"if str (g )in ga else "  n/a"

            print (f"  {level :>3}  | {cond :>14} | {fmt (s ['val_acc']):>8} | "
            f"{fmt (s ['test_avg']):>9} | {fmt (s ['wg_acc']):>8} | "
            f"{fmtg (0 ):>6} | {fmtg (1 ):>6} | {fmtg (2 ):>6} | {fmtg (3 ):>6}")
    print ("="*88 +"\n")


def check_targets (all_stats :dict )->list :
    warnings =[]
    for level in [1 ,2 ,3 ]:
        target =LEVEL_TARGETS [level ]
        fix_key =f"L{level }_fix"
        if fix_key not in all_stats :
            continue 
        wg_mean =all_stats [fix_key ]["summary"]["wg_acc"]["mean"]
        if wg_mean <target :
            warnings .append (
            f"WARN  Level {level }: recommended fix achieves WG={wg_mean :.3f} "
            f"< target {target :.2f}. "
            f"Consider loosening the shortcut strength or reducing the target."
            )
    return warnings 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--levels",nargs ="+",type =int ,default =[1 ,2 ,3 ])
    parser .add_argument ("--seeds",nargs ="+",type =int ,default =[42 ,1 ,2 ])
    parser .add_argument ("--epochs",type =int ,default =None ,
    help ="Override epochs for all conditions (useful for quick tests)")
    args =parser .parse_args ()

    all_stats ={}
    fix_results_by_level ={}
    base_results_by_level ={}

    for level in args .levels :
        print (f"\n{'='*60 }")
        print (f"  LEVEL {level }")
        print (f"{'='*60 }")


        tr_ds ,va_ds ,te_ds =build_datasets (level ,seed =42 )

        for cond in ["baseline","fix","aug_only","loss_only"]:
            print (f"\n  -- {cond } --")
            stats ,res0 =run_condition (level ,cond ,args .seeds ,
            epochs_override =args .epochs )
            key =f"L{level }_{cond }"
            all_stats [key ]=stats 
            if cond =="baseline":
                base_results_by_level [level ]=res0 
            elif cond =="fix":
                fix_results_by_level [level ]=res0 

        base_r =base_results_by_level [level ]
        fix_r =fix_results_by_level [level ]


        plot_group_bars (all_stats ,level ,DOCS_DIR /f"level{level }_groups.png")
        plot_comparison (all_stats ,level ,DOCS_DIR /f"level{level }_comparison.png")
        save_gradcam_examples (level ,base_r ,fix_r ,te_ds ,DOCS_DIR )


    json_path =RESULTS_DIR /"results.json"
    with open (json_path ,"w")as f :
        json .dump (all_stats ,f ,indent =2 )
    print (f"\nSaved results to {json_path }")


    print_table (all_stats )


    warnings =check_targets (all_stats )
    if warnings :
        print ("\n".join (warnings ))
    else :
        print ("All level targets are reachable with the documented fix.")


if __name__ =="__main__":
    main ()
