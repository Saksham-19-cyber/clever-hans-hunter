"""
app.py  –  Clever Hans Hunter  –  Gradio UI entry point.

Run with:
    python app.py

The app builds and caches the baseline models for each level, then
presents the full game UI with diagnosis tools and a training console.
"""

from __future__ import annotations 

import os 
import sys 
import copy 
import pickle 
import threading 
import time 
import json 
from pathlib import Path 
from typing import Optional 

import numpy as np 
import torch 
import gradio as gr 
import matplotlib 
matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
import matplotlib .patches as mpatches 
from PIL import Image 


from data import build_datasets ,ShortcutDataset ,CLASS_NAMES ,IMAGE_SIZE 
from model import build_model 
from train import (
TrainConfig ,TrainResult ,run_training ,train_baseline ,
evaluate ,LOSS_FN_MAP ,
)
from gradcam import make_gradcam_panel ,_tensor_to_pil 
from data import get_normalize_transform 


CACHE_DIR =Path ("./cache")
CACHE_DIR .mkdir (exist_ok =True )


SEED =42 
torch .manual_seed (SEED )
np .random .seed (SEED )


class GameState :
    def __init__ (self ):
        self .level :int =1 
        self .train_ds :Optional [ShortcutDataset ]=None 
        self .val_ds :Optional [ShortcutDataset ]=None 
        self .test_ds :Optional [ShortcutDataset ]=None 
        self .baseline :Optional [TrainResult ]=None 
        self .current :Optional [TrainResult ]=None 
        self .leaderboard :list =[]
        self .datasets_loaded :dict ={}
        self .baselines :dict ={}

STATE =GameState ()


LEVEL_TARGETS ={1 :0.72 ,2 :0.70 ,3 :0.68 }

LEVEL_SHORTCUTS ={
1 :"🎨 **Color Tint** – cats get a blue background, dogs a red one (95 % of train/val). "
"The model learns 'blue = cat' instead of actual cat features.",
2 :"🟦 **Corner Patch** – a small solid-colour square in the top-left corner "
"signals the class (blue square → cat, red square → dog).",
3 :"〰️ **Stripe Texture** – faint horizontal stripes overlay cat images, "
"vertical stripes overlay dog images.",
}

LEVEL_FIX_HINTS ={
1 :"Try **ColorJitter** augmentation (destroys the tint) + **group-reweighted loss**.",
2 :"Try **RandomErasing** or **RandomResizedCrop** (erases the corner patch) + group loss.",
3 :"Try **RandomGrayscale** (wipes texture channels) + group-reweighted loss.",
}


def _cache_path (level :int ,kind :str )->Path :
    return CACHE_DIR /f"level{level }_{kind }.pkl"


def _load_or_build_datasets (level :int ):
    if level in STATE .datasets_loaded :
        return STATE .datasets_loaded [level ]

    cp =_cache_path (level ,"datasets")
    if cp .exists ():
        with open (cp ,"rb")as f :
            datasets =pickle .load (f )
    else :
        datasets =build_datasets (level ,seed =SEED )
        with open (cp ,"wb")as f :
            pickle .dump (datasets ,f )

    STATE .datasets_loaded [level ]=datasets 
    return datasets 


def _load_or_build_baseline (level :int ,progress_cb =None ):
    if level in STATE .baselines :
        return STATE .baselines [level ]

    cp =_cache_path (level ,"baseline")
    if cp .exists ():
        with open (cp ,"rb")as f :
            baseline =pickle .load (f )
    else :
        tr ,va ,te =_load_or_build_datasets (level )
        baseline =train_baseline (level ,tr ,va ,te ,progress_cb )
        with open (cp ,"wb")as f :
            pickle .dump (baseline ,f )

    STATE .baselines [level ]=baseline 
    return baseline 


_DARK_BG ="#0f1117"
_CARD_BG ="#1a1d27"
_ACCENT ="#6c63ff"
_GREEN ="#4ade80"
_YELLOW ="#facc15"
_RED ="#f87171"
_TEXT ="#e2e8f0"

def _style_fig (fig ):
    fig .patch .set_facecolor (_DARK_BG )
    for ax in fig .axes :
        ax .set_facecolor (_CARD_BG )
        ax .tick_params (colors =_TEXT )
        ax .xaxis .label .set_color (_TEXT )
        ax .yaxis .label .set_color (_TEXT )
        ax .title .set_color (_TEXT )
        for spine in ax .spines .values ():
            spine .set_edgecolor ("#2a2d3e")
    return fig 


def make_group_bar_chart (group_accs :dict ,target :float )->plt .Figure :
    """Bar chart of per-group accuracies with a target line."""
    GROUP_LABELS =[
    "Cat\n(no shortcut)","Cat\n(shortcut)","Dog\n(no shortcut)","Dog\n(shortcut)"
    ]
    vals =[group_accs .get (g ,0.0 )for g in range (4 )]
    colors =[
    _GREEN if v >=target else (_YELLOW if v >=target *0.85 else _RED )
    for v in vals 
    ]

    fig ,ax =plt .subplots (figsize =(7 ,3.5 ))
    bars =ax .bar (GROUP_LABELS ,vals ,color =colors ,edgecolor ="#2a2d3e",zorder =2 )
    ax .axhline (target ,color =_ACCENT ,linestyle ="--",linewidth =1.5 ,
    label =f"Target ({target :.0%})",zorder =3 )
    ax .set_ylim (0 ,1.05 )
    ax .set_ylabel ("Accuracy",color =_TEXT )
    ax .set_title ("Per-Group Accuracy (test set)",color =_TEXT ,fontsize =11 )
    ax .legend (facecolor =_CARD_BG ,edgecolor ="#2a2d3e",labelcolor =_TEXT )
    ax .grid (axis ="y",color ="#2a2d3e",linestyle ="--",zorder =1 )

    for bar ,v in zip (bars ,vals ):
        ax .text (
        bar .get_x ()+bar .get_width ()/2 ,v +0.02 ,
        f"{v :.1%}",ha ="center",va ="bottom",color =_TEXT ,fontsize =9 ,
        )

    _style_fig (fig )
    fig .tight_layout ()
    return fig 


def make_confusion_matrix_fig (cm :np .ndarray )->plt .Figure :
    fig ,ax =plt .subplots (figsize =(4 ,3.5 ))
    im =ax .imshow (cm ,cmap ="Blues",vmin =0 )

    ax .set_xticks ([0 ,1 ]);ax .set_yticks ([0 ,1 ])
    ax .set_xticklabels (CLASS_NAMES ,color =_TEXT )
    ax .set_yticklabels (CLASS_NAMES ,color =_TEXT )
    ax .set_xlabel ("Predicted",color =_TEXT )
    ax .set_ylabel ("True",color =_TEXT )
    ax .set_title ("Confusion Matrix",color =_TEXT ,fontsize =11 )

    total =cm .sum ()
    for i in range (2 ):
        for j in range (2 ):
            pct =f"{cm [i ,j ]}\n({cm [i ,j ]/max (total ,1 ):.1%})"
            ax .text (j ,i ,pct ,ha ="center",va ="center",
            color ="white"if cm [i ,j ]>cm .max ()*0.5 else _TEXT ,fontsize =9 )

    _style_fig (fig )
    fig .tight_layout ()
    return fig 


def make_accuracy_comparison_fig (baseline :TrainResult ,current :Optional [TrainResult ]=None )->plt .Figure :
    labels =["Val Acc\n(misleading)","Test Avg Acc","Worst-Group\nAcc"]

    b_vals =[baseline .val_acc ,baseline .test_avg_acc ,baseline .worst_group_acc ]
    c_vals =(
    [current .val_acc ,current .test_avg_acc ,current .worst_group_acc ]
    if current else None 
    )

    x =np .arange (len (labels ))
    width =0.35 if c_vals else 0.5 

    fig ,ax =plt .subplots (figsize =(7 ,3.5 ))
    ax .bar (x -(width /2 if c_vals else 0 ),b_vals ,width ,
    label ="Baseline",color =_RED ,alpha =0.85 ,zorder =2 )
    if c_vals :
        ax .bar (x +width /2 ,c_vals ,width ,
        label ="Your model",color =_GREEN ,alpha =0.85 ,zorder =2 )

    ax .set_xticks (x );ax .set_xticklabels (labels ,color =_TEXT )
    ax .set_ylim (0 ,1.1 )
    ax .set_ylabel ("Accuracy",color =_TEXT )
    ax .set_title ("Accuracy Comparison",color =_TEXT ,fontsize =11 )
    ax .legend (facecolor =_CARD_BG ,edgecolor ="#2a2d3e",labelcolor =_TEXT )
    ax .grid (axis ="y",color ="#2a2d3e",linestyle ="--",zorder =1 )

    for i ,v in enumerate (b_vals ):
        offset =-width /2 if c_vals else 0 
        ax .text (i +offset ,v +0.02 ,f"{v :.1%}",ha ="center",color =_TEXT ,fontsize =8 )
    if c_vals :
        for i ,v in enumerate (c_vals ):
            ax .text (i +width /2 ,v +0.02 ,f"{v :.1%}",ha ="center",color =_TEXT ,fontsize =8 )

    _style_fig (fig )
    fig .tight_layout ()
    return fig 


def make_training_history_fig (history :list )->plt .Figure :
    epochs =[h ["epoch"]for h in history ]
    tr_acc =[h ["train_acc"]for h in history ]
    va_acc =[h ["val_acc"]for h in history ]

    fig ,ax =plt .subplots (figsize =(6 ,3 ))
    ax .plot (epochs ,tr_acc ,"o-",color =_ACCENT ,label ="Train acc",linewidth =2 )
    ax .plot (epochs ,va_acc ,"s--",color =_YELLOW ,label ="Val acc",linewidth =2 )
    ax .set_xlabel ("Epoch",color =_TEXT )
    ax .set_ylabel ("Accuracy",color =_TEXT )
    ax .set_title ("Training History",color =_TEXT ,fontsize =11 )
    ax .legend (facecolor =_CARD_BG ,edgecolor ="#2a2d3e",labelcolor =_TEXT )
    ax .set_ylim (0 ,1.05 )
    ax .grid (color ="#2a2d3e",linestyle ="--")
    _style_fig (fig )
    fig .tight_layout ()
    return fig 


def _pil_grid (pil_images :list ,cols :int =5 )->Image .Image :
    """Tile a list of PIL images into a grid."""
    if not pil_images :
        return Image .new ("RGB",(320 ,64 ),color ="#1a1d27")
    w ,h =pil_images [0 ].size 
    rows =(len (pil_images )+cols -1 )//cols 
    grid =Image .new ("RGB",(cols *w ,rows *h ),color ="#1a1d27")
    for idx ,img in enumerate (pil_images ):
        r ,c =divmod (idx ,cols )
        grid .paste (img ,(c *w ,r *h ))
    return grid 


def run_gradcam (sample_idx :int ,split :str ="test"):
    """Explain a single sample and return 3 panel images + text."""
    model =(STATE .current or STATE .baseline ).model if STATE .baseline else None 
    if model is None :
        return None ,None ,None ,"Load a level first."

    if split =="test":
        ds =STATE .test_ds 
    elif split =="val":
        ds =STATE .val_ds 
    else :
        ds =STATE .train_ds 

    if ds is None or sample_idx >=len (ds ):
        return None ,None ,None ,"Invalid sample index."

    normalize =get_normalize_transform ()
    raw_img ,true_lbl ,group_id =ds [sample_idx ]
    img_norm =normalize (raw_img .unsqueeze (0 ))

    try :
        orig_pil ,heat_pil ,overlay_pil ,pred_cls ,conf =make_gradcam_panel (
        model ,img_norm 
        )
    except Exception as e :
        return None ,None ,None ,f"Grad-CAM error: {e }"

    label_text =(
    f"**True class:** {CLASS_NAMES [true_lbl ]}  |  "
    f"**Prediction:** {CLASS_NAMES [pred_cls ]}  |  "
    f"**Confidence:** {conf :.1%}  |  "
    f"**Group:** {group_id }  |  "
    f"{'✅ Correct'if pred_cls ==true_lbl else '❌ Wrong'}"
    )
    return orig_pil ,heat_pil ,overlay_pil ,label_text 


def build_suspicious_gallery (result :TrainResult ,ds :ShortcutDataset ,n :int =10 ):
    images ,captions =[],[]
    for idx ,true_lbl ,pred_lbl ,conf in result .suspicious [:n ]:
        if idx >=len (ds ):
            continue 
        raw_img =ds [idx ][0 ]
        pil =_tensor_to_pil (raw_img )
        pil =pil .resize ((96 ,96 ),Image .NEAREST )
        caption =(
        f"True:{CLASS_NAMES [true_lbl ]} "
        f"Pred:{CLASS_NAMES [pred_lbl ]} "
        f"({conf :.0%})"
        )
        images .append ((pil ,caption ))
    return images 


def _cfg_summary (cfg :TrainConfig )->str :
    augs =[]
    if cfg .use_color_jitter :augs .append ("ColorJitter")
    if cfg .use_random_gray :augs .append ("RandGray")
    if cfg .use_random_crop :augs .append ("RandCrop")
    if cfg .use_random_erasing :augs .append ("Erasing")
    if cfg .use_hflip :augs .append ("HFlip")
    if cfg .use_gaussian_noise :augs .append ("Noise")
    return (
    f"Mode={cfg .finetune_mode } | Loss={cfg .loss_fn_name } | "
    f"LR={cfg .learning_rate } | Ep={cfg .epochs } | "
    f"Aug=[{', '.join (augs )or 'none'}]"
    )


def _leaderboard_md (entries :list ,target :float )->str :
    if not entries :
        return "_No attempts yet. Train a model to start!_"
    rows =["| Rank | Score (WG Acc) | Settings |","|------|--------------|---------|"]
    for i ,(score ,summary )in enumerate (entries ,1 ):
        medal ="🥇"if i ==1 else ("🥈"if i ==2 else ("🥉"if i ==3 else f"#{i }"))
        won ="✅"if score >=target else ""
        rows .append (f"| {medal }{won } | {score :.1%} | {summary } |")
    return "\n".join (rows )


CSS ="""
/* ─── Global ───────────────────────────────────────────── */
:root {
    --bg:      #0f1117;
    --card:    #1a1d27;
    --accent:  #6c63ff;
    --accent2: #a78bfa;
    --green:   #4ade80;
    --red:     #f87171;
    --yellow:  #facc15;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --border:  #2a2d3e;
    --radius:  12px;
}
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, sans-serif;
}
/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
}
/* Header banner */
.hero {
    background: linear-gradient(135deg, #1a1d27 0%, #12151f 50%, #1a1027 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #6c63ff, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero p { color: var(--muted); margin: 0; font-size: 1rem; }
/* Score bar */
.score-bar-wrap { margin-top: 0.4rem; }
/* Tabs */
.tab-nav { border-bottom: 2px solid var(--border) !important; }
.tab-nav button { color: var(--muted) !important; }
.tab-nav button.selected { color: var(--accent2) !important; border-bottom: 2px solid var(--accent2) !important; }
/* Inputs */
label { color: var(--text) !important; }
.gr-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.primary-btn { background: linear-gradient(135deg, #6c63ff, #a78bfa) !important; color: white !important; }
.primary-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(108,99,255,0.4) !important; }
.danger-btn { background: linear-gradient(135deg, #f87171, #ef4444) !important; color: white !important; }
/* Progress bar */
.progress-wrap { height: 12px; background: var(--border); border-radius: 99px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 99px; transition: width 0.4s ease;
    background: linear-gradient(90deg, #6c63ff, #a78bfa); }
.score-number { font-size: 2.4rem; font-weight: 800; color: var(--accent2); }
"""

def build_ui ():
    with gr .Blocks (css =CSS ,theme =gr .themes .Base (),title ="Clever Hans Hunter")as demo :


        gr .HTML ("""
        <div class="hero">
          <h1>🔍 Clever Hans Hunter</h1>
          <p>Find the shortcut your model learned — then fix it. Score = Worst-Group Accuracy on a rigged test set.</p>
        </div>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        """)


        with gr .Row ():
            with gr .Column (scale =2 ):
                level_radio =gr .Radio (
                choices =[("🎨 Level 1 – Color Tint",1 ),
                ("🟦 Level 2 – Corner Patch",2 ),
                ("〰️ Level 3 – Stripe Texture",3 )],
                value =1 ,label ="Select Shortcut Level",
                )
            with gr .Column (scale =1 ):
                load_btn =gr .Button ("⚡ Load Level",elem_classes =["primary-btn"])
                load_status =gr .Markdown ("_Click **Load Level** to begin._")


        with gr .Row ():
            with gr .Column ():
                gr .HTML ('<div class="card">')
                baseline_md =gr .Markdown ("**Baseline** — not loaded yet")
                gr .HTML ('</div>')
            with gr .Column ():
                gr .HTML ('<div class="card">')
                score_md =gr .Markdown ("**Your Score** — not trained yet")
                score_bar =gr .HTML ('<div class="progress-wrap"><div class="progress-fill" style="width:0%"></div></div>')
                gr .HTML ('</div>')


        with gr .Tabs ():


            with gr .TabItem ("🔬 Diagnose"):
                gr .Markdown ("### Accuracy Breakdown")
                acc_fig =gr .Plot (label ="Accuracy Comparison")

                with gr .Row ():
                    group_fig =gr .Plot (label ="Per-Group Accuracy")
                    cm_fig =gr .Plot (label ="Confusion Matrix")

                gr .Markdown ("---")
                gr .Markdown ("### 🎯 Grad-CAM Explorer")
                with gr .Row ():
                    with gr .Column (scale =1 ):
                        cam_split =gr .Dropdown (
                        ["test","val","train"],
                        value ="test",label ="Dataset split"
                        )
                        cam_idx =gr .Slider (0 ,499 ,step =1 ,value =0 ,label ="Sample index")
                        cam_btn =gr .Button ("▶ Explain this sample",elem_classes =["primary-btn"])
                    with gr .Column (scale =3 ):
                        with gr .Row ():
                            cam_orig =gr .Image (label ="Original",height =160 )
                            cam_heat =gr .Image (label ="Heatmap",height =160 )
                            cam_overlay =gr .Image (label ="Overlay",height =160 )
                        cam_label =gr .Markdown ("_Run Grad-CAM to see predictions._")

                gr .Markdown ("---")
                gr .Markdown ("### 🚨 Most Confidently Wrong Predictions")
                gallery =gr .Gallery (
                label ="Suspicious samples (true label / prediction / confidence)",
                columns =5 ,height =220 ,object_fit ="contain",
                )


            with gr .TabItem ("🏋️ Train"):
                gr .Markdown ("### Player Toolbox — pick your fixes and hit Train")

                with gr .Row ():
                    with gr .Column (scale =1 ):
                        gr .Markdown ("**🔧 Fine-tuning Depth**")
                        ft_depth =gr .Radio (
                        [("Head only (fastest)","head_only"),
                        ("Last block + head","last_block"),
                        ("Full fine-tune","full")],
                        value ="head_only",label ="Trainable layers",
                        )
                        gr .Markdown ("**⚙️ Optimiser**")
                        opt_choice =gr .Radio (
                        [("Adam","adam"),("SGD (momentum)","sgd")],
                        value ="adam",label ="Optimizer",
                        )
                        lr_slider =gr .Slider (1e-5 ,1e-1 ,value =1e-3 ,step =1e-5 ,
                        label ="Learning rate",scale =1 )
                        epochs_sl =gr .Slider (1 ,5 ,value =3 ,step =1 ,label ="Epochs")

                    with gr .Column (scale =1 ):
                        gr .Markdown ("**📦 Loss Function**")
                        loss_choice =gr .Radio (
                        [("Cross-Entropy","cross_entropy"),
                        ("Class-Weighted CE","class_weighted_ce"),
                        ("Group-Reweighted CE ⭐","group_reweighted_ce")],
                        value ="cross_entropy",label ="Loss",
                        )
                        gr .Markdown (
                        """<small>⭐ **Group-reweighted loss** uses the group labels (shortcut present / absent) 
                            to up-weight samples where the shortcut is absent, forcing the model to learn 
                            real features. This is the key mechanic for beating the shortcut.</small>"""
                        )

                    with gr .Column (scale =1 ):
                        gr .Markdown ("**🎲 Augmentations**")
                        aug_hflip =gr .Checkbox (label ="Horizontal Flip")
                        aug_cj =gr .Checkbox (label ="ColorJitter")
                        aug_gray =gr .Checkbox (label ="RandomGrayscale")
                        aug_crop =gr .Checkbox (label ="RandomResizedCrop")
                        aug_erase =gr .Checkbox (label ="RandomErasing / Cutout")
                        aug_noise =gr .Checkbox (label ="Gaussian Noise")
                        aug_str =gr .Slider (0.1 ,1.0 ,value =0.5 ,step =0.05 ,
                        label ="Augmentation strength")

                train_btn =gr .Button ("🚀 Train My Model",elem_classes =["primary-btn"])
                train_status =gr .Markdown ("_Awaiting training command…_")
                hist_fig =gr .Plot (label ="Training History")


            with gr .TabItem ("🏆 Leaderboard & Reveal"):
                gr .Markdown ("### Session Leaderboard")
                leaderboard_md =gr .Markdown ("_No attempts yet._")

                gr .Markdown ("---")
                gr .Markdown ("### 🔓 Reveal the Shortcut")
                reveal_btn =gr .Button ("👁️ Reveal",elem_classes =["danger-btn"])
                reveal_text =gr .Markdown ("_Click Reveal to see the planted shortcut and the best fix._")


        def on_load_level (level ,progress =gr .Progress ()):
            level =int (level )
            STATE .level =level 
            target =LEVEL_TARGETS [level ]

            progress (0.0 ,desc ="Loading datasets …")
            STATE .train_ds ,STATE .val_ds ,STATE .test_ds =_load_or_build_datasets (level )

            progress (0.3 ,desc ="Loading/training baseline …")

            def _prog (frac ,msg ):
                progress (0.3 +frac *0.65 ,desc =msg )

            baseline =_load_or_build_baseline (level ,_prog )
            STATE .baseline =baseline 
            STATE .current =None 

            bline_text =(
            f"**Baseline (no fixes)**\n\n"
            f"- Val Acc: **{baseline .val_acc :.1%}** ← looks great!\n"
            f"- Test Avg Acc: **{baseline .test_avg_acc :.1%}**\n"
            f"- ⚠️ Worst-Group Acc: **{baseline .worst_group_acc :.1%}** ← this is what counts"
            )

            acc_f =make_accuracy_comparison_fig (baseline )
            grp_f =make_group_bar_chart (baseline .group_accs ,target )
            cm_f =make_confusion_matrix_fig (baseline .confusion_matrix )
            gallery_items =build_suspicious_gallery (baseline ,STATE .test_ds )
            hist_f =make_training_history_fig (baseline .history )

            score_text =f"**Your Score** — train a model to compete!"
            bar_html ='<div class="progress-wrap"><div class="progress-fill" style="width:0%"></div></div>'
            board_md =_leaderboard_md (STATE .leaderboard ,target )
            status_md =f"✅ Level {level } loaded! Target worst-group accuracy: **{target :.0%}**"

            progress (1.0 ,desc ="Ready!")
            return (
            status_md ,
            bline_text ,
            score_text ,
            bar_html ,
            acc_f ,grp_f ,cm_f ,
            gallery_items ,
            hist_f ,
            board_md ,
            )

        load_btn .click (
        on_load_level ,
        inputs =[level_radio ],
        outputs =[
        load_status ,baseline_md ,score_md ,score_bar ,
        acc_fig ,group_fig ,cm_fig ,
        gallery ,
        hist_fig ,
        leaderboard_md ,
        ],
        )


        cam_btn .click (
        run_gradcam ,
        inputs =[cam_idx ,cam_split ],
        outputs =[cam_orig ,cam_heat ,cam_overlay ,cam_label ],
        )


        def on_train (
        ft_mode ,optimizer ,lr ,epochs ,
        loss_name ,
        hflip ,cj ,gray ,crop ,erase ,noise ,aug_strength ,
        progress =gr .Progress (),
        ):
            if STATE .train_ds is None :
                return (
                "⚠️ Load a level first.",
                None ,None ,None ,None ,
                "—","—","—",
                )

            level =STATE .level 
            target =LEVEL_TARGETS [level ]

            cfg =TrainConfig (
            finetune_mode =ft_mode ,
            optimizer =optimizer ,
            learning_rate =float (lr ),
            epochs =int (epochs ),
            loss_fn_name =loss_name ,
            use_hflip =hflip ,
            use_color_jitter =cj ,
            use_random_gray =gray ,
            use_random_crop =crop ,
            use_random_erasing =erase ,
            use_gaussian_noise =noise ,
            aug_strength =float (aug_strength ),
            )

            progress (0.0 ,desc ="Starting training …")

            def _prog (frac ,msg ):
                progress (frac ,desc =msg )

            result =run_training (
            cfg ,STATE .train_ds ,STATE .val_ds ,STATE .test_ds ,
            progress_cb =_prog ,
            )
            STATE .current =result 


            entry =(result .worst_group_acc ,_cfg_summary (cfg ))
            STATE .leaderboard .append (entry )
            STATE .leaderboard .sort (key =lambda x :-x [0 ])


            acc_f =make_accuracy_comparison_fig (STATE .baseline ,result )
            grp_f =make_group_bar_chart (result .group_accs ,target )
            cm_f =make_confusion_matrix_fig (result .confusion_matrix )
            hist_f =make_training_history_fig (result .history )
            gallery_items =build_suspicious_gallery (result ,STATE .test_ds )
            board_md =_leaderboard_md (STATE .leaderboard ,target )

            pct =result .worst_group_acc /target 
            bar_pct =min (100 ,int (pct *100 ))
            bar_color ="#4ade80"if pct >=1.0 else ("#facc15"if pct >=0.8 else "#a78bfa")
            bar_html =(
            f'<div class="progress-wrap">'
            f'<div class="progress-fill" style="width:{bar_pct }%; background:{bar_color }"></div>'
            f'</div>'
            f'<p style="color:{bar_color };font-size:0.9rem;margin-top:4px">'
            f'{bar_pct }% toward target ({target :.0%})'
            f"{'  🎉 Level Complete!'if pct >=1.0 else ''}</p>"
            )

            score_text =(
            f"**Your Score**\n\n"
            f"- Val Acc: {result .val_acc :.1%}\n"
            f"- Test Avg: {result .test_avg_acc :.1%}\n"
            f"- **Worst-Group: {result .worst_group_acc :.1%}**"
            +(" 🏆"if result .worst_group_acc >=target else "")
            )

            won =result .worst_group_acc >=target 
            status =(
            f"{'🏆 Level Complete! 'if won else ''}Training done | "
            f"Worst-group acc = **{result .worst_group_acc :.1%}** (target {target :.0%})"
            )

            progress (1.0 ,desc ="Done!")
            return (
            status ,
            acc_f ,grp_f ,cm_f ,hist_f ,
            score_text ,bar_html ,board_md ,
            gallery_items ,
            )

        train_btn .click (
        on_train ,
        inputs =[
        ft_depth ,opt_choice ,lr_slider ,epochs_sl ,
        loss_choice ,
        aug_hflip ,aug_cj ,aug_gray ,aug_crop ,aug_erase ,aug_noise ,aug_str ,
        ],
        outputs =[
        train_status ,
        acc_fig ,group_fig ,cm_fig ,hist_fig ,
        score_md ,score_bar ,leaderboard_md ,
        gallery ,
        ],
        )


        def on_reveal ():
            lv =STATE .level 
            txt =(
            f"## 🔓 Level {lv } — Shortcut Revealed\n\n"
            f"### What was planted\n{LEVEL_SHORTCUTS [lv ]}\n\n"
            f"### Why average accuracy is misleading\n"
            f"The model achieves **{STATE .baseline .val_acc :.1%}** validation accuracy "
            f"because the shortcut is *correlated* with the label in val. "
            f"But on the test set where shortcuts are random (50/50), "
            f"samples that lack the expected shortcut fool the model — "
            f"worst-group acc collapses to **{STATE .baseline .worst_group_acc :.1%}**.\n\n"
            f"### Best fix\n{LEVEL_FIX_HINTS [lv ]}\n\n"
            f"### Course concepts demonstrated\n"
            f"| Concept | How it appears here |\n"
            f"|---------|--------------------|\n"
            f"| Transfer Learning | ResNet18 backbone (ImageNet weights) — backbone features are reused; only the head is replaced by default |\n"
            f"| Data Augmentation | ColorJitter / RandomErasing remove the shortcut signal from training images |\n"
            f"| Loss Functions | Group-reweighted CE directly penalises failures on shortcut-absent groups |\n"
            f"| Metrics | Val accuracy ≈ {STATE .baseline .val_acc :.1%} looks fine; worst-group acc ≈ {STATE .baseline .worst_group_acc :.1%} exposes the failure — always report both! |\n"
            )
            return txt 

        reveal_btn .click (on_reveal ,outputs =[reveal_text ])

    return demo 


if __name__ =="__main__":
    print ("╔══════════════════════════════════════════════╗")
    print ("║       Clever Hans Hunter  –  starting up     ║")
    print ("╚══════════════════════════════════════════════╝")
    print ("Pre-caching baseline models for all 3 levels …")
    for lv in [1 ,2 ,3 ]:
        print (f"  Level {lv } … ",end ="",flush =True )
        _load_or_build_datasets (lv )
        _load_or_build_baseline (lv )
        print ("done")

    print ("\nLaunching Gradio UI …")
    demo =build_ui ()
    demo .launch (
    server_name ="0.0.0.0",
    server_port =7860 ,
    share =False ,
    inbrowser =True ,
    )
