"""
streamlit_app.py  –  Clever Hans Hunter  (Streamlit port)

Run:
    streamlit run streamlit_app.py

State management
  - st.session_state  : current model, leaderboard, unlocked levels
  - @st.cache_resource: datasets and baseline models (heavy, shared across reruns)
"""

from __future__ import annotations 

import copy 
import io 
import os 
import pickle 
import sys 
import time 
from pathlib import Path 
from typing import Dict ,List ,Optional ,Tuple 

import matplotlib 
matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
import numpy as np 
import streamlit as st 
import torch 


sys .path .insert (0 ,str (Path (__file__ ).parent ))

from data import (
CLASS_NAMES ,IMAGE_SIZE ,build_datasets ,get_normalize_transform ,
shortcut_correlation ,ShortcutDataset ,
)
from gradcam import _tensor_to_pil ,make_gradcam_panel 
from model import build_model 
from train import (
LOSS_FN_MAP ,TrainConfig ,TrainResult ,evaluate ,run_training ,train_baseline ,
_compute_group_accs ,_confusion_matrix ,
)


ROOT =Path (__file__ ).parent 
CACHE_DIR =ROOT /"cache"
CHECKPOINTS_DIR =ROOT /"checkpoints"
SEED =42 

LEVEL_TARGETS ={1 :0.55 ,2 :0.60 ,3 :0.52 }


LEVEL_SHORTCUTS ={
1 :("Color Tint",
"Cat images get a blue background tint; dog images a red tint. "
"Correlated with the label 95 % of the time in train/val. "
"In the test set the tint is assigned randomly (50/50)."),
2 :("Corner Patch",
"A small solid-colour square (blue=cat, red=dog) is pasted in the "
"top-left corner of every image. Same 95 %/50 % train-vs-test rule."),
3 :("Stripe Texture",
"Faint horizontal stripes overlay cat images; vertical stripes "
"overlay dog images. Same 95 %/50 % train-vs-test rule."),
}

LEVEL_FIXES ={
1 :"ColorJitter + RandomGrayscale augmentation + group-reweighted CE loss.",
2 :"RandomErasing + RandomResizedCrop + group-reweighted CE loss.",
3 :"RandomGrayscale + Gaussian Noise + group-reweighted CE loss.",
}

GROUP_LABELS =[
"Cat (no shortcut)","Cat (shortcut)","Dog (no shortcut)","Dog (shortcut)"
]

_BG ="#0f1117"
_CARD ="#1a1d27"
_ACC ="#6c63ff"
_GRN ="#4ade80"
_YEL ="#facc15"
_RED ="#f87171"
_TXT ="#e2e8f0"
_BRD ="#2a2d3e"


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


def _fig_to_img (fig ):
    buf =io .BytesIO ()
    fig .savefig (buf ,format ="png",bbox_inches ="tight",dpi =110 )
    plt .close (fig )
    buf .seek (0 )
    return buf .read ()


@st .cache_resource (show_spinner ="Loading dataset ...")
def cached_datasets (level :int ):
    try :
        CACHE_DIR .mkdir (exist_ok =True )
    except Exception :
        pass 
    cp =CACHE_DIR /f"level{level }_datasets.pkl"
    if cp .exists ():
        try :
            with open (cp ,"rb")as f :
                return pickle .load (f )
        except Exception :
            pass 
    result =build_datasets (level ,seed =SEED )
    try :
        with open (cp ,"wb")as f :
            pickle .dump (result ,f )
    except Exception :
        pass 
    return result 


@st .cache_resource (show_spinner ="Loading baseline ...")
def cached_baseline (level :int ):
    ckpt_path =CHECKPOINTS_DIR /f"level{level }_baseline.pt"
    if ckpt_path .exists ():
        try :
            ckpt =torch .load (ckpt_path ,map_location ="cpu",weights_only =False )
            model =build_model (finetune_mode ="head_only",pretrained =True )
            if "fc_state_dict"in ckpt :
                model .fc .load_state_dict (ckpt ["fc_state_dict"])
            elif "model_state_dict"in ckpt :
                model .load_state_dict (ckpt ["model_state_dict"])
            model .eval ()
            return TrainResult (
            val_acc =ckpt ["val_acc"],
            test_avg_acc =ckpt ["test_avg_acc"],
            worst_group_acc =ckpt ["worst_group_acc"],
            group_accs =ckpt ["group_accs"],
            confusion_matrix =np .array (ckpt ["confusion_matrix"]),
            history =ckpt ["history"],
            model =model ,
            suspicious =ckpt .get ("suspicious",[]),
            )
        except Exception :
            pass 

    try :
        CACHE_DIR .mkdir (exist_ok =True )
    except Exception :
        pass 
    cp =CACHE_DIR /f"level{level }_baseline.pkl"
    if cp .exists ():
        try :
            with open (cp ,"rb")as f :
                return pickle .load (f )
        except Exception :
            pass 
    tr ,va ,te =cached_datasets (level )
    result =train_baseline (level ,tr ,va ,te )
    try :
        with open (cp ,"wb")as f :
            pickle .dump (result ,f )
    except Exception :
        pass 
    return result 


def _init_state ():
    defaults ={
    "current_result":None ,
    "leaderboard":[],
    "unlocked":set (),
    "active_level":1 ,
    }
    for k ,v in defaults .items ():
        if k not in st .session_state :
            st .session_state [k ]=v 


def fig_group_bars (group_accs ,target ,title ="Per-Group Accuracy (test set)"):
    vals =[group_accs .get (g ,0.0 )for g in range (4 )]
    colors =[_GRN if v >=target else (_YEL if v >=target *0.85 else _RED )
    for v in vals ]
    fig ,ax =plt .subplots (figsize =(7 ,3.2 ))
    bars =ax .bar (GROUP_LABELS ,vals ,color =colors ,edgecolor =_BRD ,zorder =2 )
    ax .axhline (target ,color =_ACC ,linestyle ="--",linewidth =1.5 ,
    label =f"Target {target :.0%}",zorder =3 )
    ax .set_ylim (0 ,1.08 )
    ax .set_ylabel ("Accuracy")
    ax .set_title (title )
    ax .legend (facecolor =_CARD ,edgecolor =_BRD ,labelcolor =_TXT )
    ax .grid (axis ="y",color =_BRD ,linestyle ="--",zorder =1 )
    for bar ,v in zip (bars ,vals ):
        ax .text (bar .get_x ()+bar .get_width ()/2 ,v +0.015 ,
        f"{v :.1%}",ha ="center",color =_TXT ,fontsize =8 )
    _style (fig )
    fig .tight_layout ()
    return _fig_to_img (fig )


def fig_comparison (baseline ,current =None ):
    labels =["Val Acc\n(misleading)","Test Avg","Worst-Group\n(score)"]
    bv =[baseline .val_acc ,baseline .test_avg_acc ,baseline .worst_group_acc ]
    cv =([current .val_acc ,current .test_avg_acc ,current .worst_group_acc ]
    if current else None )
    x =np .arange (3 )
    w =0.35 if cv else 0.5 
    fig ,ax =plt .subplots (figsize =(7 ,3.2 ))
    ax .bar (x -(w /2 if cv else 0 ),bv ,w ,label ="Baseline",color =_RED ,alpha =0.85 ,zorder =2 )
    if cv :
        ax .bar (x +w /2 ,cv ,w ,label ="Your model",color =_GRN ,alpha =0.85 ,zorder =2 )
    ax .set_xticks (x );ax .set_xticklabels (labels )
    ax .set_ylim (0 ,1.12 );ax .set_ylabel ("Accuracy")
    ax .set_title ("Accuracy Comparison")
    ax .legend (facecolor =_CARD ,edgecolor =_BRD ,labelcolor =_TXT )
    ax .grid (axis ="y",color =_BRD ,linestyle ="--",zorder =1 )
    for i ,v in enumerate (bv ):
        ax .text (i -(w /2 if cv else 0 ),v +0.02 ,f"{v :.1%}",
        ha ="center",color =_TXT ,fontsize =8 )
    if cv :
        for i ,v in enumerate (cv ):
            ax .text (i +w /2 ,v +0.02 ,f"{v :.1%}",ha ="center",color =_TXT ,fontsize =8 )
    _style (fig )
    fig .tight_layout ()
    return _fig_to_img (fig )


def fig_cm (cm ):
    fig ,ax =plt .subplots (figsize =(4 ,3.2 ))
    ax .imshow (cm ,cmap ="Blues",vmin =0 )
    ax .set_xticks ([0 ,1 ]);ax .set_yticks ([0 ,1 ])
    ax .set_xticklabels (CLASS_NAMES );ax .set_yticklabels (CLASS_NAMES )
    ax .set_xlabel ("Predicted");ax .set_ylabel ("True")
    ax .set_title ("Confusion Matrix")
    total =cm .sum ()
    for i in range (2 ):
        for j in range (2 ):
            ax .text (j ,i ,f"{cm [i ,j ]}\n({cm [i ,j ]/max (total ,1 ):.1%})",
            ha ="center",va ="center",
            color ="white"if cm [i ,j ]>cm .max ()*0.5 else _TXT ,fontsize =9 )
    _style (fig )
    fig .tight_layout ()
    return _fig_to_img (fig )


def fig_history (history ):
    ep =[h ["epoch"]for h in history ]
    tr =[h ["train_acc"]for h in history ]
    va =[h ["val_acc"]for h in history ]
    fig ,ax =plt .subplots (figsize =(5 ,2.8 ))
    ax .plot (ep ,tr ,"o-",color =_ACC ,label ="Train",linewidth =2 )
    ax .plot (ep ,va ,"s--",color =_YEL ,label ="Val",linewidth =2 )
    ax .set_ylim (0 ,1.05 );ax .set_xlabel ("Epoch");ax .set_ylabel ("Accuracy")
    ax .set_title ("Training History")
    ax .legend (facecolor =_CARD ,edgecolor =_BRD ,labelcolor =_TXT )
    ax .grid (color =_BRD ,linestyle ="--")
    _style (fig )
    fig .tight_layout ()
    return _fig_to_img (fig )


def cfg_summary (cfg ):
    augs =[]
    if cfg .use_color_jitter :augs .append ("CJ")
    if cfg .use_random_gray :augs .append ("Gray")
    if cfg .use_random_crop :augs .append ("Crop")
    if cfg .use_random_erasing :augs .append ("Erase")
    if cfg .use_hflip :augs .append ("Flip")
    if cfg .use_gaussian_noise :augs .append ("Noise")
    return (f"depth={cfg .finetune_mode } loss={cfg .loss_fn_name } "
    f"lr={cfg .learning_rate } ep={cfg .epochs } "
    f"aug=[{','.join (augs )or 'none'}]")


def score_bar_html (score ,target ):
    pct =min (1.0 ,score /target )*100 
    color ="#4ade80"if pct >=100 else ("#facc15"if pct >=80 else "#a78bfa")
    label ="Level Complete!"if pct >=100 else f"{pct :.0f}% toward {target :.0%} target"
    return (f'<div style="background:#2a2d3e;border-radius:99px;height:14px;'
    f'overflow:hidden;margin:6px 0">'
    f'<div style="width:{pct :.1f}%;background:{color };height:100%;'
    f'border-radius:99px"></div></div>'
    f'<p style="color:{color };font-size:0.85rem;margin:2px 0">{label }</p>')


def game_train (level :int ,cfg :TrainConfig )->TrainResult :
    """Run a training experiment. Returns TrainResult."""
    tr ,va ,te =cached_datasets (level )
    return run_training (cfg ,tr ,va ,te )


def game_leaderboard_append (leaderboard :list ,level :int ,
result :TrainResult ,cfg :TrainConfig )->list :
    """Append result to leaderboard and return sorted copy."""
    entry =(level ,result .worst_group_acc ,cfg_summary (cfg ))
    updated =leaderboard +[entry ]
    updated .sort (key =lambda x :-x [1 ])
    return updated 


def game_is_unlocked (result :TrainResult ,level :int )->bool :
    """True when worst-group accuracy meets the level target."""
    return result .worst_group_acc >=LEVEL_TARGETS [level ]


def main ():
    st .set_page_config (
    page_title ="Clever Hans Hunter",
    page_icon ="search",
    layout ="wide",
    )
    _init_state ()

    st .markdown ("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif}
    .hero{background:linear-gradient(135deg,#1a1d27,#12151f,#1a1027);
          border:1px solid #2a2d3e;border-radius:16px;padding:1.8rem;
          text-align:center;margin-bottom:1rem}
    .hero h1{font-size:2rem;font-weight:800;
             background:linear-gradient(90deg,#a78bfa,#6c63ff,#818cf8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin:0 0 0.4rem 0}
    .hero p{color:#94a3b8;margin:0}
    .mcard{background:#1a1d27;border:1px solid #2a2d3e;border-radius:12px;
           padding:1rem;text-align:center}
    .mbig{font-size:1.8rem;font-weight:800;color:#a78bfa}
    .mlbl{font-size:0.8rem;color:#94a3b8}
    </style>
    """,unsafe_allow_html =True )

    st .markdown ("""
    <div class="hero">
      <h1>Clever Hans Hunter</h1>
      <p>Your model has high validation accuracy but learned a <em>shortcut</em>.
         Find it. Fix it. Score = Worst-Group Accuracy on a rigged test set.</p>
    </div>
    """,unsafe_allow_html =True )


    with st .sidebar :
        st .markdown ("## Game Controls")
        level =st .radio (
        "Select Shortcut Level",
        options =[1 ,2 ,3 ],
        format_func =lambda x :{
        1 :"Level 1 - Color Tint",
        2 :"Level 2 - Corner Patch",
        3 :"Level 3 - Stripe Texture",
        }[x ],
        index =st .session_state .active_level -1 ,
        )
        if level !=st .session_state .active_level :
            st .session_state .active_level =level 
            st .session_state .current_result =None 

        target =LEVEL_TARGETS [level ]
        current =st .session_state .current_result 

        st .markdown (f"**Target:** Worst-Group >= **{target :.0%}**")
        if current is not None :
            sc =current .worst_group_acc 
            st .markdown (f"**Your score:** {sc :.1%}")
            st .markdown (score_bar_html (sc ,target ),unsafe_allow_html =True )
        else :
            st .caption ("Train a model to see your score.")

        st .markdown ("---")
        st .markdown ("### Session Leaderboard")
        board =[e for e in st .session_state .leaderboard if e [0 ]==level ]
        if board :
            for rank ,(_ ,sc2 ,summ )in enumerate (board [:5 ],1 ):
                medal =["1st","2nd","3rd","4th","5th"][rank -1 ]
                won =" DONE"if sc2 >=target else ""
                st .markdown (f"**{medal }{won }** `{sc2 :.1%}`")
                st .caption (summ )
        else :
            st .caption ("No attempts yet.")


    train_ds ,val_ds ,test_ds =cached_datasets (level )
    baseline =cached_baseline (level )


    c1 ,c2 ,c3 ,c4 =st .columns (4 )
    with c1 :
        st .markdown (f"<div class='mcard'><div class='mbig'>{baseline .val_acc :.1%}</div>"
        f"<div class='mlbl'>Baseline Val Acc</div></div>",
        unsafe_allow_html =True )
    with c2 :
        st .markdown (f"<div class='mcard'><div class='mbig'>{baseline .test_avg_acc :.1%}</div>"
        f"<div class='mlbl'>Baseline Test Avg</div></div>",
        unsafe_allow_html =True )
    with c3 :
        st .markdown (f"<div class='mcard'><div class='mbig' style='color:#f87171'>"
        f"{baseline .worst_group_acc :.1%}</div>"
        f"<div class='mlbl'>Baseline Worst-Group</div></div>",
        unsafe_allow_html =True )
    with c4 :
        if current is not None :
            sc =current .worst_group_acc 
            col ="#4ade80"if sc >=target else "#facc15"
            st .markdown (f"<div class='mcard'><div class='mbig' style='color:{col }'>"
            f"{sc :.1%}</div><div class='mlbl'>Your Worst-Group</div></div>",
            unsafe_allow_html =True )
        else :
            st .markdown ("<div class='mcard'><div class='mbig' style='color:#94a3b8'>--</div>"
            "<div class='mlbl'>Your Worst-Group</div></div>",
            unsafe_allow_html =True )

    st .markdown ("")


    tab_diag ,tab_train ,tab_reveal =st .tabs (["Diagnose","Train","Reveal"])


    with tab_diag :
        result =current if current is not None else baseline 
        which ="Your model"if current is not None else "Baseline (train a model to compare)"
        st .caption (f"Showing: {which }")

        dc1 ,dc2 =st .columns ([3 ,2 ])
        with dc1 :
            st .markdown ("#### Accuracy Comparison")
            st .image (fig_comparison (baseline ,current ),use_container_width =True )
        with dc2 :
            st .markdown ("#### Confusion Matrix")
            st .image (fig_cm (result .confusion_matrix ),use_container_width =True )

        st .markdown ("#### Per-Group Accuracy")
        st .image (fig_group_bars (result .group_accs ,target ),use_container_width =True )

        rows =[]
        for g in range (4 ):
            acc =result .group_accs .get (g ,float ("nan"))
            rows .append ({
            "Group":GROUP_LABELS [g ],
            "Accuracy":f"{acc :.1%}"if not np .isnan (acc )else "n/a",
            "Status":("OK"if (not np .isnan (acc )and acc >=target )
            else ("WARN"if (not np .isnan (acc )and acc >=target *0.85 )
            else "FAIL")),
            })
        st .table (rows )

        if current is not None :
            st .markdown ("#### Training History")
            st .image (fig_history (current .history ),use_container_width =True )

        st .markdown ("---")
        st .markdown ("#### Grad-CAM Explorer")
        gc1 ,gc2 =st .columns ([1 ,2 ])
        with gc1 :
            cam_split =st .selectbox ("Split",["test","val","train"],key ="cam_split")
            ds_map ={"test":test_ds ,"val":val_ds ,"train":train_ds }
            cam_idx =st .slider ("Sample index",0 ,len (ds_map [cam_split ])-1 ,0 ,
            key ="cam_idx")
            run_cam =st .button ("Explain this sample",type ="primary")

        with gc2 :
            if run_cam :
                chosen_ds =ds_map [cam_split ]
                norm =get_normalize_transform ()
                raw_img ,true_lbl ,group_id =chosen_ds [cam_idx ]
                img_norm =norm (raw_img .unsqueeze (0 ))
                with st .spinner ("Running Grad-CAM ..."):
                    try :
                        orig_pil ,heat_pil ,ov_pil ,pred_cls ,conf =make_gradcam_panel (result .model ,img_norm )
                        s1 ,s2 ,s3 =st .columns (3 )
                        s1 .image (orig_pil ,caption ="Original",use_container_width =True )
                        s2 .image (heat_pil ,caption ="Heatmap",use_container_width =True )
                        s3 .image (ov_pil ,caption ="Overlay",use_container_width =True )
                        correct ="Correct"if pred_cls ==true_lbl else "Wrong"
                        st .info (f"True: {CLASS_NAMES [true_lbl ]} | "
                        f"Pred: {CLASS_NAMES [pred_cls ]} | "
                        f"Conf: {conf :.1%} | Group: {group_id } | {correct }")
                    except Exception as e :
                        st .error (f"Grad-CAM error: {e }")
            else :
                st .caption ("Select a sample and click Explain.")

        st .markdown ("---")
        st .markdown ("#### Most Confidently Wrong Predictions")
        if result .suspicious :
            n_show =min (10 ,len (result .suspicious ))
            cols =st .columns (5 )
            for i ,(idx ,true_lbl ,pred_lbl ,conf )in enumerate (result .suspicious [:n_show ]):
                with cols [i %5 ]:
                    if idx <len (test_ds ):
                        raw_img =test_ds [idx ][0 ]
                        pil =_tensor_to_pil (raw_img ).resize ((96 ,96 ))
                        st .image (pil ,
                        caption =f"T:{CLASS_NAMES [true_lbl ]} "
                        f"P:{CLASS_NAMES [pred_lbl ]} ({conf :.0%})",
                        use_container_width =True )
        else :
            st .caption ("No wrong predictions.")


    with tab_train :
        st .markdown ("### Player Toolbox")
        tc1 ,tc2 ,tc3 =st .columns (3 )

        with tc1 :
            st .markdown ("**Fine-tuning Depth**")
            ft_mode =st .radio (
            "Trainable layers",
            ["head_only","last_block","full"],
            format_func ={"head_only":"Head only (fastest)",
            "last_block":"Last block + head",
            "full":"Full fine-tune"}.get ,
            index =0 ,key ="ft_mode",
            )
            st .markdown ("**Optimiser**")
            opt_name =st .radio ("Optimizer",["adam","sgd"],
            format_func ={"adam":"Adam",
            "sgd":"SGD + momentum"}.get ,
            key ="opt_name")
            lr =st .select_slider (
            "Learning rate",
            options =[1e-5 ,5e-5 ,1e-4 ,5e-4 ,1e-3 ,3e-3 ,1e-2 ,3e-2 ,1e-1 ],
            value =1e-3 ,key ="lr",
            )
            epochs =st .slider ("Epochs (capped at 4 for CPU budget)",1 ,4 ,3 ,key ="epochs")

        with tc2 :
            st .markdown ("**Loss Function**")
            loss_name =st .radio (
            "Loss",
            ["cross_entropy","class_weighted_ce","group_reweighted_ce"],
            format_func ={
            "cross_entropy":"Cross-Entropy",
            "class_weighted_ce":"Class-Weighted CE",
            "group_reweighted_ce":"Group-Reweighted CE (recommended)",
            }.get ,
            index =0 ,key ="loss_name",
            )
            st .caption (
            "Group-reweighted CE uses group labels (shortcut present/absent) "
            "to up-weight hard samples, forcing the model to learn real features."
            )

        with tc3 :
            st .markdown ("**Augmentations**")
            aug_hflip =st .checkbox ("Horizontal Flip",key ="aug_hflip")
            aug_cj =st .checkbox ("ColorJitter",key ="aug_cj")
            aug_gray =st .checkbox ("RandomGrayscale",key ="aug_gray")
            aug_crop =st .checkbox ("RandomResizedCrop",key ="aug_crop")
            aug_erase =st .checkbox ("RandomErasing / Cutout",key ="aug_erase")
            aug_noise =st .checkbox ("Gaussian Noise",key ="aug_noise")
            aug_str =st .slider ("Augmentation strength",0.1 ,1.0 ,0.5 ,
            step =0.05 ,key ="aug_str")

        train_clicked =st .button ("Train My Model",type ="primary",
        use_container_width =True )

        if train_clicked :
            cfg =TrainConfig (
            finetune_mode =ft_mode ,
            optimizer =opt_name ,
            learning_rate =float (lr ),
            epochs =int (epochs ),
            loss_fn_name =loss_name ,
            use_hflip =aug_hflip ,
            use_color_jitter =aug_cj ,
            use_random_gray =aug_gray ,
            use_random_crop =aug_crop ,
            use_random_erasing =aug_erase ,
            use_gaussian_noise =aug_noise ,
            aug_strength =float (aug_str ),
            )

            prog_bar =st .progress (0.0 ,text ="Starting ...")

            def _cb (frac ,msg ):
                prog_bar .progress (min (frac ,1.0 ),text =msg )

            with st .spinner ("Training ..."):
                result =run_training (cfg ,train_ds ,val_ds ,test_ds ,
                progress_cb =_cb )

            prog_bar .empty ()
            st .session_state .current_result =result 

            entry =(level ,result .worst_group_acc ,cfg_summary (cfg ))
            st .session_state .leaderboard .append (entry )
            st .session_state .leaderboard .sort (key =lambda x :-x [1 ])

            if result .worst_group_acc >=LEVEL_TARGETS [level ]:
                st .session_state .unlocked .add (level )
                st .balloons ()
                st .success (f"Level {level } Complete! "
                f"Worst-group acc = {result .worst_group_acc :.1%}")
            else :
                st .info (f"Worst-group acc = {result .worst_group_acc :.1%} "
                f"(target {LEVEL_TARGETS [level ]:.0%}) -- keep going!")
            st .rerun ()


    with tab_reveal :
        st .markdown ("### What Was Planted?")
        name ,desc =LEVEL_SHORTCUTS [level ]
        st .markdown (f"**{name }**")
        st .markdown (desc )

        st .markdown ("---")
        st .markdown ("### Why Average Accuracy Is Misleading")
        bl =baseline 
        st .markdown (
        f"The baseline achieves **{bl .val_acc :.1%}** validation accuracy "
        f"because the shortcut is correlated with the label **95%** of the time in train/val.  \n"
        f"On the test set (shortcuts randomised 50/50), samples that lack the expected "
        f"shortcut fool the model -- worst-group accuracy collapses to "
        f"**{bl .worst_group_acc :.1%}**.  \n"
        f"This is why we report **worst-group accuracy** as the score, not the average."
        )

        st .markdown ("---")
        st .markdown ("### Recommended Fix")
        st .info (LEVEL_FIXES [level ])

        st .markdown ("---")
        st .markdown ("### Course Concepts Demonstrated")
        st .table ([
        {"Concept":"Transfer Learning",
        "How it appears":"ResNet18 (ImageNet weights) -- backbone reused, only head replaced by default."},
        {"Concept":"Data Augmentation",
        "How it appears":"ColorJitter / RandomErasing / RandomGrayscale destroy the shortcut signal."},
        {"Concept":"Loss Functions",
        "How it appears":"Group-reweighted CE directly penalises failures on shortcut-absent groups."},
        {"Concept":"Metrics",
        "How it appears":f"Val acc {bl .val_acc :.1%} looks fine; worst-group {bl .worst_group_acc :.1%} exposes the failure."},
        ])


if __name__ =="__main__":
    main ()
