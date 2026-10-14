"""
train.py  –  Training loop, augmentation pipeline, loss variants, and metrics.

Public API
----------
    cfg  = TrainConfig(...)
    result = run_training(cfg, train_ds, val_ds, test_ds, progress_cb=None)

result is a TrainResult dataclass with accuracy tables, confusion matrix,
worst-group accuracy, and the trained model.
"""

from __future__ import annotations 

import time 
from dataclasses import dataclass ,field 
from typing import Callable ,Dict ,List ,Optional ,Tuple 

import numpy as np 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
import torchvision .transforms as T 
from torch .utils .data import DataLoader 

from data import ShortcutDataset ,make_dataloaders ,CLASS_NAMES 
from model import build_model ,set_finetune_mode 


def build_augmentation_transform (
use_color_jitter :bool =False ,
use_random_gray :bool =False ,
use_random_crop :bool =False ,
use_random_erasing :bool =False ,
use_hflip :bool =False ,
use_gaussian_noise :bool =False ,
aug_strength :float =0.5 ,
image_size :int =64 ,
)->Optional [nn .Module ]:
    """
    Build a batched augmentation transform (operates on B×C×H×W tensors).
    Returns None if nothing is selected.
    """
    ops :List [nn .Module ]=[]

    if use_hflip :
        ops .append (T .RandomHorizontalFlip (p =0.5 ))

    if use_random_crop :
        scale_lo =max (0.5 ,1.0 -0.4 *aug_strength )
        ops .append (T .RandomResizedCrop (
        image_size ,
        scale =(scale_lo ,1.0 ),
        ratio =(0.8 ,1.2 ),
        antialias =True ,
        ))

    if use_color_jitter :
        b =0.3 *aug_strength 
        c =0.3 *aug_strength 
        s =0.3 *aug_strength 
        h =0.1 *aug_strength 
        ops .append (T .ColorJitter (brightness =b ,contrast =c ,saturation =s ,hue =h ))

    if use_random_gray :
        ops .append (T .RandomGrayscale (p =0.2 *aug_strength +0.05 ))

    if use_random_erasing :
        ops .append (T .RandomErasing (
        p =0.5 ,
        scale =(0.02 ,0.2 *aug_strength +0.05 ),
        ratio =(0.3 ,3.3 ),
        value =0 ,
        ))

    if not ops and not use_gaussian_noise :
        return None 

    class _AugPipeline (nn .Module ):
        def __init__ (self ,transform_ops ,add_noise ,noise_strength ):
            super ().__init__ ()
            self .pipe =T .Compose (transform_ops )if transform_ops else None 
            self .add_noise =add_noise 
            self .noise_strength =noise_strength 

        def forward (self ,x :torch .Tensor )->torch .Tensor :


            if self .pipe is not None :
                x =torch .stack ([self .pipe (img )for img in x ])
            if self .add_noise :
                noise =torch .randn_like (x )*0.05 *self .noise_strength 
                x =torch .clamp (x +noise ,0.0 ,1.0 )
            return x 

    return _AugPipeline (ops ,use_gaussian_noise ,aug_strength )


def cross_entropy_loss (logits ,labels ,groups =None ,group_weights =None ):
    """Plain cross-entropy loss (ignores groups)."""
    return F .cross_entropy (logits ,labels )


def class_weighted_ce_loss (logits ,labels ,groups =None ,group_weights =None ):
    """
    Class-weighted cross-entropy.
    Weights are inversely proportional to class frequency in the batch.
    """
    counts =torch .bincount (labels ,minlength =2 ).float ().clamp (min =1 )
    weights =counts .sum ()/(2 *counts )
    return F .cross_entropy (logits ,labels ,weight =weights .to (logits .device ))


def group_reweighted_ce_loss (logits ,labels ,groups ,group_weights ):
    """
    Group-reweighted cross-entropy (group-reweighted ERM).

    group_weights : 4-element tensor, one weight per group id.
    Each sample loss is multiplied by the weight of its group, so
    under-represented / worst-performing groups are up-weighted.

    This is the key pedagogical mechanic: the loss now explicitly
    penalises errors on groups where the shortcut is *absent*.
    """
    per_sample =F .cross_entropy (logits ,labels ,reduction ="none")
    w =group_weights [groups ].to (per_sample .device )
    return (per_sample *w ).mean ()


LOSS_FN_MAP ={
"cross_entropy":cross_entropy_loss ,
"class_weighted_ce":class_weighted_ce_loss ,
"group_reweighted_ce":group_reweighted_ce_loss ,
}


@dataclass 
class TrainConfig :

    finetune_mode :str ="head_only"
    pretrained :bool =True 


    optimizer :str ="adam"
    learning_rate :float =1e-3 
    epochs :int =3 
    batch_size :int =64 


    loss_fn_name :str ="cross_entropy"


    use_color_jitter :bool =False 
    use_random_gray :bool =False 
    use_random_crop :bool =False 
    use_random_erasing :bool =False 
    use_hflip :bool =False 
    use_gaussian_noise :bool =False 
    aug_strength :float =0.5 
    group_weights_power :float =1.0 


    image_size :int =64 
    seed :int =42 
    device :str ="cpu"


@dataclass 
class TrainResult :
    val_acc :float 
    test_avg_acc :float 
    worst_group_acc :float 
    group_accs :Dict [int ,float ]
    confusion_matrix :np .ndarray 
    history :List [Dict ]
    model :nn .Module 

    suspicious :List [Tuple [int ,int ,int ,float ]]=field (default_factory =list )


def _compute_group_accs (
all_preds :List [int ],
all_labels :List [int ],
all_groups :List [int ],
)->Dict [int ,float ]:
    """Per-group accuracy dict (4 groups)."""
    group_correct ={g :0 for g in range (4 )}
    group_total ={g :0 for g in range (4 )}
    for p ,l ,g in zip (all_preds ,all_labels ,all_groups ):
        group_total [g ]+=1 
        group_correct [g ]+=int (p ==l )
    return {
    g :(group_correct [g ]/group_total [g ]if group_total [g ]>0 else float ("nan"))
    for g in range (4 )
    }


def _confusion_matrix (preds ,labels ,n =2 )->np .ndarray :
    cm =np .zeros ((n ,n ),dtype =int )
    for p ,l in zip (preds ,labels ):
        cm [l ][p ]+=1 
    return cm 


def evaluate (
model :nn .Module ,
loader :DataLoader ,
device :str ="cpu",
)->Tuple [float ,List [int ],List [int ],List [int ],List [float ]]:
    """
    Run model in eval mode over *loader*.

    Returns
    -------
    accuracy, all_preds, all_labels, all_groups, all_confidences
    """
    model .eval ()
    model .to (device )

    all_preds ,all_labels ,all_groups ,all_confs =[],[],[],[]
    correct =total =0 

    with torch .no_grad ():
        for imgs ,labels ,groups in loader :
            imgs =imgs .to (device )
            labels =labels .to (device )

            logits =model (imgs )
            probs =F .softmax (logits ,dim =1 )
            preds =logits .argmax (dim =1 )

            correct +=(preds ==labels ).sum ().item ()
            total +=len (labels )

            all_preds .extend (preds .cpu ().tolist ())
            all_labels .extend (labels .cpu ().tolist ())
            all_groups .extend (groups .tolist ())
            all_confs .extend (probs .max (dim =1 ).values .cpu ().tolist ())

    acc =correct /total if total >0 else 0.0 
    return acc ,all_preds ,all_labels ,all_groups ,all_confs 


def run_training (
cfg :TrainConfig ,
train_ds :ShortcutDataset ,
val_ds :ShortcutDataset ,
test_ds :ShortcutDataset ,
progress_cb :Optional [Callable [[float ,str ],None ]]=None ,
base_model :Optional [nn .Module ]=None ,
)->TrainResult :
    """
    Train (or fine-tune) a model according to *cfg*.

    progress_cb(fraction, message) is called after each batch/epoch so that
    the UI can update a progress bar without blocking.

    If *base_model* is provided, its weights are used as initialisation
    (allows re-training on top of the baseline).
    """
    torch .manual_seed (cfg .seed )
    np .random .seed (cfg .seed )

    device =cfg .device 


    if base_model is not None :
        import copy 
        model =copy .deepcopy (base_model )
        set_finetune_mode (model ,cfg .finetune_mode )
    else :
        model =build_model (
        finetune_mode =cfg .finetune_mode ,
        pretrained =cfg .pretrained ,
        )
    model .to (device )


    aug =build_augmentation_transform (
    use_color_jitter =cfg .use_color_jitter ,
    use_random_gray =cfg .use_random_gray ,
    use_random_crop =cfg .use_random_crop ,
    use_random_erasing =cfg .use_random_erasing ,
    use_hflip =cfg .use_hflip ,
    use_gaussian_noise =cfg .use_gaussian_noise ,
    aug_strength =cfg .aug_strength ,
    image_size =cfg .image_size ,
    )


    train_loader ,val_loader ,test_loader =make_dataloaders (
    train_ds ,val_ds ,test_ds ,
    train_transform =aug ,
    batch_size =cfg .batch_size ,
    )


    group_counts =torch .zeros (4 )
    for g in train_ds .group_ids :
        group_counts [g ]+=1 
    group_counts =group_counts .clamp (min =1 )
    group_weights =(group_counts .sum ()/(4 *group_counts ))**cfg .group_weights_power 



    trainable =[p for p in model .parameters ()if p .requires_grad ]
    if cfg .optimizer =="sgd":
        optimizer =torch .optim .SGD (trainable ,lr =cfg .learning_rate ,momentum =0.9 ,weight_decay =1e-4 )
    else :
        optimizer =torch .optim .Adam (trainable ,lr =cfg .learning_rate ,weight_decay =1e-4 )

    scheduler =torch .optim .lr_scheduler .CosineAnnealingLR (
    optimizer ,T_max =cfg .epochs *len (train_loader )
    )

    loss_fn =LOSS_FN_MAP [cfg .loss_fn_name ]


    history =[]
    total_steps =cfg .epochs *len (train_loader )
    step =0 
    t0 =time .time ()

    for epoch in range (cfg .epochs ):
        model .train ()
        epoch_loss =0.0 
        epoch_correct =epoch_total =0 

        for imgs ,labels ,groups in train_loader :
            imgs =imgs .to (device )
            labels =labels .to (device )
            groups =groups .to (device )

            optimizer .zero_grad ()
            logits =model (imgs )
            loss =loss_fn (logits ,labels ,groups ,group_weights .to (device ))
            loss .backward ()
            optimizer .step ()
            scheduler .step ()

            preds =logits .argmax (dim =1 )
            epoch_correct +=(preds ==labels ).sum ().item ()
            epoch_total +=len (labels )
            epoch_loss +=loss .item ()

            step +=1 
            if progress_cb and step %max (1 ,len (train_loader )//5 )==0 :
                frac =step /total_steps 
                elapsed =time .time ()-t0 
                msg =(
                f"Epoch {epoch +1 }/{cfg .epochs } | "
                f"loss={epoch_loss /step :.3f} | "
                f"elapsed={elapsed :.1f}s"
                )
                progress_cb (frac *0.8 ,msg )

        train_acc =epoch_correct /epoch_total 
        val_acc ,*_ =evaluate (model ,val_loader ,device )

        history .append ({
        "epoch":epoch +1 ,
        "train_acc":train_acc ,
        "val_acc":val_acc ,
        "train_loss":epoch_loss /len (train_loader ),
        })

        if progress_cb :
            progress_cb (
            0.8 +0.1 *(epoch +1 )/cfg .epochs ,
            f"Epoch {epoch +1 } done | val_acc={val_acc :.3f}",
            )


    if progress_cb :
        progress_cb (0.92 ,"Evaluating on test set …")

    test_acc ,test_preds ,test_labels ,test_groups ,test_confs =evaluate (
    model ,test_loader ,device 
    )

    group_accs =_compute_group_accs (test_preds ,test_labels ,test_groups )
    valid_accs =[v for v in group_accs .values ()if not np .isnan (v )]
    worst_group =min (valid_accs )if valid_accs else 0.0 
    cm =_confusion_matrix (test_preds ,test_labels )


    suspicious =[]
    for i ,(p ,l ,c )in enumerate (zip (test_preds ,test_labels ,test_confs )):
        if p !=l :
            suspicious .append ((i ,l ,p ,c ))
    suspicious .sort (key =lambda x :-x [3 ])

    if progress_cb :
        progress_cb (1.0 ,f"Done! Worst-group acc={worst_group :.3f}")

    final_val_acc =history [-1 ]["val_acc"]if history else 0.0 

    return TrainResult (
    val_acc =final_val_acc ,
    test_avg_acc =test_acc ,
    worst_group_acc =worst_group ,
    group_accs =group_accs ,
    confusion_matrix =cm ,
    history =history ,
    model =model ,
    suspicious =suspicious [:20 ],
    )


def train_baseline (
level :int ,
train_ds :ShortcutDataset ,
val_ds :ShortcutDataset ,
test_ds :ShortcutDataset ,
progress_cb =None ,
)->TrainResult :
    """
    Train the 'no fixes' baseline model:
      head_only, Adam, 3 epochs, plain CE, no augmentation.
    """
    cfg =TrainConfig (
    finetune_mode ="head_only",
    optimizer ="adam",
    learning_rate =1e-3 ,
    epochs =3 ,
    loss_fn_name ="cross_entropy",
    )
    return run_training (cfg ,train_ds ,val_ds ,test_ds ,progress_cb )
