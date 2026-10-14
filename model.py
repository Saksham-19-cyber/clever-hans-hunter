"""
model.py  –  ResNet18 wrapper with configurable fine-tuning depth.

Fine-tuning depth options
  "head_only"      – freeze everything except the final FC layer
  "last_block"     – unfreeze layer4 + FC
  "full"           – unfreeze all parameters
"""

from pathlib import Path 
import torch 
import torch .nn as nn 
from torchvision .models import resnet18 ,ResNet18_Weights 

FINETUNE_MODES =["head_only","last_block","full"]


LOCAL_WEIGHTS_PATH =Path (__file__ ).parent /"weights"/"resnet18_imagenet.pth"


def build_model (
num_classes :int =2 ,
finetune_mode :str ="head_only",
pretrained :bool =True ,
)->nn .Module :
    if pretrained :
        if LOCAL_WEIGHTS_PATH .is_file ():
            model =resnet18 (weights =None )
            state_dict =torch .load (LOCAL_WEIGHTS_PATH ,map_location ="cpu",weights_only =True )
            model .load_state_dict (state_dict )
        else :
            weights =ResNet18_Weights .IMAGENET1K_V1 
            model =resnet18 (weights =weights )
    else :
        model =resnet18 (weights =None )

    in_features =model .fc .in_features 
    model .fc =nn .Linear (in_features ,num_classes )

    set_finetune_mode (model ,finetune_mode )
    return model 


def set_finetune_mode (model :nn .Module ,mode :str )->None :
    """Freeze / unfreeze parameter groups according to *mode*."""
    if mode not in FINETUNE_MODES :
        raise ValueError (f"Unknown finetune_mode '{mode }'. Choose from {FINETUNE_MODES }")

    if mode =="head_only":

        for name ,param in model .named_parameters ():
            param .requires_grad =(name .startswith ("fc."))

    elif mode =="last_block":

        for name ,param in model .named_parameters ():
            param .requires_grad =(
            name .startswith ("layer4.")or name .startswith ("fc.")
            )

    elif mode =="full":

        for param in model .parameters ():
            param .requires_grad =True 


def count_trainable (model :nn .Module )->int :
    return sum (p .numel ()for p in model .parameters ()if p .requires_grad )


def get_last_conv_layer (model :nn .Module )->nn .Module :
    """Return the last convolutional layer (used by Grad-CAM hook)."""
    return model .layer4 [-1 ]


if __name__ =="__main__":
    for mode in FINETUNE_MODES :
        m =build_model (finetune_mode =mode )
        print (f"{mode :15s} → {count_trainable (m ):>9,} trainable params")
