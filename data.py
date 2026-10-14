"""
data.py  –  Dataset construction, shortcut injection, and group labelling.

Classes used: cat (CIFAR-10 label 3) vs dog (CIFAR-10 label 5).
Group id = 2 * true_class + shortcut_present  →  {0,1,2,3}
"""

import os 
import random 
from pathlib import Path 
from typing import Tuple ,List ,Union 

import numpy as np 
import torch 
from torch .utils .data import Dataset ,DataLoader ,Subset 
import torchvision 
import torchvision .transforms as T 
from PIL import Image ,ImageFilter 


CIFAR_CAT =3 
CIFAR_DOG =5 
CLASS_NAMES =["cat","dog"]


N_TRAIN =1500 
N_VAL =500 
N_TEST =500 

IMAGE_SIZE =64 
SHORTCUT_CORR_TRAIN =0.95 
SHORTCUT_CORR_TEST =0.50 

DATA_ROOT =Path ("./cifar_data")


def _blue_tint (img :torch .Tensor ,strength :float =0.25 )->torch .Tensor :
    """Add a blue tint (level-1 shortcut for class 0 / cat)."""
    img =img .clone ()
    img [2 ]=torch .clamp (img [2 ]+strength ,0.0 ,1.0 )
    img [0 ]=torch .clamp (img [0 ]-strength *0.3 ,0.0 ,1.0 )
    return img 


def _red_tint (img :torch .Tensor ,strength :float =0.25 )->torch .Tensor :
    """Add a red tint (level-1 shortcut for class 1 / dog)."""
    img =img .clone ()
    img [0 ]=torch .clamp (img [0 ]+strength ,0.0 ,1.0 )
    img [2 ]=torch .clamp (img [2 ]-strength *0.3 ,0.0 ,1.0 )
    return img 


def _color_patch (img :torch .Tensor ,cls :int ,patch_size :int =8 )->torch .Tensor :
    """Paste a small colored square in the top-left corner (level-2 shortcut)."""
    img =img .clone ()
    color =torch .tensor ([0.1 ,0.1 ,0.9 ])if cls ==0 else torch .tensor ([0.9 ,0.1 ,0.1 ])
    h ,w =img .shape [1 ],img .shape [2 ]
    img [:,:patch_size ,:patch_size ]=color [:,None ,None ].expand (3 ,patch_size ,patch_size )
    return img 


def _stripe_texture (img :torch .Tensor ,cls :int ,strength :float =0.08 )->torch .Tensor :
    """Overlay faint horizontal (class-0) or vertical (class-1) stripes (level-3)."""
    img =img .clone ()
    h ,w =img .shape [1 ],img .shape [2 ]
    if cls ==0 :

        stripe =torch .zeros (h ,w )
        stripe [::4 ,:]=1.0 
    else :

        stripe =torch .zeros (h ,w )
        stripe [:,::4 ]=1.0 
    img =img +strength *stripe .unsqueeze (0 )
    return torch .clamp (img ,0.0 ,1.0 )


def apply_shortcut (
img :torch .Tensor ,
cls :int ,
level :int ,
correlated :bool ,
)->Tuple [torch .Tensor ,bool ]:
    """
    Apply the level-specific shortcut.

    Parameters
    ----------
    img        : C×H×W float tensor in [0,1]
    cls        : 0 = cat, 1 = dog
    level      : 1, 2, or 3
    correlated : if True the shortcut matches the class label;
                 if False it is assigned to the *other* class.

    Returns
    -------
    modified image tensor, shortcut_present (bool)
    """

    signal_cls =cls if correlated else (1 -cls )

    if level ==1 :
        if signal_cls ==0 :
            img =_blue_tint (img )
        else :
            img =_red_tint (img )
        shortcut_present =(signal_cls ==cls )
    elif level ==2 :
        img =_color_patch (img ,signal_cls )
        shortcut_present =(signal_cls ==cls )
    elif level ==3 :
        img =_stripe_texture (img ,signal_cls )
        shortcut_present =(signal_cls ==cls )
    else :
        raise ValueError (f"Unknown level {level }")

    return img ,shortcut_present 


def make_group_id (cls :int ,shortcut_present :bool )->int :
    """
    Group id encoding:
      0 = cat  + shortcut absent/flipped
      1 = cat  + shortcut present (correlated)
      2 = dog  + shortcut absent/flipped
      3 = dog  + shortcut present (correlated)
    """
    return 2 *cls +int (shortcut_present )


class ShortcutDataset (Dataset ):
    """
    CIFAR-10 cat/dog subset with a planted shortcut.

    Items returned: (image_tensor, label, group_id)
    """

    def __init__ (
    self ,
    images :List [torch .Tensor ],
    labels :List [int ],
    group_ids :List [int ],
    transform =None ,
    ):
        self .images =images 
        self .labels =labels 
        self .group_ids =group_ids 
        self .transform =transform 

    def __len__ (self )->int :
        return len (self .images )

    def __getitem__ (self ,idx :int ):
        img =self .images [idx ]
        if self .transform is not None :
            img =self .transform (img )
        return img ,self .labels [idx ],self .group_ids [idx ]


def _load_raw_cifar (root :str |Path )->Tuple [np .ndarray ,np .ndarray ]:
    """Download CIFAR-10 and return all images + labels."""
    ds =torchvision .datasets .CIFAR10 (root =str (root ),train =True ,download =True )
    ds_test =torchvision .datasets .CIFAR10 (root =str (root ),train =False ,download =True )

    all_imgs =np .concatenate ([np .array (ds .data ),np .array (ds_test .data )],axis =0 )
    all_labels =np .concatenate ([np .array (ds .targets ),np .array (ds_test .targets )],axis =0 )
    return all_imgs ,all_labels 


def _resize_to_tensor (arr :np .ndarray ,size :int )->torch .Tensor :
    """HWC uint8 → C×H×W float32 [0,1] resized to size×size."""
    pil =Image .fromarray (arr ).resize ((size ,size ),Image .BILINEAR )
    return T .ToTensor ()(pil )


def build_datasets (
level :int ,
seed :int =42 ,
image_size :int =IMAGE_SIZE ,
data_root :Union [str ,Path ]=DATA_ROOT ,
)->Tuple [ShortcutDataset ,ShortcutDataset ,ShortcutDataset ]:
    """
    Build train / val / test ShortcutDatasets for the given shortcut level.

    Shortcut correlation:
      train / val  → 95 % correlated
      test         → 50 % (random, so effectively ~50/50 present vs absent)
    """
    rng =random .Random (seed )
    np .random .seed (seed )

    all_imgs ,all_labels =_load_raw_cifar (data_root )


    cat_idx =np .where (all_labels ==CIFAR_CAT )[0 ]
    dog_idx =np .where (all_labels ==CIFAR_DOG )[0 ]

    rng_np =np .random .RandomState (seed )
    rng_np .shuffle (cat_idx )
    rng_np .shuffle (dog_idx )


    n_tr ,n_va ,n_te =N_TRAIN ,N_VAL ,N_TEST 
    total =n_tr +n_va +n_te 

    cat_idx =cat_idx [:total ]
    dog_idx =dog_idx [:total ]

    splits ={
    "train":(slice (0 ,n_tr ),SHORTCUT_CORR_TRAIN ),
    "val":(slice (n_tr ,n_tr +n_va ),SHORTCUT_CORR_TRAIN ),
    "test":(slice (n_tr +n_va ,total ),SHORTCUT_CORR_TEST ),
    }

    datasets ={}
    for split_name ,(sl ,corr )in splits .items ():
        images ,labels ,group_ids =[],[],[]

        for cls_idx ,(cls_indices ,cls_label )in enumerate ([
        (cat_idx [sl ],0 ),
        (dog_idx [sl ],1 ),
        ]):
            for raw_idx in cls_indices :
                raw_img =all_imgs [raw_idx ]
                tensor =_resize_to_tensor (raw_img ,image_size )


                correlated =rng .random ()<corr 
                aug_tensor ,shortcut_present =apply_shortcut (
                tensor ,cls_label ,level ,correlated 
                )
                gid =make_group_id (cls_label ,shortcut_present )

                images .append (aug_tensor )
                labels .append (cls_label )
                group_ids .append (gid )

        datasets [split_name ]=ShortcutDataset (images ,labels ,group_ids )

    return datasets ["train"],datasets ["val"],datasets ["test"]


IMAGENET_MEAN =[0.485 ,0.456 ,0.406 ]
IMAGENET_STD =[0.229 ,0.224 ,0.225 ]

def get_normalize_transform ():
    return T .Normalize (mean =IMAGENET_MEAN ,std =IMAGENET_STD )


def make_dataloaders (
train_ds :ShortcutDataset ,
val_ds :ShortcutDataset ,
test_ds :ShortcutDataset ,
train_transform =None ,
batch_size :int =64 ,
num_workers :int =0 ,
)->Tuple [DataLoader ,DataLoader ,DataLoader ]:
    """
    Wrap datasets in DataLoaders.
    train_transform is applied inside the loader (augmentation pipeline).
    val / test use only normalisation.
    """
    normalize =get_normalize_transform ()

    def collate_with_train_aug (batch ):
        imgs ,lbls ,gids =zip (*batch )
        imgs =torch .stack (imgs )
        if train_transform is not None :
            imgs =train_transform (imgs )
        imgs =normalize (imgs )
        return imgs ,torch .tensor (lbls ,dtype =torch .long ),torch .tensor (gids ,dtype =torch .long )

    def collate_eval (batch ):
        imgs ,lbls ,gids =zip (*batch )
        imgs =normalize (torch .stack (imgs ))
        return imgs ,torch .tensor (lbls ,dtype =torch .long ),torch .tensor (gids ,dtype =torch .long )

    train_loader =DataLoader (
    train_ds ,batch_size =batch_size ,shuffle =True ,
    num_workers =num_workers ,collate_fn =collate_with_train_aug ,
    )
    val_loader =DataLoader (
    val_ds ,batch_size =batch_size ,shuffle =False ,
    num_workers =num_workers ,collate_fn =collate_eval ,
    )
    test_loader =DataLoader (
    test_ds ,batch_size =batch_size ,shuffle =False ,
    num_workers =num_workers ,collate_fn =collate_eval ,
    )
    return train_loader ,val_loader ,test_loader 


def shortcut_correlation (ds :ShortcutDataset )->float :
    """
    Fraction of samples where the shortcut is present and the label matches
    (i.e. the shortcut is correlated with the true class).
    Groups 1 and 3 are "shortcut present & correlated".
    """
    n_corr =sum (1 for g in ds .group_ids if g in (1 ,3 ))
    return n_corr /len (ds )


if __name__ =="__main__":

    for level in [1 ,2 ,3 ]:
        tr ,va ,te =build_datasets (level )
        tr_corr =shortcut_correlation (tr )
        te_corr =shortcut_correlation (te )
        print (
        f"Level {level } | "
        f"train shortcut-corr={tr_corr :.3f} (target ~0.95) | "
        f"test  shortcut-corr={te_corr :.3f} (target ~0.50)"
        )
        print (f"  train groups: { {g :tr .group_ids .count (g )for g in range (4 )} }")
        print (f"  test  groups: { {g :te .group_ids .count (g )for g in range (4 )} }")
