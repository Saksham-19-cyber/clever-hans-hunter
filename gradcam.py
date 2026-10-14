"""
gradcam.py  –  Manual Grad-CAM implementation using PyTorch hooks.

No external Grad-CAM library is used.  Only torchvision + PIL + numpy.

API
----
    cam = GradCAM(model)
    heatmap, overlay, pred_cls, confidence = cam.explain(img_tensor, target_cls=None)
    cam.remove_hooks()

Returns
-------
heatmap  : H×W numpy float32 array in [0,1]
overlay  : PIL Image with heatmap blended onto original
pred_cls : int (0 or 1)
confidence : float in [0,1]
"""

from __future__ import annotations 
from typing import Optional ,Tuple 
import numpy as np 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from PIL import Image 
import cv2 


class GradCAM :
    """
    Gradient-weighted Class Activation Mapping.

    Hooks are attached to the *last* convolutional block of ResNet18.
    """

    def __init__ (self ,model :nn .Module ):
        self .model =model 
        self .model .eval ()

        self ._gradients :Optional [torch .Tensor ]=None 
        self ._activations :Optional [torch .Tensor ]=None 


        self ._target_layer =model .layer4 [-1 ]
        self ._register_hooks ()


    def _register_hooks (self ):
        def fwd_hook (module ,input ,output ):
            self ._activations =output .detach ()

        def bwd_hook (module ,grad_in ,grad_out ):
            self ._gradients =grad_out [0 ].detach ()

        self ._fwd_handle =self ._target_layer .register_forward_hook (fwd_hook )
        self ._bwd_handle =self ._target_layer .register_full_backward_hook (bwd_hook )

    def remove_hooks (self ):
        self ._fwd_handle .remove ()
        self ._bwd_handle .remove ()


    def explain (
    self ,
    img_tensor :torch .Tensor ,
    target_cls :Optional [int ]=None ,
    ):
        """
        Parameters
        ----------
        img_tensor  : 1×C×H×W float tensor (already normalised)
        target_cls  : class index to explain; if None uses argmax

        Returns
        -------
        heatmap     : H×W numpy float32 [0,1]
        overlay_pil : PIL Image (RGB)
        pred_cls    : int
        confidence  : float
        """
        self .model .eval ()


        self .model .zero_grad ()
        x_req =img_tensor .clone ().detach ().requires_grad_ (True )
        logits =self .model (x_req )

        probs =F .softmax (logits ,dim =1 )
        pred_cls =int (logits .argmax (dim =1 ).item ())
        confidence =float (probs [0 ,pred_cls ].item ())

        if target_cls is None :
            target_cls =pred_cls 


        self .model .zero_grad ()
        score =logits [0 ,target_cls ]
        score .backward ()


        weights =self ._gradients .mean (dim =(2 ,3 ),keepdim =True )
        cam =(weights *self ._activations ).sum (dim =1 ,keepdim =True )
        cam =F .relu (cam )
        cam =cam .squeeze ().cpu ().numpy ()


        cam_min ,cam_max =cam .min (),cam .max ()
        if cam_max -cam_min >1e-8 :
            cam =(cam -cam_min )/(cam_max -cam_min )
        else :
            cam =np .zeros_like (cam )


        h =img_tensor .shape [2 ]
        w =img_tensor .shape [3 ]

        heatmap =_array_to_heatmap (cam ,h ,w )
        original =_tensor_to_pil (img_tensor [0 ])

        overlay_pil =_blend (original ,heatmap ,alpha =0.45 )

        return cam ,overlay_pil ,pred_cls ,confidence 


_MEAN =np .array ([0.485 ,0.456 ,0.406 ],dtype =np .float32 )
_STD =np .array ([0.229 ,0.224 ,0.225 ],dtype =np .float32 )


def _tensor_to_pil (tensor :torch .Tensor )->Image .Image :
    """
    Convert a C×H×W normalised float tensor back to a PIL RGB image.
    """
    arr =tensor .cpu ().numpy ().transpose (1 ,2 ,0 )
    arr =arr *_STD +_MEAN 
    arr =np .clip (arr *255 ,0 ,255 ).astype (np .uint8 )
    return Image .fromarray (arr )


def _array_to_heatmap (cam :np .ndarray ,h :int ,w :int )->np .ndarray :
    """
    Resize a 2-D CAM (float [0,1]) to H×W and apply a colour map.
    Returns H×W×3 uint8.
    """
    cam_uint8 =(cam *255 ).astype (np .uint8 )
    cam_resized =cv2 .resize (cam_uint8 ,(w ,h ),interpolation =cv2 .INTER_LINEAR )
    heatmap_bgr =cv2 .applyColorMap (cam_resized ,cv2 .COLORMAP_JET )
    heatmap_rgb =cv2 .cvtColor (heatmap_bgr ,cv2 .COLOR_BGR2RGB )
    return heatmap_rgb 


def _blend (original :Image .Image ,heatmap_arr :np .ndarray ,alpha :float )->Image .Image :
    """Alpha-blend original PIL image with a colour heatmap array."""
    orig_arr =np .array (original ,dtype =np .float32 )
    heat_arr =heatmap_arr .astype (np .float32 )
    blended =(1 -alpha )*orig_arr +alpha *heat_arr 
    return Image .fromarray (np .clip (blended ,0 ,255 ).astype (np .uint8 ))


def make_gradcam_panel (
model :nn .Module ,
img_tensor :torch .Tensor ,
target_cls :Optional [int ]=None ,
)->tuple :
    """
    Convenience wrapper: creates a fresh GradCAM instance, explains one image,
    cleans up hooks and returns the four panel images.

    Returns
    -------
    original_pil, heatmap_pil, overlay_pil, pred_cls, confidence
    """
    gcam =GradCAM (model )
    try :
        cam_arr ,overlay_pil ,pred_cls ,confidence =gcam .explain (img_tensor ,target_cls )
    finally :
        gcam .remove_hooks ()

    original_pil =_tensor_to_pil (img_tensor [0 ])


    h ,w =img_tensor .shape [2 ],img_tensor .shape [3 ]
    heatmap_rgb =_array_to_heatmap (cam_arr ,h ,w )
    heatmap_pil =Image .fromarray (heatmap_rgb )

    return original_pil ,heatmap_pil ,overlay_pil ,pred_cls ,confidence 
