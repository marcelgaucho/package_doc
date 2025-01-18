# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:44:18 2024

@author: Marcel
"""

# Build a specified model, among the available ones

# %% Imports

from .unet import build_model_unet
from .resunet import build_model_resunet
from .unetr_2d import build_unetr_2d

# Could be SegFormer_B0, SegFormer_B1, SegFormer_B2,
# SegFormer_B3, SegFormer_B4, SegFormer_B5, but here
# we are using only SegFormer_B5, the most powerful, 
# but with more parameters

# Using Keras 2. If it is Keras 3, uncomment the following line and comment
# the line importing the model class in Keras 2
# from .segformer_tf_k3.models import SegFormer_B5
from .segformer_tf_k2.models import SegFormer_B5

# %% Build Model Function
        
def build_model(input_shape, n_classes, model_type='unet', config_dict=None):
    if model_type == 'unet':
        return build_model_unet(input_shape, n_classes)
    
    elif model_type == 'resunet':
        return build_model_resunet(input_shape, n_classes)
    
    elif model_type == 'unetr':
        return build_unetr_2d(input_shape, n_classes, config_dict)
    
    elif model_type == 'segformer_b5':
        return SegFormer_B5(input_shape, n_classes)  
    
    else:
        raise Exception("Model options are 'unet' and 'resunet' and 'unetr' and 'segformer_b5'")