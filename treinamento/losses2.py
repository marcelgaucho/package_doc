# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:56:29 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras import backend as K

# %%

def dice_loss(y_true, y_pred):
    """
    Computes the Dice Loss for the foreground class (roads), ignoring background.
    Assumes channels-last format (Batch, Height, Width, Channels) and that 
    y_pred has already been passed through a Sigmoid/Softmax activation.
    """
    # Smooth constant added to numerator and denominator to 
    # stabilize training and avoid zero division
    smooth = 1e-7
    
    # 1. Ensure tensors in float32 format
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 2. Isolate channel 1 (roads) due to massive background (class imbalance)
    # Assumes a 2-channel input where channel 0 is background.
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]
    
    # 3. Sum over spatial dimensions (Height, Width) -> axes 1 and 2
    # This keeps the Batch (axis 0) and Class (axis 3) dimensions intact
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    true_sum = tf.reduce_sum(y_true, axis=(1, 2))
    pred_sum = tf.reduce_sum(y_pred, axis=(1, 2))
    
    # 5. Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
    
    # 6. Averages dice over the batch, then returns (1.0 - Dice) for minimization 
    return 1.0 - tf.reduce_mean(dice)
        


# %% --- OPCIONAL: COMBO LOSS (CCE + DICE) ATUALIZADA ---

def combo_loss(smooth=1e-6, dice_weight=0.5):
    cce = tf.keras.losses.CategoricalCrossentropy()
    d_loss = dice_loss(smooth)
    
    def loss(y_true, y_pred):
        cce_loss = cce(y_true, y_pred)
        dice = d_loss(y_true, y_pred)
        return (1.0 - dice_weight) * cce_loss + dice_weight * dice
        
    return loss
