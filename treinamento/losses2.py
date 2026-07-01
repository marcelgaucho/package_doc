# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:56:29 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras import backend as K

# %% Dice Loss

def dice_loss(y_true, y_pred):
    """
    Computes the Dice Loss for the foreground class (roads), ignoring background.
    Assumes channels-last format (Batch, Height, Width, Channels).
    """
    # Smooth constant added to numerator and denominator to 
    # stabilize training and avoid zero division
    smooth = 1e-7
    
    # 1. Ensure tensors in float32 format
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 2. Uncomment to Isolate channel 1 (roads) due to massive background (class imbalance)
    # Assumes a 2-channel input where channel 0 is background.
    # y_true = y_true[..., 1:]
    # y_pred = y_pred[..., 1:]
    
    # 3. Sum over spatial dimensions (Height, Width) -> axes 1 and 2
    # This keeps the Batch (axis 0) and Class (axis 3) dimensions intact
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    true_sum = tf.reduce_sum(y_true, axis=(1, 2))
    pred_sum = tf.reduce_sum(y_pred, axis=(1, 2))
    
    # 5. Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
    
    # 6. Averages dice over the batch, then returns (1.0 - Dice) for minimization 
    return 1.0 - tf.reduce_mean(dice)
        


# %% Combo Loss: CCE + Dice

def combo_loss(weight_cce=1., weight_dice=1.):
    def loss(y_true, y_pred):
        # 1. Ensure tensors in float32 format
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 2. Calculate the standard cross-entropy (no reduction)
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss_cce = tf.reduce_mean(loss_cce)
        
        # 3. Calculate the dice loss
        loss_dice = dice_loss(y_true, y_pred)
        
        # 4. Return the combined loss
        return (weight_cce * loss_cce) + (weight_dice * loss_dice)
     
    # Return loss
    return loss


