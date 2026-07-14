# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:39:45 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf

# %% Dice Loss

def dice_loss(y_true, y_pred):
    """
    Computes the Dice Loss. Assumes channels-last format (Batch, Height, Width, Channels) and one-hot encoding of Y.
    """
    # Smooth constant added to numerator and denominator to 
    # stabilize training and avoid zero division
    smooth = 1e-5
    
    # 1. Ensure tensors in float32 format
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 2. Uncomment to Isolate channel 1 (roads) due to massive background (class imbalance)
    # Assumes a 2-channel input where channel 0 is background.
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]
    
    # 3. Sum over batch and spatial dimensions (Batch, Height, Width) -> axes 0, 1 and 2
    # This keeps intact only the Class dimension (axis 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
    true_sum = tf.reduce_sum(y_true, axis=(0, 1, 2))
    pred_sum = tf.reduce_sum(y_pred, axis=(0, 1, 2))
    
    # 5. Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
    
    # 6. Averages dice over the batch, then returns (1.0 - Dice) for minimization 
    return 1.0 - tf.reduce_mean(dice)