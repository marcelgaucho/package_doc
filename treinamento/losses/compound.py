# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:32:24 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from .region import dice_loss

# %% Combo Loss: CCE + Dice

def combo_loss(loss_weights):
    """
    Combined Categorical Cross-Entropy (CCE) and Dice Loss.

    Args:
        loss_weights (list or tuple): A two-element array containing the multiplier 
            weights for the CCE loss and Dice loss, respectively (e.g., [0.5, 0.5]).

    Returns:
        callable: The custom loss function `loss(y_true, y_pred)` for training.
    """
    loss_weights = tf.constant(loss_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        # 1. Ensure tensors in float32 format
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 2. Calculate the standard cross-entropy and reduce it
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss_cce = tf.reduce_mean(loss_cce)
        
        # 3. Calculate the dice loss
        loss_dice = dice_loss(y_true, y_pred)
        
        # 4. Return the combined loss
        return (loss_weights[0] * loss_cce) + (loss_weights[1] * loss_dice)
     
    # Return loss
    return loss