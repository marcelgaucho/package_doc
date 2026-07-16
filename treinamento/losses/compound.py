# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:32:24 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from .distribution import get_categorical_crossentropy
from .region import get_dice_loss

# %% Combo Loss: CCE + Dice

def get_combo_loss(loss_weights, use_mask=True):
    """
    Factory function that returns a Combined CCE and Dice Loss function.

    Args:
        loss_weights (list or tuple): A two-element array containing the multiplier 
            weights for the CCE loss and Dice loss, respectively (e.g., [0.5, 0.5]).
        use_mask (bool): If True, ignores spatial locations (e.g., pixels) where 
            y_true is all zeros across all classes.

    Returns:
        callable: The custom loss function `loss(y_true, y_pred)` for training.
    """
    # Embed weights directly into the TensorFlow graph as constants
    loss_weights = tf.constant(loss_weights, dtype=tf.float32)
    
    # Instantiate the individual masked loss functions once
    # (This assumes get_categorical_crossentropy and get_dice_loss are in the same script)
    cce_loss_fn = get_categorical_crossentropy(use_mask=use_mask)
    dice_loss_fn = get_dice_loss(use_mask=use_mask)

    def loss(y_true, y_pred):
        # 1. Ensure tensors in float32 format
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 2. Calculate the masked cross-entropy
        loss_cce = cce_loss_fn(y_true, y_pred)
        
        # 3. Calculate the masked dice loss
        loss_dice = dice_loss_fn(y_true, y_pred)
        
        # 4. Return the combined weighted loss
        return (loss_weights[0] * loss_cce) + (loss_weights[1] * loss_dice)
      
    return loss