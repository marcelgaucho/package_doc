#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:31:27 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf
from .uce_distribution import get_u_categorical_crossentropy
from .uce_region import get_u_dice_loss

# %% U-CE Combo Loss: CCE + Dice

def get_u_combo_loss(loss_weights=[0.5, 0.5], use_mask=True, alpha=1.0):
    """
    Factory function that returns an Uncertainty-Aware Combined CCE and Dice Loss.

    Args:
        loss_weights (list or tuple): Multiplier weights for the CCE and Dice loss.
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
        alpha (float): Hyperparameter controlling the intensity of the U-CE penalty.

    Returns:
        callable: The custom loss function `loss(y_true, y_pred, sigma=None)` for training.
    """
    # 1. Embed weights directly into the TensorFlow graph as constants
    loss_weights = tf.constant(loss_weights, dtype=tf.float32)
    
    # 2. Instantiate the individual Uncertainty-Aware masked loss functions once
    # (This assumes get_u_categorical_crossentropy and get_u_dice_loss are in your script)
    u_cce_loss_fn = get_u_categorical_crossentropy(use_mask=use_mask, alpha=alpha)
    u_dice_loss_fn = get_u_dice_loss(use_mask=use_mask, alpha=alpha)

    def loss(y_true, y_pred, sigma=None):
        # 3. Ensure tensors are in float32 format
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 4. Calculate the masked, uncertainty-aware cross-entropy
        loss_cce = u_cce_loss_fn(y_true, y_pred, sigma=sigma)
        
        # 5. Calculate the masked, uncertainty-aware dice loss
        loss_dice = u_dice_loss_fn(y_true, y_pred, sigma=sigma)
        
        # 6. Return the combined weighted loss
        return (loss_weights[0] * loss_cce) + (loss_weights[1] * loss_dice)
      
    return loss
