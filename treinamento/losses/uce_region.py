#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:16:37 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf

# %% U-CE Dice Loss

def get_u_dice_loss(use_mask=True, alpha=1.0):
    """
    Factory function for an Uncertainty-Aware Masked Dice Loss.
    Uses a sigma map to penalize uncertain predictions based on the U-CE paper.
    """
    def loss(y_true, y_pred, sigma=None):
        smooth = 1e-5
        
        # 1. Ensure tensors are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 2. Compute the mask before slicing (ignores pixels where y_true is all 0s) 
        if use_mask:
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
        
        # 3. Isolate channel 1 (positive class) 
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
        
        # 4. Apply the exclusion mask
        if use_mask:
            y_true = y_true * mask
            y_pred = y_pred * mask

        # 5. Calculate and Apply the Uncertainty (U-CE) Weight 
        if sigma is not None:
            # Replicate the paper's penalty formula: W = (1 + sigma)^alpha 
            dice_weight_map = (1.0 + sigma) ** alpha
            # Safely expand dimensions to broadcast against the class channel
            dice_weight_map = tf.expand_dims(tf.cast(dice_weight_map, tf.float32), axis=-1)
            
            # Linearly inject the weight into the intersection and margins 
            weighted_y_true = y_true * dice_weight_map
            weighted_y_pred = y_pred * dice_weight_map
            weighted_intersection = (y_true * y_pred) * dice_weight_map
        else:
            weighted_y_true = y_true
            weighted_y_pred = y_pred
            weighted_intersection = y_true * y_pred

        # 6. Global Batch Reductions (Micro-Average) 
        intersection = tf.reduce_sum(weighted_intersection, axis=(0, 1, 2))
        true_sum = tf.reduce_sum(weighted_y_true, axis=(0, 1, 2))
        pred_sum = tf.reduce_sum(weighted_y_pred, axis=(0, 1, 2))
        
        # 7. Calculate final weighted Dice 
        dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
        
        return 1.0 - tf.reduce_mean(dice)
        
    return loss