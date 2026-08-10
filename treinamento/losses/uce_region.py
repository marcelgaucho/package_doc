#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:16:37 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf

# %% U-CE Dice Loss

# %% U-CE Dice Loss

def get_u_dice_loss(class_weights=None, use_mask=True, alpha=1.0, downweight_uncertainty=False):
    """
    Factory function for an Uncertainty-Aware Dice Loss.
    
    Args:
        class_weights: Optional list/array of weights per class (e.g., [0.0, 1.0]).
        use_mask: Boolean to ignore all-zero ground truth pixels.
        alpha: Exponential scaling factor for the uncertainty penalty.
        downweight_uncertainty: Boolean to test the inverse formula W = 1 / (1+sigma)^alpha.
    """
    # Embed weights directly into the graph as constants
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred, sigma=None):
        smooth = 1e-5
        
        # 1. Ensure tensors are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 2. Compute the spatial exclusion mask
        if use_mask:
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
        
        # 3. Apply the exclusion mask
        if use_mask:
            y_true = y_true * mask
            y_pred = y_pred * mask

        # 4. Apply the Uncertainty SPATIAL Weight
        if sigma is not None:
            if downweight_uncertainty:
                # Higher uncertainty -> lower contribution to Dice
                dice_weight_map = 1.0 / ((1.0 + sigma) ** alpha)
            else:
                # Higher uncertainty -> higher contribution to Dice
                # Consistent with the U-CE weighting philosophy
                dice_weight_map = (1.0 + sigma) ** alpha
                
            # Safely expand dimensions to broadcast against the class channel 
            # Expected sigma shape: (B, H, W)
            # (B, H, W) -> (B, H, W, 1)
            dice_weight_map = tf.expand_dims(tf.cast(dice_weight_map, tf.float32), axis=-1)
            
            # Linearly inject the weight into the intersection and margins
            weighted_y_true = y_true * dice_weight_map
            weighted_y_pred = y_pred * dice_weight_map
            weighted_intersection = (y_true * y_pred) * dice_weight_map
        else:
            weighted_y_true = y_true
            weighted_y_pred = y_pred
            weighted_intersection = y_true * y_pred

        # 5. Global Batch Reductions (Micro-Average)
        intersection = tf.reduce_sum(weighted_intersection, axis=(0, 1, 2))
        true_sum = tf.reduce_sum(weighted_y_true, axis=(0, 1, 2))
        pred_sum = tf.reduce_sum(weighted_y_pred, axis=(0, 1, 2))
        
        # 6. Calculate unreduced array of Dice scores [Shape: (num_classes,)]
        dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
        
        # 7. Apply Static CLASS Weights safely to the final scores
        if class_weights is not None:
            weighted_dice = dice * class_weights
            # Normalize by the sum of the weights (e.g., dividing by 1.0 if weights are [0.0, 1.0])
            return 1.0 - (tf.reduce_sum(weighted_dice) / (tf.reduce_sum(class_weights)))
        else:
            return 1.0 - tf.reduce_mean(dice)
        
    return loss