#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:21:50 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf

# %% U-CE Categorical Cross Entropy (with weights and mask options) 

def get_u_categorical_crossentropy(class_weights=None, use_mask=True, alpha=1.0):
    """
    Factory function for an Uncertainty-Aware Categorical Cross-Entropy loss, with class weights option.
    
    Args:
        weights (list or np.array): Class weights matching the number of channels.
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
        alpha (float): Hyperparameter controlling the intensity of the U-CE penalty.
    """
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred, sigma=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 1. Compute standard cross-entropy loss per pixel (unreduced) -> [B, H, W]
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # 2. Extract pixel-level class weights
        if class_weights is not None:
            pixel_class_weights = tf.reduce_sum(y_true * class_weights, axis=-1)
        else:
            # Fallback to an array of 1.0s if no weights are passed
            pixel_class_weights = tf.ones_like(loss_cce)
            
        # Apply the static class weights to the numerator
        loss_cce = loss_cce * pixel_class_weights
        
        # 3. Apply Uncertainty (U-CE) Weight dynamically if sigma is provided
        if sigma is not None:
            sigma = tf.cast(sigma, tf.float32)
            if len(sigma.shape) > len(loss_cce.shape):
                sigma = tf.squeeze(sigma, axis=-1)
                
            u_ce_weight = (1.0 + sigma) ** alpha
            loss_cce = loss_cce * u_ce_weight
            
        # 4. Mask and Reduce (Properly Normalized)
        if use_mask:
            # 0 for pixels where y_true is all zeros across all classes
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            
            masked_loss = loss_cce * mask
            
            # DIVIDE BY SUM OF WEIGHTS IN THE MASK
            denominator = tf.reduce_sum(pixel_class_weights * mask) + tf.keras.backend.epsilon()
            return tf.reduce_sum(masked_loss) / denominator
        else:
            # DIVIDE BY SUM OF WEIGHTS ACROSS ENTIRE SPATIAL GRID
            denominator = tf.reduce_sum(pixel_class_weights) + tf.keras.backend.epsilon()
            return tf.reduce_sum(loss_cce) / denominator
            
    return loss