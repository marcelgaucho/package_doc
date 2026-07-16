#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:21:50 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf

# %% U-CE Categorical Cross Entropy (with mask option) 

def get_u_categorical_crossentropy(use_mask=True, alpha=1.0):
    """
    Factory function that returns an Uncertainty-Aware Categorical Cross-Entropy loss.
    
    Args:
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
        alpha (float): Hyperparameter controlling the intensity of the uncertainty penalty.

    Returns:
        function: A loss function expecting y_true, y_pred, and an optional sigma map.
    """
    def loss(y_true, y_pred, sigma=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 1. Compute the standard cross-entropy loss (without reduction) [B, H, W]
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # 2. Apply the Uncertainty (U-CE) Weight dynamically if sigma is provided
        if sigma is not None:
            sigma = tf.cast(sigma, tf.float32)
            
            # Ensure sigma shape matches loss_cce shape [B, H, W]
            # If sigma has a trailing channel dimension (e.g., [B, H, W, 1]), squeeze it
            if len(sigma.shape) > len(loss_cce.shape):
                sigma = tf.squeeze(sigma, axis=-1)
                
            # Replicate the paper's penalty formula: W = (1 + sigma)^alpha
            u_ce_weight = (1.0 + sigma) ** alpha
            
            # Linearly scale the unreduced CCE loss
            loss_cce = loss_cce * u_ce_weight
            
        # 3. Mask and Reduce
        if use_mask:
            # Compute the mask (0 for pixels with all 0s in all one-hot classes)
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            
            # Mask the loss
            masked_loss = loss_cce * mask
            
            # Divide by the total number of valid (non-ignored) pixels
            denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
            return tf.reduce_sum(masked_loss) / denominator
        else:
            # Standard unmasked reduction: average over all pixels in the batch
            return tf.reduce_mean(loss_cce)
            
    return loss

# %% U-CE Weighted Categorical Cross Entropy (with mask option) 

def get_u_weighted_categorical_crossentropy(weights, use_mask=True, alpha=1.0):
    """
    Factory function for an Uncertainty-Aware Weighted Categorical Cross-Entropy loss.
    
    Args:
        weights (list or np.array): Class weights matching the number of channels.
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
        alpha (float): Hyperparameter controlling the intensity of the U-CE penalty.
    """
    # Embed the class weights directly into the graph as constants
    weights = tf.constant(weights, dtype=tf.float32)
    
    def loss(y_true, y_pred, sigma=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 1. Compute the standard cross loss (without reduction) [B, H, W]
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # 2. Apply Uncertainty (U-CE) Weight dynamically if sigma is provided
        if sigma is not None:
            sigma = tf.cast(sigma, tf.float32)
            
            # Ensure sigma shape matches loss_cce shape [B, H, W]
            if len(sigma.shape) > len(loss_cce.shape):
                sigma = tf.squeeze(sigma, axis=-1)
                
            # Replicate the paper's penalty formula: W = (1 + sigma)^alpha
            u_ce_weight = (1.0 + sigma) ** alpha
            
            # Linearly scale the unreduced CCE loss
            loss_cce = loss_cce * u_ce_weight
            
        # 3. Create the Class Weight Tensor based on the true class per location
        weight_tensor = tf.reduce_sum(weights * y_true, axis=-1)
        
        # 4. Mask, Weight, and Reduce
        if use_mask:
            # Compute the mask (0 for pixels with all 0s in all one-hot classes)
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            
            # Combine the U-CE scaled loss, the Class Weights, and the Mask
            weighted_masked_loss = loss_cce * weight_tensor * mask
            
            # Divide by the sum of (weight_tensor * mask) to keep the gradient consistent
            denominator = tf.reduce_sum(weight_tensor * mask) + tf.keras.backend.epsilon()
            return tf.reduce_sum(weighted_masked_loss) / denominator
        else:
            # If not masking, still apply class weights across the entire spatial array
            weighted_loss = loss_cce * weight_tensor
            
            # Divide by the sum of the applied class weights
            denominator = tf.reduce_sum(weight_tensor) + tf.keras.backend.epsilon()
            return tf.reduce_sum(weighted_loss) / denominator

    return loss