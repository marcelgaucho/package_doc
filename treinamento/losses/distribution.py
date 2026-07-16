# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:27:01 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

# %% Categorical Cross Entropy (with mask option) 

def get_categorical_crossentropy(use_mask=True):
    """
    Factory function that returns a categorical cross-entropy loss function.

    Args:
        use_mask (bool): If True, ignores spatial locations (e.g., pixels) where 
          y_true is all zeros across all classes (acting as an ignore-label).

    Returns:
        function: A loss function expecting y_true and y_pred tensors of shape 
          (batch_size, ..., num_classes).
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute the standard cross loss (without reduction)
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
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

# %% Weighted Categorical Cross Entropy (with mask option) 

def get_weighted_categorical_crossentropy(weights, use_mask=True):
    """
    Factory function that configures and returns a weighted categorical cross-entropy loss.
    
    Args:
        weights (list or np.array): Class weights matching the number of channels.
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
    """
    weights = tf.constant(weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute the standard cross loss (without reduction)
        loss_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # Create the Weight Tensor based on the true class per location
        weight_tensor = tf.reduce_sum(weights * y_true, axis=-1)
        
        if use_mask:
            # Compute the mask (0 for pixels with all 0s in all one-hot classes)
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            
            # Weight the loss and mask it
            weighted_masked_loss = loss_cce * weight_tensor * mask
            
            # Divide by the sum of (weight_tensor * mask) to keep the gradient consistent
            denominator = tf.reduce_sum(weight_tensor * mask) + tf.keras.backend.epsilon()
            return tf.reduce_sum(weighted_masked_loss) / denominator
        else:
            # If not masking, still apply class weights across the entire spatial array
            weighted_loss = loss_cce * weight_tensor
            denominator = tf.reduce_sum(weight_tensor) + tf.keras.backend.epsilon()
            return tf.reduce_sum(weighted_loss) / denominator

    return loss

# %% U-CE Loss

def uce_categorical_crossentropy(y_true, y_pred, sigma, alpha=1.0, e_batch=None):
    """
    Computes the Uncertainty-aware Categorical Cross-Entropy.
    Assumes y_true is one-hot encoded, and [0,0,...] indicates ignored pixels.
    """
    # 1. Base Categorical Cross-Entropy (Unreduced)
    # y_true: [B, H, W, C], y_pred: [B, H, W, C]
    # base_ce shape: [B, H, W]
    base_ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    
    # 2. Apply U-CE Weighting: w = (1 + sigma)^alpha
    u_ce_weight = (1.0 + sigma) ** alpha
    weighted_loss = base_ce * u_ce_weight
    
    # 3. Create the Mask to ignore past deforestation
    # For valid pixels [1, 0] or [0, 1], the sum is 1.0. 
    # For ignored pixels [0, 0], the sum is 0.0.
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1), dtype=weighted_loss.dtype)
    
    # Combine with external sample weights (e_batch) if provided
    if e_batch is not None:
        if len(e_batch.shape) > len(mask.shape):
            e_batch = tf.squeeze(e_batch, axis=-1)
        mask = mask * tf.cast(e_batch, dtype=mask.dtype)
        
    # Zero out the loss for ignored pixels
    masked_loss = weighted_loss * mask
    
    # 4. Safe Mean Reduction (divide by valid pixels only)
    valid_pixels = tf.reduce_sum(mask)
    
    # divide_no_nan prevents errors if a batch happens to be 100% ignored pixels
    final_loss = tf.math.divide_no_nan(tf.reduce_sum(masked_loss), valid_pixels)
    
    return final_loss
