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


