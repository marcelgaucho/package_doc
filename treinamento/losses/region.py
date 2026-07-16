# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:39:45 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf

# %% Dice Loss

def get_dice_loss(use_mask=True):
    """
    Factory function that returns a Dice Loss function.

    Args:
        use_mask (bool): If True, ignores spatial locations (e.g., pixels) where 
          y_true is all zeros across all classes (acting as an ignore-label).

    Returns:
        function: A loss function expecting y_true and y_pred tensors of shape 
          (batch_size, ..., num_classes).
    """
    def loss(y_true, y_pred):
        # Smooth constant added to numerator and denominator to 
        # stabilize training and avoid zero division
        smooth = 1e-5
        
        # 1. Ensure tensors in float32 format
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 2. Compute the mask BEFORE slicing the channels
        if use_mask:
            # 0 for pixels with all 0s in all one-hot classes
            mask = tf.reduce_sum(y_true, axis=-1)
            mask = tf.cast(mask > 0, dtype=tf.float32)
            # Expand dimensions to broadcast across the remaining channels safely
            mask = tf.expand_dims(mask, axis=-1)
        
        # 3. Isolate channel 1 (roads) due to massive background (class imbalance)
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
        
        # 4. Apply the mask to both ground truth and predictions
        if use_mask:
            y_true = y_true * mask
            y_pred = y_pred * mask
            
        # 5. Sum over batch and spatial dimensions (Batch, Height, Width) -> axes 0, 1 and 2
        # This keeps intact only the Class dimension (axis 3)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
        true_sum = tf.reduce_sum(y_true, axis=(0, 1, 2))
        pred_sum = tf.reduce_sum(y_pred, axis=(0, 1, 2))
        
        # 6. Calculate Dice coefficient
        dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
        
        # 7. Average dice over the batch (if multiple classes remain), then return
        return 1.0 - tf.reduce_mean(dice)
        
    return loss

