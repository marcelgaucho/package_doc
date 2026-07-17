# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:03:40 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy

# %% 

def custom_entropy_loss(y_true, y_pred, entropy_weight):
    ''' Masked loss with L = CE * Entropy '''
    # Squeeze and Cast entropy as float32
    entropy_weight = tf.cast(tf.squeeze(entropy_weight, axis=-1), tf.float32)
    
    # Compute the standard cross loss (without reduction)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Compute the mask with 0 for ignored pixels (all 0s in one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Mask the loss and multiply by entropy
    masked_loss = cce_loss * mask
    masked_loss = masked_loss * entropy_weight
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator

# %%

class CustomEntropyLoss(Loss):
    ''' Loss with L = CE * Entropy . Entropy is the last channel ''' 
    def __init__(self, name="custom_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def __call__(self, y_true, y_pred, entropy_weight):
        # cast y_true, y_pred and entropy (extracted from input_tensor) as float dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        entropy = tf.cast(entropy_weight[..., -1:], tf.float32)
        
        # Compute cross-entropy with TF class 
        cross_loss = CategoricalCrossentropy(reduction='none')
        cross_tf = cross_loss(y_true, y_pred)
        cross_tf = tf.expand_dims(cross_tf, axis=-1)
        
        # Multiply entropy by cross-entropy result
        loss = entropy * cross_tf
        
        # Return the tensor mean
        return tf.math.reduce_mean(loss)
    
# %% Custom entropy loss (with stepped approach)

def custom_entropy_loss_weights(y_true, y_pred, entropy_weight):
    """
    Tweaked loss function using a stepped-weight approach to prevent 
    gradient vanishing and catastrophic forgetting during fine-tuning.
    """
    entropy_threshold = 0.4 
    high_entropy_weight = 2.0 
    low_entropy_weight = 0.1
    
    # Squeeze and Cast entropy as float32
    entropy_weight = tf.cast(tf.squeeze(entropy_weight, axis=-1), tf.float32)
    
    # Compute the standard cross loss (without reduction)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Compute the mask with 0 for ignored pixels (all 0s in one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # --- Stepped Weight Approach ---
    # Assign a baseline weight to low-entropy regions so they aren't zeroed out,
    # and a higher weight to high-entropy regions to focus the fine-tuning.
    stepped_weights = tf.where(
        entropy_weight > entropy_threshold,
        high_entropy_weight,
        low_entropy_weight
    )
    
    # Mask the loss and multiply by stepped weights
    masked_loss = cce_loss * mask
    masked_loss = masked_loss * stepped_weights
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator