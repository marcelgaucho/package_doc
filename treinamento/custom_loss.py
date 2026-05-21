# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 00:31:26 2025

@author: Marcel
"""

# Custom loss implementation that uses entropy as a weight

# %% Imports

import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy

# %% Loss Class

class CustomEntropyLoss(Loss):
    def __init__(self, name="custom_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def __call__(self, y_true, y_pred, entropy_weight):
        # cast y_true, y_pred and entropy (extracted from input_tensor) as float dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        entropy = tf.cast(entropy_weight[..., 3:4], tf.float32)
        
        # Compute cross-entropy with TF class 
        cross_loss = CategoricalCrossentropy(reduction='none')
        cross_tf = cross_loss(y_true, y_pred)
        cross_tf = tf.expand_dims(cross_tf, axis=-1)
        
        # Multiply entropy by cross-entropy result
        loss = entropy * cross_tf
        
        # Return the tensor mean
        return tf.math.reduce_mean(loss)
    
# %% Masked Weighted Categorical Cross Entropy (Implementação IA Google)

def masked_weighted_cce(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # print('y_true', type(y_true), y_true)
        # print('y_pred', type(y_pred), y_pred)
        
        # Compute the standard cross loss (without reduction)
        cce_tf = CategoricalCrossentropy(reduction='none')
        cce_loss = cce_tf(y_true, y_pred)
        
        # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
        mask = tf.reduce_sum(y_true, axis=-1)
        mask = tf.cast(mask > 0, dtype=tf.float32)
        
        # Create the Weight Tensor
        weight_tensor = tf.reduce_sum(weights * y_true, axis=-1)
        
        # Weight the loss and mask it
        weighted_masked_loss = cce_loss * weight_tensor * mask
        
        # Reduce loss dividing by the sum of the (weight_tensor * mask) to keep the gradient consistent
        denominator = tf.reduce_sum(weight_tensor * mask) + tf.keras.backend.epsilon()
        
        # Return the division
        return tf.reduce_sum(weighted_masked_loss) / denominator

    return loss

# %% Masked Categorical Cross Entropy 

def masked_cce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # print('y_true', type(y_true), y_true)
    # print('y_pred', type(y_pred), y_pred)
    
    # Compute the standard cross loss (without reduction)
    cce_tf = CategoricalCrossentropy(reduction='none')
    cce_loss = cce_tf(y_true, y_pred)
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Mask the loss
    masked_loss = cce_loss * mask
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator

# %% Custom entropy loss

def custom_entropy_loss(y_true, y_pred, entropy_weight):
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

# %% Custom offset entropy loss

def custom_offset_entropy_loss(y_true, y_pred, entropy_weight):
    ''' Masked loss with L = CE * (1 + Entropy) ''' 
    # Squeeze and Cast entropy as float32
    entropy_weight = tf.cast(tf.squeeze(entropy_weight, axis=-1), tf.float32)
    
    # Compute the standard cross loss (without reduction)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Multiply by (1+entropy) and mask the loss
    masked_loss = cce_loss * (1 + entropy_weight)
    masked_loss = masked_loss * mask
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator


# %% Custom addictive entropy loss

def custom_add_entropy_loss(y_true, y_pred, entropy_weight):
    ''' Masked loss with L = CE + Entropy ''' 
    # Squeeze and Cast entropy as float32
    entropy_weight = tf.cast(tf.squeeze(entropy_weight, axis=-1), tf.float32)
    
    # Compute the standard cross loss (without reduction)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Sum with entropy and mask the loss
    masked_loss = cce_loss + entropy_weight
    masked_loss = masked_loss * mask
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator

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
