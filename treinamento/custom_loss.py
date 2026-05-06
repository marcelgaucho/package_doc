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
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Mask the loss and multiply by entropy
    masked_loss = cce_loss * mask
    masked_loss = masked_loss * entropy_weight
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator

# %% Custom offset entropy loss

def custom_offset_entropy_loss(y_true, y_pred, entropy_weight):
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


