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
        
    def __call__(self, y_true, y_pred, input_tensor):
        # cast y_true, y_pred and entropy (extracted from input_tensor) as float dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        entropy = tf.cast(input_tensor[..., 3:4], tf.float32)
        
        # Compute cross-entropy with TF class 
        cross_loss = CategoricalCrossentropy(reduction='none')
        cross_tf = cross_loss(y_true, y_pred)
        cross_tf = tf.expand_dims(cross_tf, axis=-1)
        
        # Multiply entropy by cross-entropy result
        loss = entropy * cross_tf
        
        # Return the tensor mean
        return tf.math.reduce_mean(loss)