#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:29:15 2026

@author: rotunno
"""

from abc import ABC, abstractmethod
import tensorflow as tf

class BaseUNetBlock(tf.keras.layers.Layer, ABC):
    """Abstract Base Class for UNet blocks to enforce architectural consistency."""
    
    def __init__(self, filters: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Instantiate dropout only if a rate is provided.
        # This keeps the computational graph perfectly clean if dropout_rate=0.0
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None

    @abstractmethod
    def call(self, inputs, training=None, mc_inference=False):
        pass

    def apply_dropout(self, x, training, mc_inference):
        """Clean helper to route the MC Dropout logic."""
        if self.dropout is not None:
            # Force training=True if we are doing Monte Carlo inference
            # Otherwise, fall back to the standard Keras training state
            dropout_state = True if mc_inference else training
            return self.dropout(x, training=dropout_state)
        return x
    
    
class ResidualEncoderBlock(BaseUNetBlock):
    def __init__(self, filters: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(filters, dropout_rate, **kwargs)
        
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        # 1x1 Conv for the residual shortcut matching
        self.shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')
        
    def call(self, inputs, training=None, mc_inference=False):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        # Apply the abstract dropout logic
        x = self.apply_dropout(x, training, mc_inference)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut path
        res = self.shortcut(inputs)
        
        return tf.nn.relu(x + res)
    
class ResUNet(tf.keras.Model):
    def __init__(self, num_classes: int, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        # Instantiate your blocks
        self.encoder1 = ResidualEncoderBlock(64, dropout_rate)
        self.encoder2 = ResidualEncoderBlock(128, dropout_rate)
        # ... remaining encoder/decoder blocks ...
        
        self.final_conv = tf.keras.layers.Conv2D(
            num_classes, 1, activation='sigmoid'
        )

    def call(self, inputs, training=None, mc_inference=False):
        # Forward the mc_inference flag down the structural hierarchy
        x = self.encoder1(inputs, training=training, mc_inference=mc_inference)
        # ... standard UNet pooling and skipping ...
        
        return self.final_conv(x)
    
# %%

'''
# 1. Standard Evaluation (Dropout OFF)
# Used for standard metrics (IoU, F1)
standard_preds = model(geo_patches, training=False, mc_inference=False)

# 2. Uncertainty-Aware Inference (Dropout ON)
# Generate T stochastic forward passes
T = 20
mc_predictions = []

for _ in range(T):
    # Dropout remains active despite training=False
    preds = model(geo_patches, training=False, mc_inference=True)
    mc_predictions.append(preds)

# Stack and calculate predictive variance for thresholding/filtering logic
stacked_preds = tf.stack(mc_predictions)
mean_prediction = tf.reduce_mean(stacked_preds, axis=0)
uncertainty_variance = tf.math.reduce_variance(stacked_preds, axis=0)
'''