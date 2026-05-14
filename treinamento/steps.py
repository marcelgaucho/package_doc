#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:32:24 2026

@author: rotunno
"""

import tensorflow as tf

# %% Do a train step

@tf.function
def train_step(x, y, model, loss_fn, optimizer, metrics_train, entropy_weight=None):
    with tf.GradientTape() as tape:
        model_result = model(x, training=True)
        if entropy_weight is not None:
            loss_value = loss_fn(y, model_result, entropy_weight=entropy_weight)
        else:
            loss_value = loss_fn(y, model_result)
    
    # Calculate and apply gradient
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # Update training metrics
    for train_metric in metrics_train:
        train_metric.update_state(y, model_result)
                
    return loss_value

# %% Do a validation step

@tf.function
def test_step(x, y, model, loss_fn, metrics_val, entropy_weight=None):
    val_result = model(x, training=False)
    if entropy_weight is not None:
        loss_value = loss_fn(y, val_result, entropy_weight=entropy_weight)
    else:
        loss_value = loss_fn(y, val_result)
        
    # Update valid metrics
    for val_metric in metrics_val:
        val_metric.update_state(y, val_result)
        
    return loss_value