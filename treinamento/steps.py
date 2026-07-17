#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:32:24 2026

@author: rotunno
"""

import tensorflow as tf

# %% Train step (for normal losses and for U-CE)

@tf.function
def train_step(x_batch, y_batch, model, loss_fn, optimizer, metrics_train, 
               e_batch=None, use_uce=False, mc_samples=5):
    
    # 1. Compute Uncertainty SAFELY (Outside GradientTape)
    if use_uce:
        # Run MC passes without tracking gradients to save massive VRAM
        mc_preds = []
        for _ in range(mc_samples):
            mc_preds.append(model(x_batch, training=True))
            
        mc_preds_stack = tf.stack(mc_preds)
        
        # Calculate Mean and Variance
        mean_preds = tf.reduce_mean(mc_preds_stack, axis=0)
        bessel_factor = tf.cast(mc_samples, tf.float32) / tf.cast(mc_samples - 1, tf.float32)
        variance = tf.math.reduce_variance(mc_preds_stack, axis=0) * bessel_factor 
        std_preds = tf.sqrt(variance + 1e-7) # Epsilon prevents NaN
        
        # Extract standard deviation for the predicted class
        pred_class = tf.argmax(mean_preds, axis=-1)
        num_classes = tf.shape(mean_preds)[-1]
        pred_class_one_hot = tf.one_hot(pred_class, depth=num_classes)
        
        # Calculate sigma and lock it as a constant
        sigma = tf.reduce_sum(std_preds * pred_class_one_hot, axis=-1)
        sigma = tf.stop_gradient(sigma) 
    else:
        sigma = None

    # 2. Build Loss Arguments 
    # This prepares the exact parameters your specific loss function expects
    loss_kwargs = {}
    if sigma is not None:
        loss_kwargs['sigma'] = sigma
    if e_batch is not None:
        loss_kwargs['e_batch'] = e_batch

    # 3. Execute Forward Pass and Loss 
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        
        # Unpack the dictionary seamlessly into the loss function
        loss_value = loss_fn(y_true=y_batch, y_pred=preds, **loss_kwargs)

    # 4. Backpropagate
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 5. Update Metrics
    for train_metric in metrics_train:
        train_metric.update_state(y_batch, preds)
        
    return loss_value

# %% Do a validation step

@tf.function
def test_step(x_batch, y_batch, model, loss_fn, metrics_val, e_batch=None):
    
    # 1. Execute Deterministic Forward Pass (No Dropout)
    preds = model(x_batch, training=False)
    
    # 2. Build Loss Arguments Dynamically
    # We only inject the parameters that are explicitly provided
    loss_kwargs = {}
    if e_batch is not None:
        loss_kwargs['e_batch'] = e_batch
        
    # 3. Calculate Loss seamlessly
    # Sigma safely defaults to None inside the loss function signatures
    loss_value = loss_fn(y_true=y_batch, y_pred=preds, **loss_kwargs)
        
    # 4. Update Validation Metrics
    for val_metric in metrics_val:
        val_metric.update_state(y_batch, preds)
        
    return loss_value




