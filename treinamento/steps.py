#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:32:24 2026

@author: rotunno
"""

import tensorflow as tf
from .custom_loss import uce_categorical_crossentropy

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


# %%


@tf.function
def train_step_uce(x_batch, y_batch, model, optimizer, metrics_train, e_batch=None, mc_samples=5, alpha=1.0):
    with tf.GradientTape() as tape:
        # 1. MC-Dropout: Run multiple forward passes
        mc_preds = []
        for _ in range(mc_samples):
            mc_preds.append(model(x_batch, training=True))
            
        mc_preds_stack = tf.stack(mc_preds)
        
        # Mean prediction across the MC samples
        mean_preds = tf.reduce_mean(mc_preds_stack, axis=0)
        
        # 2. Calculate Predictive Uncertainty (Sigma)
        std_preds = tf.math.reduce_std(mc_preds_stack, axis=0)
        pred_class = tf.argmax(mean_preds, axis=-1)
        num_classes = tf.shape(mean_preds)[-1]
        pred_class_one_hot = tf.one_hot(pred_class, depth=num_classes)
        
        # Extract the standard deviation specifically for the predicted class
        sigma = tf.reduce_sum(std_preds * pred_class_one_hot, axis=-1)
        
        # 3. Compute Loss using our modular function
        loss_value = uce_categorical_crossentropy(
            y_true=y_batch, 
            y_pred=mean_preds, 
            sigma=sigma, 
            alpha=alpha, 
            e_batch=e_batch
        )

    # 4. Backpropagation
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 5. Update Metrics (passing the mask so metrics ignore [0,0] pixels)
    # This prevents your accuracy/IoU from being artificially inflated by ignoring pixels
    metric_mask = tf.reduce_sum(y_batch, axis=-1)
    if e_batch is not None:
        metric_mask = metric_mask * tf.cast(tf.squeeze(e_batch), dtype=metric_mask.dtype)
        
    for metric in metrics_train:
        metric.update_state(y_batch, mean_preds, sample_weight=metric_mask)
        
    return loss_value