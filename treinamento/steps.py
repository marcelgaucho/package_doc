#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:32:24 2026

@author: rotunno
"""

import tensorflow as tf
from . import losses
from .losses.distribution import uce_categorical_crossentropy

# %% Train step (for normal losses and for U-CE)

@tf.function
def train_step(x_batch, y_batch, model, loss_fn, optimizer, metrics_train, e_batch=None, 
               use_uce=False, mc_samples=5):
    
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
        variance = tf.math.reduce_variance(mc_preds_stack, axis=0) * bessel_factor # Apply bessel correction to variance
        std_preds = tf.sqrt(variance + 1e-7) # Epsilon prevents NaN
        
        # Extract standard deviation for the predicted class
        pred_class = tf.argmax(mean_preds, axis=-1)
        num_classes = tf.shape(mean_preds)[-1]
        pred_class_one_hot = tf.one_hot(pred_class, depth=num_classes)
        
        # Calculate sigma and turns it into a constant by detaching it from the computation graph
        sigma = tf.reduce_sum(std_preds * pred_class_one_hot, axis=-1)
        sigma = tf.stop_gradient(sigma) 
    else:
        sigma = None
        mean_preds = None # Not needed if not U-CE

    # 2. Execute Loss (Inside GradientTape)
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        
        # Call whatever generic loss_fn was passed in!
        if e_batch is not None:
            loss_value = loss_fn(y_true=y_batch, y_pred=preds, sigma=sigma, e_batch=e_batch)

    # 3. Backprop
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 4. Update Metrics
    for train_metric in metrics_train:
        train_metric.update_state(y_batch, preds)
        
    return loss_value


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
    
    # ==========================================
    # 1. UNCERTAINTY ESTIMATION (Outside GradientTape)
    # ==========================================
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
    
    sigma = tf.reduce_sum(std_preds * pred_class_one_hot, axis=-1)
    
    # Crucial: Detach sigma from the computation graph.
    # This turns sigma into a constant weight map for the loss function.
    sigma = tf.stop_gradient(sigma)

    # ==========================================
    # 2. SINGLE PASS & GRADIENTS (Inside GradientTape)
    # ==========================================
    with tf.GradientTape() as tape:
        # Run exactly ONE forward pass to track activations for gradients
        preds = model(x_batch, training=True)
        
        # Compute Loss using the single prediction and the detached uncertainty map
        loss_value = uce_categorical_crossentropy(
            y_true=y_batch, 
            y_pred=preds,  # Using the single pass prediction here
            sigma=sigma, 
            alpha=alpha, 
            e_batch=e_batch
        )

    # 3. Backpropagation (Now 5x faster and uses 1/5th the VRAM)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # ==========================================
    # 4. METRICS UPDATE
    # ==========================================
    metric_mask = tf.reduce_sum(y_batch, axis=-1)
    if e_batch is not None:
        metric_mask = metric_mask * tf.cast(tf.squeeze(e_batch), dtype=metric_mask.dtype)
        
    # We update metrics using the mean ensemble prediction because it represents 
    # the true "Bayesian" output of the model for this step.
    for metric in metrics_train:
        metric.update_state(y_batch, mean_preds, sample_weight=metric_mask)
        
    return loss_value

# %%

@tf.function
def test_step_uce(x_batch, y_batch, model, metrics_val, e_batch=None):
    # 1. Deterministic forward pass (Dropout is disabled)
    preds = model(x_batch, training=False)
    
    # 2. Base Categorical Cross-Entropy (Unreduced)
    # y_batch: [B, H, W, C], preds: [B, H, W, C]
    # base_ce shape: [B, H, W]
    base_ce = tf.keras.losses.categorical_crossentropy(y_batch, preds, from_logits=False)
    
    # 3. Create the Mask to ignore past deforestation
    # For valid pixels [1, 0] or [0, 1], the sum is 1.0. 
    # For ignored pixels [0, 0], the sum is 0.0.
    mask = tf.cast(tf.reduce_sum(y_batch, axis=-1), dtype=base_ce.dtype)
    
    # Combine with external sample weights (e_batch) if provided
    if e_batch is not None:
        if len(e_batch.shape) > len(mask.shape):
            e_batch = tf.squeeze(e_batch, axis=-1)
        mask = mask * tf.cast(e_batch, dtype=mask.dtype)
        
    # 4. Safe Mean Reduction for Validation Loss
    masked_loss = base_ce * mask
    valid_pixels = tf.reduce_sum(mask)
    
    val_loss = tf.math.divide_no_nan(tf.reduce_sum(masked_loss), valid_pixels)
    
    # 5. Update Metrics (crucial: pass the mask as sample_weight)
    for metric in metrics_val:
        metric.update_state(y_batch, preds, sample_weight=mask)
        
    return val_loss