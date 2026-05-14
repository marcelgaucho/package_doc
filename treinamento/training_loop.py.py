#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:20:24 2026

@author: rotunno
"""

# %% Import libraries

import numpy as np
import tensorflow as tf 
from .lr_decay import ReduceOnPlateau
from .steps import train_step, test_step
from .utils import transform_augment_batch
import time


# %%

def train_model_loop(model, epochs, early_stopping_epochs, train_dataset, valid_dataset, optimizer, 
                     loss_fn, metrics_train=[], metrics_val=[], model_path='best_model.keras',
                     early_stopping_delta=0.01, data_augmentation=False, reduce_on_plateau=True,
                     mode='max', augment_batch_factor=2):
    
    # 1. Setup Tracking
    history_train, history_valid = [], []
    best_val_score = -float('inf') if mode == 'max' else float('inf')
    no_improvement_count = 0
    
    # Use Keras metrics for mean loss to handle varying batch sizes automatically
    train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
    val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    if reduce_on_plateau:
        schedule = ReduceOnPlateau(optimizer, decay_factor=0.5, patience=5, min_lr=1e-6, 
                                   min_delta=early_stopping_delta, mode=mode)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} | LR: {optimizer.learning_rate.numpy():.6f}")
        start_time = time.time()

        # --- TRAINING ---
        for step, batches in enumerate(train_dataset):
            # Dynamic unpacking
            x_batch, y_batch = batches[0], batches[1]
            e_batch = batches[2] if len(batches) == 3 else None

            if data_augmentation:
                # Generate multiple augmented versions
                aug_x_list, aug_y_list, aug_e_list = [x_batch], [y_batch], ([e_batch] if e_batch is not None else [])
                
                for _ in range(augment_batch_factor - 1):
                    items = (x_batch, y_batch, e_batch) if e_batch is not None else (x_batch, y_batch)
                    augmented = transform_augment_batch(items)
                    
                    aug_x_list.append(augmented[0])
                    aug_y_list.append(augmented[1])
                    if e_batch is not None: aug_e_list.append(augmented[2])

                x_batch = tf.concat(aug_x_list, axis=0)
                y_batch = tf.concat(aug_y_list, axis=0)
                if e_batch is not None: e_batch = tf.concat(aug_e_list, axis=0)

            # Execution
            loss_value = train_step(x_batch, y_batch, model, loss_fn, optimizer, metrics_train, e_batch)
            train_loss_tracker.update_state(loss_value)

            if step % 200 == 0:
                print(f"Step {step}: Loss {loss_value:.4f}")

        # Finalize Train Metrics
        current_train_loss = train_loss_tracker.result().numpy()
        current_train_metric = metrics_train[0].result().numpy()
        history_train.append(np.array([[current_train_loss, current_train_metric]]))
        
        train_loss_tracker.reset_state()
        for m in metrics_train: m.reset_state()

        # --- VALIDATION ---
        for batches in valid_dataset:
            x_v, y_v = batches[0], batches[1]
            e_v = batches[2] if len(batches) == 3 else None
            
            val_loss_val = test_step(x_v, y_v, model, loss_fn, metrics_val, e_v)
            val_loss_tracker.update_state(val_loss_val)

        # Finalize Val Metrics
        current_val_loss = val_loss_tracker.result().numpy()
        current_val_metric = metrics_val[0].result().numpy()
        history_valid.append(np.array([[current_val_loss, current_val_metric]]))

        print(f"Val Loss: {current_val_loss:.4f} | Val Metric: {current_val_metric:.4f} | Time: {time.time()-start_time:.2f}s")

        # --- SCHEDULERS & EARLY STOPPING ---
        monitor_value = current_val_metric if mode == 'max' else current_val_loss
        if reduce_on_plateau:
            schedule.step(monitor_value)

        # Improvement Logic
        if mode == 'max':
            improved = monitor_value > (best_val_score + early_stopping_delta)
        else:
            improved = monitor_value < (best_val_score - early_stopping_delta)

        if improved:
            best_val_score = monitor_value
            no_improvement_count = 0
            model.save(model_path)
            print("✓ Improvement found. Model saved.")
        else:
            no_improvement_count += 1
            print(f"× No improvement. Patience: {no_improvement_count}/{early_stopping_epochs}")
            if early_stopping_epochs and no_improvement_count >= early_stopping_epochs:
                print("!!! Early Stopping Triggered !!!")
                break

        val_loss_tracker.reset_state()
        for m in metrics_val: m.reset_state()

    return [history_train, history_valid]