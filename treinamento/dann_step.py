# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:11:48 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf



# %%

@tf.function
def dann_train_step(batches, model, loss_fn, optimizer, metrics_train):
    # 1. Unpack the zipped batches 
    (source_x, source_y), target_x = batches
    
    # 2. Create Domain Labels (0 for Source, 1 for Target) [cite: 26]
    batch_size = tf.shape(source_x)[0]
    domain_labels_source = tf.zeros((batch_size, 1))
    domain_labels_target = tf.ones((batch_size, 1))
    
    combined_x = tf.concat([source_x, target_x], axis=0)
    combined_domain_labels = tf.concat([domain_labels_source, domain_labels_target], axis=0)
    
    # Binary Crossentropy for the Domain Classifier
    domain_loss_fn = tf.keras.losses.BinaryCrossentropy()

    with tf.GradientTape() as tape:
        # Assuming your model returns: label_preds, domain_preds
        # Or you pass through sub-models if decoupled
        label_preds, domain_preds = model(combined_x, training=True)
        
        # Split label predictions to only evaluate source data where we have ground truth
        source_label_preds = label_preds[:batch_size]
        
        # Calculate individual losses 
        seg_loss = loss_fn(source_y, source_label_preds)
        dom_loss = domain_loss_fn(combined_domain_labels, domain_preds)
        
        # Total loss for backpropagation
        total_loss = seg_loss + dom_loss

    # Backpropagate and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics (assumes metrics_train[0] is segmentation accuracy/F1)
    metrics_train[0].update_state(source_y, source_label_preds)
    
    return total_loss