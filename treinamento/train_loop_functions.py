# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:18:54 2024

@author: Marcel
"""

# Functions used in train loop

# %% Imports 
import time, gc
import numpy as np
import tensorflow as tf
from .utils import transform_augment
from inspect import signature

# %% Do a train step

@tf.function
def train_step(x, y, model, loss_fn, optimizer, metrics_train, input_tensor=None):
    with tf.GradientTape() as tape:
        model_result = model(x, training=True)
        if input_tensor is not None:
            loss_value = loss_fn(y, model_result, input_tensor=input_tensor)
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
def test_step(x, y, model, loss_fn, metrics_val, input_tensor=None):
    val_result = model(x, training=False)
    if input_tensor is not None:
        loss_value = loss_fn(y, val_result, input_tensor=input_tensor)
    else:
        loss_value = loss_fn(y, val_result)
        
    # Update valid metrics
    for val_metric in metrics_val:
        val_metric.update_state(y, val_result)
        
    return loss_value

# %% Vectorize map function to increase velocity

@tf.function
def vectorized_map(map_augment_function, tuple_x_y):
    return tf.vectorized_map(map_augment_function, tuple_x_y)

# %% Do the train loop

def train_model_loop(model, epochs, early_stopping_epochs, train_dataset, valid_dataset, optimizer, 
                    loss_fn, metrics_train=[], metrics_val=[], model_path='best_model.keras',
                    early_stopping_delta=0.01, data_augmentation=False, 
                    early_stopping_on_metric=True, augment_batch_factor=2):
    '''
    Train the model for a certain number of epochs, with early stopping of early_stopping_epochs,
    saving the model inside model_savedir directory. The early stopping is based on the 
    first metric of metrics_val list, in case early_stopping_on_metric is True, otherwise the loss
    is used; in both cases the best performance, on metric or loss, is saved.
    '''
    # Parameters of loss call (more than y_true and y_pred?)
    sig_losscall = signature(loss_fn.__call__)
    loss_parameters_names = [param.name for param in sig_losscall.parameters.values()]
    
    # Loss and metric values are stored in lists
    history_train = [] 
    history_valid = []
    
    # Initialize the variables that store the best value of metric and loss
    valid_metric_best_model = 1e-20
    valid_loss_best_model = float('inf')
    
    # Epochs without improvement
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch))
        start_time = time.time()
        
        # Initialize accumulated loss in train and validation
        train_loss = 0
        valid_loss = 0
        
        # Loop through dataset batches
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            if data_augmentation:
                # Compute "new" batches 
                x_batches_train_augmented = []
                y_batches_train_augmented = []
                for _ in range(augment_batch_factor-1):
                    x_batch_train_augmented, y_batch_train_augmented = vectorized_map(transform_augment, (x_batch_train, y_batch_train))
                    x_batches_train_augmented.append(x_batch_train_augmented)
                    y_batches_train_augmented.append(y_batch_train_augmented)
                    
                # Concatenate original batch with new "batches"
                x_batch_train = tf.concat([x_batch_train] + x_batches_train_augmented, axis=0) 
                y_batch_train = tf.concat([y_batch_train] + y_batches_train_augmented, axis=0)                    
                
                # Delete computed variables
                del x_batch_train_augmented, y_batch_train_augmented, x_batches_train_augmented[:], y_batches_train_augmented[:]
                gc.collect()

            # Train Step
            if 'input_tensor' in loss_parameters_names:
                loss_value = train_step(x_batch_train, y_batch_train, model, loss_fn, optimizer, metrics_train,
                                        input_tensor=x_batch_train)
            else:
                loss_value = train_step(x_batch_train, y_batch_train, model, loss_fn, optimizer, metrics_train)
            
            # Increase train loss accumulator
            train_loss = train_loss + loss_value            
                
            # Log every 200 batches
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * x_batch_train.shape[0])) # x_batch_train.shape[0] é o batch size final, já com aumento de dados
        
        # Write to train history the train loss (mean loss or accumulated loss/number of batches) and the first metric for the epoch
        history_train.append(   np.array([[train_loss/(step+1), metrics_train[0].result()]])   )         
        
        # Show train metrics
        for train_metric in metrics_train:
            result_train_metric = train_metric.result()            
            print(f"Training {train_metric.__class__.__name__.lower()} over epoch: {result_train_metric:.4f}")
            train_metric.reset_state() # Reset train metric in the end of the epoch
            
        # Compute validation results in the end of the epoch
        for x_batch_val, y_batch_val in valid_dataset:
            # Validation step
            if 'input_tensor' in loss_parameters_names:
                loss_value = test_step(x_batch_val, y_batch_val, model, loss_fn, metrics_val,
                                       input_tensor=x_batch_val)
            else:
                loss_value = test_step(x_batch_val, y_batch_val, model, loss_fn, metrics_val)

            # Increase valid loss accumulator  
            valid_loss = valid_loss + loss_value


        # Write to validation history the valid loss (mean loss or accumulated loss/number of batches) and the first metric for the epoch
        metrics_val0 = metrics_val[0].result()
        loss_val = valid_loss/(step+1)
        history_valid.append(   np.array([[loss_val, metrics_val0]])   )   

        # Reset accumulated training and validation losses for the next epoch
        train_loss = 0
        valid_loss = 0
 
        # Show validation metrics and the time taken in the end of the epoch
        for val_metric in metrics_val:
            result_val_metric = val_metric.result()
            print(f"Validation {val_metric.__class__.__name__.lower()}: {result_val_metric:.4f}")
            val_metric.reset_state()
        print("Time taken: %.2fs" % (time.time() - start_time))
            
        
        # Early Stopping
        if early_stopping_epochs:
            # Early Stopping on Metric
            if early_stopping_on_metric:
                # If the absolute value of the increase (or decrease) is below early_stopping_delta,
                # then proceed to Early Stopping count
                diff = metrics_val0 - valid_metric_best_model
                if abs(diff) < early_stopping_delta:
                    # Stop if there are no improvement during early_stopping_epochs  
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Metric decreasing, as metrics_val0 - valid_metric_best_model is less than 0
                # Because this is bad, proceed to Early Stopping count
                elif diff < 0:
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Metric increasing, as metrics_val0 - valid_metric_best_model is greater than 0
                # Normally this is positive, such as accuracy, precision, recall increasing.
                # So we save the model and reset the Early Stopping count
                else:
                    valid_metric_best_model = metrics_val0
                    no_improvement_count = 0
        
                    # Saving best model  
                    print("Saving the model...")
                    model.save(model_path)
            # Early Stopping on Loss
            else:
                diff = loss_val - valid_loss_best_model
                if abs(diff) < early_stopping_delta:
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Loss decreasing, as loss_val - valid_loss_best_model is less than 0
                # This is good, so save the model and reset Early Stopping count
                elif diff < 0:
                    valid_loss_best_model = loss_val
                    no_improvement_count = 0
        
                    # Saving best model  
                    print("Saving the model...")
                    model.save(model_path)
                       
                # Loss increasing, as loss_val - valid_loss_best_model is greater than 0
                # This is bad, so proceed to Early Stopping count
                else:
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
        # Without Early Stopping, save the model with smallest loss
        else:
            diff = loss_val - valid_loss_best_model
            if diff < 0 and abs(diff) < early_stopping_delta: 
                valid_loss_best_model = loss_val
    
                # Saving best model  
                print("Saving the model...")
                model.save(model_path)
            

    # Return list of 2 lists, each one is a list of tuples
    # The first is with training history and the second is with Primeira é com histórico de treino e segunda com histórico de validação
    # Each tuple has in its first element the loss and in its second element the metric for the epoch corresponding to its position
    return [ history_train, history_valid ]