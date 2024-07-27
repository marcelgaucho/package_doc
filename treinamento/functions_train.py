# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:25:39 2024

@author: Marcel
"""

import numpy as np
import os, shutil
import math
import tensorflow as tf
import pickle
import time
import types
import gc
from pathlib import Path
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

from .f1_metric import F1Score, RelaxedF1Score

from transformers import SegformerFeatureExtractor, TFSegformerForSemanticSegmentation

from .arquiteturas.unetr_2d import config_dict
print(config_dict)
from .arquiteturas.modelos import build_model

from tensorflow.keras.optimizers import Adam, SGD
#from focal_loss import SparseCategoricalFocalLoss

import matplotlib.pyplot as plt





# Função que mostra gráfico
def show_graph_loss_accuracy(history, metric_name = 'accuracy', save=False, save_path=r'', save_name='plotagem.png'):
    # Prepare to new function
    history_train = {}
    history_train['loss'] = list(np.array(history[0]).squeeze()[:, 0])
    history_train['f1'] = list(np.array(history[0]).squeeze()[:, 1])
    history_valid = {}
    history_valid['loss'] = list(np.array(history[1]).squeeze()[:, 0])
    history_valid['f1'] = list(np.array(history[1]).squeeze()[:, 1])
    
    # Training epochs and steps in x ticks
    total_epochs_training = len(history_train['loss'])
    x_ticks_step = 5
    
    # Create Figure
    plt.figure(figsize=(15,6))
    
    # There are 2 subplots in a row
    
    # X and X ticks for all subplots
    x = list(range(1, total_epochs_training+1)) # Could be length of other parameter too
    x_ticks = list(range(0, total_epochs_training+x_ticks_step, x_ticks_step))
    x_ticks.insert(1, 1)
    
    # First subplot (Metric)
    plt.subplot(1, 2, 1)
    
    plt.title(f'{metric_name} per Epoch', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 19}) # Title, with font name and size
    
    plt.xlabel('Epochs', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) # Label of X axis, with font
    plt.ylabel(f'{metric_name}', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) # Label of X axis, with font
    
    plt.plot(x, history_train['f1']) # Plot Train Metric
    plt.plot(x, history_valid['f1']) # Plot Valid Metric
    
    plt.ylim(bottom=0, top=1) # Set y=0 on horizontal axis, and for maximum y=1
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # Set y ticks
    plt.xticks(x_ticks) # Set x ticks
    
    plt.legend(['Train', 'Valid'], loc='upper left', fontsize=12) # Legend, with position and fontsize
    plt.grid(True) # Create grid
    
    # Second subplot (Loss)
    plt.subplot(1, 2, 2)
    
    plt.title('Loss per Epoch', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 19})
     
    plt.xlabel('Epochs', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) 
    plt.ylabel('Loss', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) 
    
    plt.plot(x, history_train['loss'])
    plt.plot(x, history_valid['loss'])
    
    
    plt.ylim(bottom=0, top=1) 
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(x_ticks)
    
    plt.legend(['Train', 'Valid'], loc='upper right', fontsize=12)
    plt.grid(True) 
    
    # Adjust layout
    plt.tight_layout()
    
    # Show or save plot
    if save:
        plt.savefig(save_path+save_name)
        plt.close()
    else:
        plt.show(block=False)
        
        
        
@tf.function
def train_step(x, y, model, loss_fn, optimizer, metrics_train):
    with tf.GradientTape() as tape:
        model_result = model(x, training=True)
        loss_value = loss_fn(y, model_result)
    
    # Calcula e aplica gradiente
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # Atualiza métricas de treino na lista
    for train_metric in metrics_train:
        train_metric.update_state(y, model_result)
                
    return loss_value


@tf.function
def test_step(x, y, model, loss_fn, metrics_val):
    val_result = model(x, training=False)
    loss_value = loss_fn(y, val_result)    
    
    # Atualiza métricas de validação na lista
    for val_metric in metrics_val:
        val_metric.update_state(y, val_result)
        
    return loss_value

@tf.function
def vectorized_map(map_augment_function, tuple_x_y):
    return tf.vectorized_map(map_augment_function, tuple_x_y)

# @tf.function
# def vectorized_map(map_augment_function, tuple_x_y):
#     return tf.map_fn(map_augment_function, tuple_x_y)

# @tf.function
# def vectorized_map(map_augment_function, x, y):
#     list_transform =  list(map(map_augment_function, x, y))
#     return tf.stack([l0[0] for l0 in list_transform]), tf.stack([l1[1] for l1 in list_transform])



def train_model_loop(model, epochs, early_stopping_epochs, train_dataset, valid_dataset, optimizer, 
                    loss_fn, metrics_train=[], metrics_val=[], model_path='best_model.keras',
                    early_stopping_delta=0.01, data_augmentation=False, 
                    early_stopping_on_metric=True, augment_batch_factor=2):
    '''
    Treina modelo por um determinado número de épocas epochs, com early stopping de early_stopping_epochs, 
    salvando o modelo no diretório model_savedir confome a primeira métrica na lista metrics_val
    '''
    # Valor da perda e valor da métrica serão armazenados em listas
    history_train = [] 
    history_valid = []
    
    # Melhor valor de métrica, ou loss, até o momento no treinamento
    valid_metric_best_model = 1e-20
    valid_loss_best_model = float('inf')
    
    # Contagem de épocas sem melhora em valid_metric_best_model
    no_improvement_count = 0
    
    # Executa treinamento
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch))
        start_time = time.time()
        
        # Inicia loss acumulada da época no treino e validação
        train_loss = 0
        valid_loss = 0
        
        # Percorre batches do dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            if data_augmentation:
                # Compute new "batches" 
                x_batches_train_augmented = []
                y_batches_train_augmented = []
                for _ in range(augment_batch_factor-1):
                    '''
                    # Time compare
                    time_map_fn_start = time.time()
                    x_batch_train_augmented, y_batch_train_augmented = tf.map_fn(transform_augment, (x_batch_train, y_batch_train)) 
                    print("Time taken with tf.map_fn: %.2fs" % (time.time() - time_map_fn_start))
                    
                    time_vectorized_map_start = time.time()
                    x_batch_train_augmented, y_batch_train_augmented = tf.vectorized_map(transform_augment, (x_batch_train, y_batch_train))
                    print("Time taken with tf.vectorized_map: %.2fs" % (time.time() - time_vectorized_map_start))
                    
                    time_map_python = time.time()
                    list_transform = list(map(transform_augment_2arg, x_batch_train, y_batch_train))
                    x_batch_train_augmented, y_batch_train_augmented = tf.stack([l0[0] for l0 in list_transform]), \
                                                                       tf.stack([l1[1] for l1 in list_transform])
                    print("Time taken with map in Python: %.2fs" % (time.time() - time_map_python))
                    
                    time_vectorized_map_tffunction = time.time()
                    x_batch_train_augmented, y_batch_train_augmented = vectorized_map(transform_augment, (x_batch_train, y_batch_train))
                    print("Time taken with tf.vectorized_map embedded in a function decorated with @tf.function: %.2fs" % (time.time() - time_vectorized_map_tffunction))
                    '''
                    x_batch_train_augmented, y_batch_train_augmented = vectorized_map(transform_augment, (x_batch_train, y_batch_train))
                    x_batches_train_augmented.append(x_batch_train_augmented)
                    y_batches_train_augmented.append(y_batch_train_augmented)
                    
                # Concatenate original batch with new "batches"
                x_batch_train = tf.concat([x_batch_train] + x_batches_train_augmented, axis=0) 
                y_batch_train = tf.concat([y_batch_train] + y_batches_train_augmented, axis=0)                    
                
                # Delete computed variables
                del x_batch_train_augmented, y_batch_train_augmented, x_batches_train_augmented[:], y_batches_train_augmented[:]
                gc.collect()


                '''
                # X e Y Batch Espelhamento Vertical (Flip)
                x_batch_train_flip = tf.image.flip_up_down(x_batch_train)
                y_batch_train_flip = tf.image.flip_up_down(y_batch_train)
                
                # X e Y Espelhamento Horizontal (Mirror)
                x_batch_train_mirror = tf.image.flip_left_right(x_batch_train)
                y_batch_train_mirror = tf.image.flip_left_right(y_batch_train)
                
                # Rotação 90 graus
                x_batch_train_rot90 = tf.image.rot90(x_batch_train, k=1)
                y_batch_train_rot90 = tf.image.rot90(y_batch_train, k=1)
                
                # Rotação 180 graus
                x_batch_train_rot180 = tf.image.rot90(x_batch_train, k=2)
                y_batch_train_rot180 = tf.image.rot90(y_batch_train, k=2)
                
                # Rotação 270 graus
                x_batch_train_rot270 = tf.image.rot90(x_batch_train, k=3)
                y_batch_train_rot270 = tf.image.rot90(y_batch_train, k=3)
                
                # Espelhamento Vertical e Rotação 90 graus
                x_batch_train_flip_rot90 = tf.image.rot90(tf.image.flip_up_down(x_batch_train), k=1)
                y_batch_train_flip_rot90 = tf.image.rot90(tf.image.flip_up_down(y_batch_train), k=1) 
                
                # Espelhamento Vertical e Rotação 270 graus
                x_batch_train_flip_rot270 = tf.image.rot90(tf.image.flip_up_down(x_batch_train), k=3)
                y_batch_train_flip_rot270 = tf.image.rot90(tf.image.flip_up_down(y_batch_train), k=3)  
    
                # Concatenate tensors
                x_batch_train = tf.concat([x_batch_train, x_batch_train_flip, x_batch_train_mirror, 
                                           x_batch_train_rot90, x_batch_train_rot180, x_batch_train_rot270, 
                                           x_batch_train_flip_rot90, x_batch_train_flip_rot270], axis=0) 
                
                y_batch_train = tf.concat([y_batch_train, y_batch_train_flip, y_batch_train_mirror, 
                                           y_batch_train_rot90, y_batch_train_rot180, y_batch_train_rot270, 
                                           y_batch_train_flip_rot90, y_batch_train_flip_rot270], axis=0)
                '''


            # Train Step
            loss_value = train_step(x_batch_train, y_batch_train, model, loss_fn, optimizer, metrics_train)
            
            # Incrementa acumulador de loss
            train_loss = train_loss + loss_value            
                
            # Log every 200 batches
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * x_batch_train.shape[0])) # x_batch_train.shape[0] é o batch size final, já com aumento de dados
        
        
        # Adiciona loss acumulada/quantidade de batches   e a métrica acumulada (primeira da lista) para o histórico de treinao
        history_train.append(   np.array([[train_loss/(step+1), metrics_train[0].result()]])   )         
        
        # Exibe métricas no final de cada época
        for train_metric in metrics_train:
            result_train_metric = train_metric.result()            
            print(f"Training {train_metric.__class__.__name__.lower()} over epoch: {result_train_metric:.4f}")
            train_metric.reset_state() # Dá reset na métrica de treino no final da época
            
        # Computa validação no final de cada época
        for x_batch_val, y_batch_val in valid_dataset:
            # Validation step
            loss_value = test_step(x_batch_val, y_batch_val, model, loss_fn, metrics_val) 

            # Incrementa acumulador de loss da validação  
            valid_loss = valid_loss + loss_value


        # Adiciona loss acumulada/quantidade de batches   e a métrica acumulada (primeira da lista) para o histórico de validacao
        metrics_val0 = metrics_val[0].result()
        loss_val = valid_loss/(step+1)
        history_valid.append(   np.array([[loss_val, metrics_val0]])   )   

        # Zera loss de treino e validação acumuladas para a próxima época
        train_loss = 0
        valid_loss = 0
 
        # Exibe métricas de Validação no final de cada época e tempo gasto para época
        for val_metric in metrics_val:
            result_val_metric = val_metric.result()
            print(f"Validation {val_metric.__class__.__name__.lower()}: {result_val_metric:.4f}")
            val_metric.reset_state()
        print("Time taken: %.2fs" % (time.time() - start_time))
            
        
        # Early Stopping
        if early_stopping_epochs:
            # Early Stopping on Metric
            if early_stopping_on_metric:
                # Se valor absoluto da diferenca de incremento (ou decremento) for abaixo de early_stopping_delta,
                # então segue para contagem do Early Stopping
                diff = metrics_val0 - valid_metric_best_model
                if abs(diff) < early_stopping_delta:
                    # Stop if there are no improvement along early_stopping_epochs  
                    # This means, the situation above described persists for
                    # early_stopping_epochs
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Métrica diminuindo, pois metrics_val0 - valid_metric_best_model é menor que 0
                # Como isso normalmente é ruim, também segue para contagem do Early Stopping
                elif diff < 0:
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Métrica aumentando, pois metrics_val0 - valid_metric_best_model é maior que 0
                # Normalmente é uma coisa positiva, como por exemplo acurácia, precisão, recall aumentando.
                # Então salvamos o modelo e zeramos a contagem do Early Stopping
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
                        
                # Loss diminuindo, pois loss_val - valid_loss_best_model é menor que 0, o que é bom
                # Salva modelo e zera contagem de Early Stopping
                elif diff < 0:
                    valid_loss_best_model = loss_val
                    no_improvement_count = 0
        
                    # Saving best model  
                    print("Saving the model...")
                    model.save(model_path)
                       
                # Loss aumentando, pois loss_val - valid_loss_best_model é maior que 0, o que é ruim
                # Segue para contagem do Early Stopping
                else:
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
        # Sem Early Stopping, salva o modelo que tiver menor loss
        else:
            diff = loss_val - valid_loss_best_model
            if diff < 0 and abs(diff) < early_stopping_delta: 
                valid_loss_best_model = loss_val
    
                # Saving best model  
                print("Saving the model...")
                model.save(model_path)
            

    # Retorna lista de 2 listas
    # Primeira é com histórico de treino e segunda com histórico de validação
    # Cada elemento tem na primeira posição a loss e na segunda a métrica para a época correspondente à posição do elemento
    return [ history_train, history_valid ]





def train_unet(net, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, filepath, 
               filename, early_stopping=True, early_loss=False, metric_name='f1score', lr_decay=False, train_with_dataset=False, 
               tensorboard_log=False, custom_train=False):
    print('Start the training...')
    
    early_stop = None
    filepath_name = os.path.join(filepath, filename+'.keras')
    
    if custom_train:
        learning_rate = 0.001
        optimizer = Adam(learning_rate = learning_rate , beta_1=0.9) # Otimizador
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics_train = [F1Score(), tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)]
        metrics_val = [F1Score(), tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)]
        
        resultado = train_model_loop(net, epochs, early_stopping_epochs, train_dataset=x_train, valid_dataset=x_valid, optimizer=optimizer, 
                                    loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val, model_path=filepath_name)
        
        return resultado
    
    # Para log no tensorboard, precisa ter callback
    if tensorboard_log:
        log_dir = os.path.join(filepath, 'logs', 'fit') #, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        shutil.rmtree(log_dir)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # Com Learning Rate Decay
    if lr_decay:
        initial_lrate = 0.001
        drop = 0.1
        epochs_drop = 30.0
        print(f'Initial Learning Rate={initial_lrate}, Drop={drop}, Epochs Drop={epochs_drop}')        
        def step_decay(epoch):
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    
    # Com Early Stopping
    if early_stopping:
        # Early Stopping por Loss
        if early_loss:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                          min_delta=early_stopping_delta, verbose=1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_weights_only=False,
                                                              verbose=1,
                                                              save_freq='epoch',
                                                              save_best_only=True)
            
        # Early Stopping por Métrica
        else:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_'+metric_name, patience=early_stopping_epochs, mode='max', restore_best_weights=True,
                                                          min_delta=early_stopping_delta, verbose=1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name,
                                                              monitor='val_'+metric_name,
                                                              mode='max',
                                                              save_weights_only=False,
                                                              verbose=1,
                                                              save_freq='epoch',
                                                              save_best_only=True)
    
    # Sem Early Stopping salva apenas o último modelo
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name, 
                                                         monitor='val_loss',
                                                         mode='auto',
                                                         save_weights_only=False,
                                                         verbose=1,
                                                         save_freq='epoch',
                                                         save_best_only=False)
    
    # Constroi lista de callbacks
    if tensorboard_log:
        if lr_decay:
            if early_stop:
                callbacks = [early_stop, cp_callback, tensorboard_callback, lrate]
            else:
                callbacks = [cp_callback, tensorboard_callback, lrate]
        
        else:
            if early_stop:
                callbacks = [early_stop, cp_callback, tensorboard_callback]
            else:
                callbacks = [cp_callback, tensorboard_callback]
    else:
        if lr_decay:
            if early_stop:
                callbacks = [early_stop, cp_callback, lrate]
            else:
                callbacks = [cp_callback, lrate]
        
        else:
            if early_stop:
                callbacks = [early_stop, cp_callback]
            else:
                callbacks = [cp_callback]
        
        
    # Treina Modelo
    if train_with_dataset:
        history = net.fit(x=x_train, epochs=epochs, verbose='auto',
                          callbacks=callbacks, validation_data=x_valid) #, steps_per_epoch=5, validation_steps=5)
    else:
        history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose='auto',
                          callbacks=callbacks, validation_data=(x_valid, y_valid))
    
    # Retorna resultados de Treino e Validação
    historia = history.history
    
    lista_loss = historia['loss']
    lista_metric = historia[metric_name]
    lista_val_loss = historia['val_loss']
    lista_val_metric = historia['val_' + metric_name]
    
    history_train = [np.array([dupla]) for dupla in zip(lista_loss, lista_metric)]
    history_valid = [np.array([dupla]) for dupla in zip(lista_val_loss, lista_val_metric)]
    resultado = [ history_train, history_valid ]
    
    return resultado


def dice_loss(y_true, y_pred, smooth=1e-6):
    '''
    Dice Coef = (2*Inter)/(Union+Inter)
    Dice Loss = 1 - Dice Coef    
    '''
    # Cast y_true as float 32
    y_true = tf.cast(y_true, tf.float32)
    
    # Flatten Arrays        
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    
    # Intersection and Union
    intersection = K.sum(y_true_f * y_pred_f)
    union_plus_inter = K.sum(y_true_f) + K.sum(y_pred_f)
    
    # Compute Dice Loss
    dice_coef = (2. * intersection + smooth) / (union_plus_inter + smooth)
    #dice_loss = 1 - dice_coef
    dice_loss = -K.log(dice_coef)
    
    return dice_loss

        
def weighted_focal_loss(alpha, gamma=0):
    """
    compute focal loss according to the prob of the sample.
    loss= -(1-p)^gamma*log(p)

    Variables:
        gamma: integer
        alpha: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        gamma = 2 #The larger the gamma value, the less importance of easily classified. Typical values: 0 to 5
        alpha = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_focal_loss(gamma, alpha)
        model.compile(loss=focal_loss,optimizer='adam')
    """
    
    alpha = K.variable(alpha)
        
    def focal_loss(y_true, y_pred):
        print('Tipo y_true =', y_true.dtype)
        print('Tipo y_pred =', y_pred.dtype)
        print('Tipo alpha =', alpha.dtype)
        # Cast y_true as float 32
        y_true = tf.cast(y_true, tf.float32)
        
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred_inv = 1.0 - y_pred
        y_pred_inv = K.pow(y_pred_inv, gamma)
        # calc
        focal_loss = y_true * K.log(y_pred) * y_pred_inv * alpha
        focal_loss = K.mean(-K.sum(focal_loss, -1))
        return focal_loss
    
    return focal_loss


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    #weights = K.variable(weights)
    weights = tf.Variable(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        #loss = y_true * K.log(y_pred) * weights
        loss = y_true * tf.math.log(y_pred) * weights
        #loss = K.mean(-K.sum(loss, -1))
        loss = tf.reduce_mean(-tf.reduce_sum(loss, axis=-1))
        
        return loss

    return loss


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def transform_augment_or_maintain(x, y):
    # Sorteia opção
    lista_opcoes = [0, 1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
    # Mantém original
    if opcao == 0:
        return x, y
    # Espelhamento Vertical (Flip)
    elif opcao == 1:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
        return x, y
    # Espelhamento Horizontal (Mirror)
    elif opcao == 2:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
        return x, y
    # Rotação 90 graus
    elif opcao == 3:
        x = tf.image.rot90(x, k=1)
        y = tf.image.rot90(y, k=1)
        return x, y
    # Rotação 180 graus
    elif opcao == 4:
        x = tf.image.rot90(x, k=2)
        y = tf.image.rot90(y, k=2)
        return x, y
    # Rotação 270 graus
    elif opcao == 5:
        x = tf.image.rot90(x, k=3)
        y = tf.image.rot90(y, k=3)
        return x, y
    # Espelhamento Vertical e Rotação 90 graus
    elif opcao == 6:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=1)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=1)
        return x, y
    # Espelhamento Vertical e Rotação 270 graus
    elif opcao == 7:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=3)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=3)
        return x, y
    
    
def transform_augment(x_y):
    x, y = x_y
    
    # Sorteia opção
    lista_opcoes = [1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
    # Espelhamento Vertical (Flip)
    if opcao == 1:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
        return x, y
    # Espelhamento Horizontal (Mirror)
    elif opcao == 2:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
        return x, y
    # Rotação 90 graus
    elif opcao == 3:
        x = tf.image.rot90(x, k=1)
        y = tf.image.rot90(y, k=1)
        return x, y
    # Rotação 180 graus
    elif opcao == 4:
        x = tf.image.rot90(x, k=2)
        y = tf.image.rot90(y, k=2)
        return x, y
    # Rotação 270 graus
    elif opcao == 5:
        x = tf.image.rot90(x, k=3)
        y = tf.image.rot90(y, k=3)
        return x, y
    # Espelhamento Vertical e Rotação 90 graus
    elif opcao == 6:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=1)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=1)
        return x, y
    # Espelhamento Vertical e Rotação 270 graus
    elif opcao == 7:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=3)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=3)
        return (x, y)
    

def transform_augment_2arg(x, y):
    # Sorteia opção
    lista_opcoes = [1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
    # Espelhamento Vertical (Flip)
    if opcao == 1:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
        return x, y
    # Espelhamento Horizontal (Mirror)
    elif opcao == 2:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
        return x, y
    # Rotação 90 graus
    elif opcao == 3:
        x = tf.image.rot90(x, k=1)
        y = tf.image.rot90(y, k=1)
        return x, y
    # Rotação 180 graus
    elif opcao == 4:
        x = tf.image.rot90(x, k=2)
        y = tf.image.rot90(y, k=2)
        return x, y
    # Rotação 270 graus
    elif opcao == 5:
        x = tf.image.rot90(x, k=3)
        y = tf.image.rot90(y, k=3)
        return x, y
    # Espelhamento Vertical e Rotação 90 graus
    elif opcao == 6:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=1)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=1)
        return x, y
    # Espelhamento Vertical e Rotação 270 graus
    elif opcao == 7:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=3)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=3)
        return x, y
    



# Função para treinar o modelo conforme os dados (arrays numpy) em uma pasta de entrada, salvando o modelo e o 
# histórico na pasta de saída
def treina_modelo(input_dir: str, y_dir: str, output_dir: str, model_type: str ='resunet chamorro', epochs=150, early_stopping=True, 
                  early_loss=False, loss='cross', weights=[0.25, 0.75], gamma=2, metric=F1Score(), best_model_filename = 'best_model',
                  train_with_dataset=False, data_augmentation=False, lr_decay=False, custom_train=False):
    '''
    

    e talvez correction=False
    Parameters
    ----------
    input_dir : str
        Relative Path for Input Folder, e.g, r"entrada/".
    output_dir : str
        Relative Path for Output Folder, e.g, r"resultados/".
    model_type : str, optional
        DESCRIPTION. The default is 'resunet'.
     : TYPE
        DESCRIPTION.
    lr_decay=Dicionário, por exemplo, {initial_lrate = 0.1,
             drop = 0.5,
             epochs_drop = 10.0}, ou None, para sem learning rate decay 

    Returns
    -------
    None.

    '''
    # TODO
    # Completar código para train_with_dataset
    
    # Marca tempo total do treinamento. Começa Contagem
    start = time.time()
    
    # Lê arrays salvos
    '''
    if correction:
        x_train = np.load(input_dir + 'pred_train.npy')
        x_valid = np.load(input_dir + 'pred_valid.npy')
    else:
        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
    '''
    if train_with_dataset:
        # Load Datasets
        train_dataset = tf.data.Dataset.load(input_dir + 'train_dataset/')
        valid_dataset = tf.data.Dataset.load(input_dir + 'valid_dataset/')
        
        print('Shape dos arrays:')
        print('Shape X Train Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(train_dataset).next()[0].numpy().shape)
        print('Shape Y Train Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(train_dataset).next()[1].numpy().shape)
        print('Shape X Valid Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(valid_dataset).next()[0].numpy().shape)
        print('Shape Y Valid Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(valid_dataset).next()[1].numpy().shape)
        print('')
        
    else:
        '''
        with open(Path(y_dir / 'y_train.pickle'), 'rb') as fp:
            y_train = pickle.load(fp)
            
        with open(Path(y_dir / 'y_valid.pickle'), 'rb') as fp:
            y_valid = pickle.load(fp)

        with open(Path(y_dir / 'x_train.pickle'), 'rb') as fp:
            x_train = pickle.load(fp)
            
        with open(Path(y_dir / 'x_valid.pickle'), 'rb') as fp:
            x_valid = pickle.load(fp)
        '''        
        
        y_train = np.load(y_dir + 'y_train.npy')        
        # Faz codificação One-Hot dos Ys se necessário
        if y_train.shape[-1] == 1:
            y_train = to_categorical(y_train, num_classes=2)

        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        # with tf.device('/CPU:0'):
        #     y_train = tf.convert_to_tensor(y_train)

        gc.collect() 


        y_valid = np.load(y_dir + 'y_valid.npy')
        # Faz codificação One-Hot dos Ys se necessário
        if y_valid.shape[-1] == 1:
            y_valid = to_categorical(y_valid, num_classes=2)

        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        # with tf.device('/CPU:0'):
        #     y_valid = tf.convert_to_tensor(y_valid)
 
        gc.collect()        


        x_train = np.load(input_dir + 'x_train.npy')
        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        # with tf.device('/CPU:0'):
        #     x_train = tf.convert_to_tensor(x_train)

        gc.collect() 


        x_valid = np.load(input_dir + 'x_valid.npy')
        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        # with tf.device('/CPU:0'):
        #     x_valid = tf.convert_to_tensor(x_valid)

        gc.collect()  

        
        print('Shape dos arrays:')
        print('Shape x_train: ', x_train.shape)
        print('Shape y_train: ', y_train.shape)
        print('Shape x_valid: ', x_valid.shape)
        print('Shape y_valid: ', y_valid.shape)
        print('')
        
        '''
        y_train = np.load(y_dir + 'y_train.npy')
        y_valid = np.load(y_dir + 'y_valid.npy')

        # Faz codificação One-Hot dos Ys se necessário
        if y_train.shape[-1] == 1:
            y_train = to_categorical(y_train, num_classes=2)
            y_valid = to_categorical(y_valid, num_classes=2)

        gc.collect()        

        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
        
        print('Shape dos arrays:')
        print('Shape x_train: ', x_train.shape)
        print('Shape y_train: ', y_train.shape)
        print('Shape x_valid: ', x_valid.shape)
        print('Shape y_valid: ', y_valid.shape)
        print('')

        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        if not train_with_dataset:
            with tf.device('/CPU:0'):
                x_train = tf.convert_to_tensor(x_train)
                y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                x_valid = tf.convert_to_tensor(x_valid)
                y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float32)
        
        # Livra memória no que for possível
        gc.collect()
        '''
        
        

    
    # Constroi modelo
    if train_with_dataset:
        # input_shape = (patch_size, patch_size, image_channels)
        input_shape = iter(train_dataset).next()[0].numpy().shape
        # input_shape = (None,) + input_shape
        num_classes = 2
        model = build_model(input_shape, num_classes, model_type=model_type)
        # model.summary()
        print('Input Patch Shape = ', input_shape)
        print()
        
    else:
        # input_shape = (patch_size, patch_size, image_channels)
        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        num_classes = 2
        model = build_model(input_shape, num_classes, model_type=model_type)
        # model.summary()
        print('Input Patch Shape = ', input_shape)
        print()
        
    # Compila o modelo
    learning_rate = 0.001 # Learning Rate
    #learning_rate = 0.01 # Learning Rate # não estava convergindo na primeira rede
    optimizer = Adam(learning_rate = learning_rate , beta_1=0.9) # Otimizador
    #optimizer = SGD(learning_rate = learning_rate , momentum=0.9) # Otimizador

    if loss == 'focal':
        # focal_loss = CategoricalFocalCrossentropy(alpha=weights, gamma=gamma)
        focal_loss = weighted_focal_loss(alpha=weights, gamma=gamma)
        model.compile(loss = focal_loss, optimizer=optimizer , metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'cross':
        #model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=[metric])
        model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'mse':
        optimizer = SGD(learning_rate = learning_rate)
        model.compile(loss = 'mse', optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'dice':
        model.compile(loss = dice_loss, optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'jaccard':
        model.compile(loss = jaccard_distance_loss, optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'wcross':
        model.compile(loss = weighted_categorical_crossentropy(weights), optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
        
        
     # Definição dos Outros hiperparâmetros
    batch_size = 16
    
    epochs = epochs

    # Parâmetros do Early Stopping
    early_stopping = early_stopping
    early_stopping_epochs = 50
    #early_stopping_epochs = 2
    #early_stopping_delta = 0.001 # aumento delta (percentual de diminuição da perda) equivalente a 0.1%
    early_stopping_delta = 0.001 # aumento delta (percentual de diminuição da perda) equivalente a 0.1%

    # Preprocessa Dataset (com shuffle, batch e prefetch)
    if train_with_dataset:
        len_dataset = len(train_dataset)
        n_repeat = 1 # Inicializa n_repeat em 1 (vai manter o tamanho do dataset como está)
        if data_augmentation:
            # Aumenta dataset de Tamanho n_repeat vezes e sorteia transformação para cada elemento do dataset
            n_repeat = 2
            train_dataset = train_dataset.repeat(n_repeat).map(transform_augment_or_maintain, num_parallel_calls=tf.data.AUTOTUNE)
            
            """
            # Essa tentativa estourou memória
            # Aumenta Dataset de Tamanho
            # Original
            train_dataset = train_dataset
            
            # Espelhamento Vertical (Flip)
            train_dataset_flip = train_dataset.map(lambda x, y: (tf.image.flip_up_down(x), 
                                                                 tf.image.flip_up_down(y)))
            
            # Espelhamento Horizontal (Mirror)
            train_dataset_mirror = train_dataset.map(lambda x, y: (tf.image.flip_left_right(x), 
                                                                 tf.image.flip_left_right(y)))
            
            # Rotação 90 graus
            train_dataset_rot90 = train_dataset.map(lambda x, y: (tf.image.rot90(x, k=1), 
                                                                  tf.image.rot90(y, k=1)))
            
            # Rotação 180 graus 
            train_dataset_rot180 = train_dataset.map(lambda x, y: (tf.image.rot90(x, k=2), 
                                                                  tf.image.rot90(y, k=2)))
            
            # Rotação 270 graus 
            train_dataset_rot270 = train_dataset.map(lambda x, y: (tf.image.rot90(x, k=3), 
                                                                  tf.image.rot90(y, k=3)))
            
            # Espelhamento Vertical e Rotação 90 graus
            train_dataset_mirror_rot90 = train_dataset.map(lambda x, y: (tf.image.rot90(tf.image.flip_up_down(x), k=1), 
                                                                 tf.image.rot90(tf.image.flip_up_down(y), k=1)))
            
            # Espelhamento Vertical e Rotação 270 graus
            train_dataset_mirror_rot270 = train_dataset.map(lambda x, y: (tf.image.rot90(tf.image.flip_up_down(x), k=3), 
                                                                 tf.image.rot90(tf.image.flip_up_down(y), k=3)))
            
            
            
            # Concatena todos os datasets
            datasets = [train_dataset, train_dataset_flip, train_dataset_mirror, train_dataset_rot90, train_dataset_rot180,
                        train_dataset_rot270, train_dataset_mirror_rot90, train_dataset_mirror_rot270]
            train_dataset = tf.data.Dataset.from_tensor_slices(datasets)
            train_dataset = train_dataset.flat_map(lambda x: x)
            """
            
            
        shuffle_buffer = len_dataset*n_repeat # *8 # 100, 1000 seriam melhores valores para o buffer do shuffle (que trariam desempenho mais ou menos 
                                       # igual sem queda de desempenho)?
        train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer).batch(batch_size=batch_size).prefetch(buffer_size=1)
        valid_dataset = valid_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    print('Hiperparâmetros:')
    print('Modelo:', model_type)
    print('Batch Size:', batch_size)
    print('Epochs:', epochs)
    print('Early Stopping:', early_stopping)
    print('Early Stopping Epochs:', early_stopping_epochs)
    print('Early Stopping Delta:', early_stopping_delta)
    print('Otimizador:', 'Adam')
    print('Learning Rate:', learning_rate)
    print('Beta 1:', 0.9)
    print('Função de Perda:', loss)
    print('Gamma para Focal Loss:', gamma)
    print()
        
    # Nome do modelo a ser salvo
    best_model_filename = best_model_filename
    

    
    # Treina o modelo
    if train_with_dataset:
        # Testa se a métrica é string
        if isinstance(metric, str):
            history = train_unet(model, train_dataset, None, valid_dataset, None, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                         output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                         metric_name=metric, lr_decay=lr_decay, train_with_dataset=True, custom_train=custom_train)
        
        # Testa se é instância
        elif isinstance(metric, object):
            history = train_unet(model, train_dataset, None, valid_dataset, None, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                             output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                             metric_name=metric.__class__.__name__.lower(), lr_decay=lr_decay, train_with_dataset=True, custom_train=custom_train)
            
        # Testa se é função
        elif isinstance(metric, (types.FunctionType, types.BuiltinFunctionType)):
            history = train_unet(model, train_dataset, None, valid_dataset, None, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                             output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                             metric_name=metric.__name__, lr_decay=lr_decay, train_with_dataset=True, custom_train=custom_train)  
            
    else:
        # Testa se a métrica é string
        if isinstance(metric, str):
            history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                         output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                         metric_name=metric, lr_decay=lr_decay, train_with_dataset=False)
        
        # Testa se é instância
        elif isinstance(metric, object):
            history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                             output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                             metric_name=metric.__class__.__name__.lower(), lr_decay=lr_decay, train_with_dataset=False)
            
        # Testa se é função
        elif isinstance(metric, (types.FunctionType, types.BuiltinFunctionType)):
            history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                             output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                             metric_name=metric.__name__, lr_decay=lr_decay, train_with_dataset=False)                                                            

        

    # Imprime e salva história do treinamento
    print('history = \n', history)
    
    # Marca tempo total do treinamento. Encerra Contagem
    end = time.time()
    
    # Salva histórico e inclui no arquivo texto a contagem de tempo gasto
    with open(os.path.join(output_dir, 'history_' + best_model_filename + '.txt'), 'w') as f:
        f.write('history = \n')
        f.write(str(history))
        f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos')
        
    with open(os.path.join(output_dir, 'history_pickle_' +  best_model_filename + '.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
        pickle.dump(history, fp)
        
    # Mostra histórico em gráfico
    show_graph_loss_accuracy(history, metric_name='F1-Score', save=True, save_path=output_dir)
    
    # Escreve hiperparâmetreos e modelo usados no Diretório
    with open(os.path.join(output_dir, 'model_configuration_used.txt'), 'w') as f:
        f.writelines([f'Model Type = {model_type}\n', f'Batch Size = {batch_size}\n', f'Epochs = {epochs}\n',
                      f'Early Stopping Epochs = {early_stopping_epochs}\n', f'Early Stopping Delta = {early_stopping_delta}\n',
                      f'Learning Rate = {learning_rate}\n', f'Função de perda = {loss}\n', f'Optimizer = {str(type(optimizer))}\n'])
        if model_type == 'unet transformer':
            f.write(str(config_dict) + '\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        
        


class ModelTrainer:
    best_model_filename = 'best_model.keras'
    early_stopping_delta = 0.01 # Delta in relation to best result for training to continue 
    def __init__(self, x_dir: str, y_dir: str, output_dir: str, model, optimizer):
        # Directories
        self.x_dir = x_dir # Dir with X data
        self.y_dir = y_dir # Dir with Y data
        self.output_dir = output_dir # Dir to save Output data
        
        self.model = model # Model object
        
        self.optimizer = optimizer # Optimizer, has to be created outside class in order to be a singleton
        
        self.model_path = output_dir + self.best_model_filename # Path to save model
        
    def _set_datasets(self):
        # Load Datasets
        self.train_dataset = tf.data.Dataset.load(self.x_dir + 'train_dataset/')
        self.valid_dataset = tf.data.Dataset.load(self.x_dir + 'valid_dataset/')
        
    def _set_numpy_arrays(self, convert_to_tensor=False):
        # Load Y Train, do One-Hot encoding if necessary. If specified, convert to tensor and clean memory 
        self.y_train = np.load(self.y_dir + 'y_train.npy')        
        if self.y_train.shape[-1] == 1:
            self.y_train = to_categorical(self.y_train, num_classes=2)
        if convert_to_tensor:    
            with tf.device('/CPU:0'):
                self.y_train = tf.convert_to_tensor(self.y_train)
            gc.collect()
        
        # Load Y Valid
        self.y_valid = np.load(self.y_dir + 'y_valid.npy')
        if self.y_valid.shape[-1] == 1:
            self.y_valid = to_categorical(self.y_valid, num_classes=2)
        if convert_to_tensor:    
            with tf.device('/CPU:0'):
                self.y_valid = tf.convert_to_tensor(self.y_valid)
            gc.collect()
        
        # Load X Train
        self.x_train = np.load(self.x_dir + 'x_train.npy')
        if convert_to_tensor:  
            with tf.device('/CPU:0'):
                self.x_train = tf.convert_to_tensor(self.x_train)
            gc.collect() 
        
        # Load X Valid
        self.x_valid = np.load(self.x_dir + 'x_valid.npy')
        if convert_to_tensor:  
            with tf.device('/CPU:0'):
                self.x_valid = tf.convert_to_tensor(self.x_valid)
            gc.collect()  

    def train_with_loop(self, epochs=2000, early_stopping_epochs=50, 
                        metrics_train=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                        metrics_val=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                        learning_rate=0.001, loss_fn=CategoricalCrossentropy(from_logits=False),
                        buffer_shuffle=None, batch_size=16, data_augmentation=False,
                        early_stopping_on_metric=True,
                        augment_batch_factor=2):
        # Lists of metrics must not be empty
        assert len(metrics_train) > 0, "List of metrics on train must have at least one element"
        assert len(metrics_val) > 0, "List of metrics on validation must have at least one element"
        
        # Dictionary of parameters used when method is invoked
        dict_parameters = locals().copy()
        del dict_parameters['self']
        
        # Clone model
        model = tf.keras.models.clone_model(self.model)
        
        # Compute total time to train. Begin to count
        start = time.time()
        
        # augment_batch_factor * maintained_batch_size = batch_size
        # Here we consider the batch size as the final batch size, after data augmentation on the batch
        if data_augmentation:
            assert batch_size % augment_batch_factor == 0, "Batch size must be divisible by the augment " \
                                                           "factor of the batch when doing data augmentation"
            batch_size = batch_size // augment_batch_factor
        
        # Set datasets
        self._set_datasets()
        
        # Optimizer and learning rate
        optimizer = self.optimizer
        optimizer.build(model.trainable_variables)
        optimizer.learning_rate.assign(learning_rate) # Assign learning rate to optimizer
        
        # By default, shuffle dataset by its length
        if not buffer_shuffle:
            train_dataset = self.train_dataset.shuffle(len(self.train_dataset)).batch(batch_size)
            valid_dataset = self.valid_dataset.shuffle(len(self.valid_dataset)).batch(batch_size)
        else:
            train_dataset = self.train_dataset.shuffle(buffer_shuffle).batch(batch_size)
            valid_dataset = self.valid_dataset.shuffle(buffer_shuffle).batch(batch_size)
            
        # Run loop of train
        result_history = train_model_loop(model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
                                  train_dataset=train_dataset, valid_dataset=valid_dataset,
                                  optimizer=optimizer, loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val,
                                  model_path=self.model_path, early_stopping_delta=self.early_stopping_delta,
                                  data_augmentation=data_augmentation, early_stopping_on_metric=True,                                  
                                  augment_batch_factor=augment_batch_factor)
        
        # Delete datasets to clean object
        del self.train_dataset, self.valid_dataset
        
        # End the counting of time
        end = time.time()
        
        # Save history and total time in text and in pickle file
        with open(os.path.join(self.output_dir, 'history_best_model.txt'), 'w') as f:
            f.write('Resultado = \n')
            f.write(str(result_history))
            f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos, {(end-start)/3600:.1f} horas.')
            
        with open(os.path.join(self.output_dir, 'history_pickle_best_model.pickle'), "wb") as fp: 
            pickle.dump(result_history, fp)
            
        # Use first metric of list of validation metrics to plot history
        metric0 = metrics_val[0]
        if isinstance(metric0, str): # Metric is string
            metric_name = metric0
        elif isinstance(metric0, object): # Metric is instance
            metric_name = metric0.__class__.__name__.lower()
        elif isinstance(metric0, (types.FunctionType, types.BuiltinFunctionType)): # Metric is function
            metric_name = metric0.__name__ 
            
        # Save output in plot
        show_graph_loss_accuracy(result_history, metric_name=metric_name, save=True, save_path=self.output_dir)
        
        # Write model summary of model used in text file and list of arguments to the method
        with open(os.path.join(self.output_dir, 'model_configuration_used.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(str(dict_parameters) + '\n')
        
        return result_history
    
    def train_with_fit(self, epochs=2000, early_stopping_epochs=50, 
                       metrics=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                       learning_rate=0.001, loss_fn=CategoricalCrossentropy(from_logits=False),
                       buffer_shuffle=None, batch_size=16, use_dataset=True, convert_to_tensor_if_numpy=False,
                       data_augmentation=False, early_stopping_on_metric=True,  
                       n_repeat=None, 
                       tensorboard_log=False, lr_decay_dict=None, **kwargs):
        # Lists of metrics must not be empty
        assert len(metrics) > 0, "List of metrics during train must have at least one element"
        
        # Dictionary of parameters used when method is invoked
        dict_parameters = locals().copy()
        del dict_parameters['self']
        
        # Clone model
        model = tf.keras.models.clone_model(self.model)
        
        # Compute total time to train. Begin to count
        start = time.time()
        
        # Set datasets or numpy arrays
        if use_dataset:
            # Set datasets
            self._set_datasets()
            if not buffer_shuffle:
                train_dataset = self.train_dataset.shuffle(len(self.train_dataset)).batch(batch_size)
                valid_dataset = self.valid_dataset.shuffle(len(self.valid_dataset)).batch(batch_size)
            else:
                train_dataset = self.train_dataset.shuffle(buffer_shuffle).batch(batch_size)
                valid_dataset = self.valid_dataset.shuffle(buffer_shuffle).batch(batch_size)
                
            # Data augmentation to dataset if specified, shuffle and batch dataset
            len_dataset = len(train_dataset)
            if not n_repeat:
                n_repeat = 1 # Set n_repeat to 1 if it is None (keep same dataset size)
            if not buffer_shuffle:
                buffer_shuffle = n_repeat*len_dataset # Set shuffle buffer to length of dataset (augmented or not) if it is None 
            if data_augmentation:
                train_dataset = self.train_dataset.repeat(n_repeat).map(transform_augment_or_maintain, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                train_dataset = self.train_dataset.repeat(n_repeat)
            train_dataset = train_dataset.shuffle(buffer_size=buffer_shuffle).batch(batch_size=batch_size).prefetch(buffer_size=1)
            valid_dataset = self.valid_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)                
        else:
            self._set_numpy_arrays(convert_to_tensor=convert_to_tensor_if_numpy)
            
        # Tensorboard callback
        if tensorboard_log:
            log_dir = os.path.join(self.output_dir, 'logs', 'fit') #, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            
        # Apply Learning Rate Decay if specified
        # Example: lr_decay_dict = {'initial_lrate':0.001, 'drop':0.1, 'epochs_drop':30}
        if lr_decay_dict:
            print(f'Initial Learning Rate={lr_decay_dict["initial_lrate"]}, Drop={lr_decay_dict["drop"]}, Epochs Drop={lr_decay_dict["epochs_drop"]}')        
            def step_decay(epoch):
                lrate = lr_decay_dict["initial_lrate"] * math.pow(lr_decay_dict["drop"], math.floor( (1+epoch) / lr_decay_dict["epochs_drop"] )  )
                return lrate
            lrate_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)
            
        # Early Stopping and Model Checkpoint Callbacks
        # If Early Stopping is defined, use first metric of list of metrics to Early Stopping
        metric0 = metrics[0]
        if isinstance(metric0, str): # Metric is string
            metric_name = metric0
        elif isinstance(metric0, object): # Metric is instance
            metric_name = metric0.__class__.__name__.lower()
        elif isinstance(metric0, (types.FunctionType, types.BuiltinFunctionType)): # Metric is function
            metric_name = metric0.__name__ 
        if early_stopping_epochs:
            # Early Stopping on Metric
            if early_stopping_on_metric:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_'+metric_name, patience=early_stopping_epochs, mode='max', restore_best_weights=True,
                                                              min_delta=self.early_stopping_delta, verbose=1)                
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                                  monitor='val_'+metric_name,
                                                                  mode='max',
                                                                  save_weights_only=False,
                                                                  verbose=1,
                                                                  save_freq='epoch',
                                                                  save_best_only=True)
            # Early Stopping on Loss
            else:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                              min_delta=self.early_stopping_delta, verbose=1)                
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                                  monitor='val_loss',
                                                                  mode='min',
                                                                  save_weights_only=False,
                                                                  verbose=1,
                                                                  save_freq='epoch',
                                                                  save_best_only=True)
        # Without Early Stopping, salve the best model, which is the model with minimum loss in validation data
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, 
                                                             monitor='val_loss',
                                                             mode='min',
                                                             save_weights_only=False,
                                                             verbose=1,
                                                             save_freq='epoch',
                                                             save_best_only=True)
            
        # Build list of callbacks
        if tensorboard_log:
            if lr_decay_dict:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback, tensorboard_callback, lrate_scheduler]
                else:
                    callbacks = [cp_callback, tensorboard_callback, lrate_scheduler]
            
            else:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback, tensorboard_callback]
                else:
                    callbacks = [cp_callback, tensorboard_callback]
        else:
            if lr_decay_dict:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback, lrate_scheduler]
                else:
                    callbacks = [cp_callback, lrate_scheduler]
            
            else:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback]
                else:
                    callbacks = [cp_callback]

        # Optimizer and learning rate
        optimizer = self.optimizer
        optimizer.build(model.trainable_variables)
        optimizer.learning_rate.assign(learning_rate) # Assign learning rate to optimizer
            
        # Compile model
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
                    
        # Train Model
        if use_dataset:
            historia = model.fit(x=train_dataset, epochs=epochs, verbose='auto',      # for fast debug run in only 5 batches:
                                 callbacks=callbacks, validation_data=valid_dataset)  #, steps_per_epoch=5, validation_steps=5) 
                                                                               
        else:
            historia = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose='auto',
                                 callbacks=callbacks, validation_data=(self.x_valid, self.y_valid))
            
        # Transform result in list of 2 lists (train and validation)
        history = historia.history
        
        list_loss = history['loss']
        list_metric = history[metric_name]
        list_val_loss = history['val_loss']
        list_val_metric = history['val_' + metric_name]
        
        history_train = [np.array([dupla]) for dupla in zip(list_loss, list_metric)]
        history_valid = [np.array([dupla]) for dupla in zip(list_val_loss, list_val_metric)]
        result_history = [ history_train, history_valid ]
            
        # Delete datasets or numpy arrays to clean object
        del self.train_dataset, self.valid_dataset
        
        # End the counting of time
        end = time.time()
        
        # Save history and total time in text and in pickle file
        with open(os.path.join(self.output_dir, 'history_best_model.txt'), 'w') as f:
            f.write('Resultado = \n')
            f.write(str(result_history))
            f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos, {(end-start)/3600:.1f} horas.')
            
        with open(os.path.join(self.output_dir, 'history_pickle_best_model.pickle'), "wb") as fp: 
            pickle.dump(result_history, fp)
            
        # Save output in plot
        show_graph_loss_accuracy(result_history, metric_name=metric_name, save=True, save_path=self.output_dir)
        
        # Write model summary of model used in text file and list of arguments to the method
        with open(os.path.join(self.output_dir, 'model_configuration_used.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(str(dict_parameters) + '\n')
        
        return result_history
        
    
    
        




def treina_modelo_ensemble(input_dir: str, output_dir_ensemble: str, n_members: int = 10, model_type: str ='resunet', epochs=150, 
                           early_stopping=True, early_loss=True, loss='focal', gamma=2, metric=F1Score(), 
                           best_model_filename = 'best_model'):
    # Loop para treinar vários modelos diversas vezes
    for i in range(n_members):
        treina_modelo(input_dir, output_dir_ensemble, model_type=model_type, epochs=epochs, early_stopping=early_stopping, 
                      early_loss=early_loss, loss=loss, gamma=gamma, metric=metric, 
                      best_model_filename=best_model_filename + '_' + str(i+1))
        

def transfer_learning_segformer(input_dir, y_dir, output_dir, model_checkpoint, model_weights_filename, learning_rate=0.001, epochs=2000, 
                                early_stopping_epochs=50, batch_size=16):
    # Marca tempo total do treinamento. Começa Contagem
    start = time.time()
    
    # Load Arrays
    x_train = np.load(input_dir + 'x_train_fine.npy')
    x_valid = np.load(input_dir + 'x_valid_fine.npy')
    y_train = np.load(y_dir + 'y_train_fine.npy')
    y_valid = np.load(y_dir + 'y_valid_fine.npy')
    
    # Convert them to tensors
    with tf.device('/CPU:0'):
        x_train = tf.convert_to_tensor(x_train)
        x_valid = tf.convert_to_tensor(x_valid)
        y_train = tf.convert_to_tensor(y_train)
        y_valid = tf.convert_to_tensor(y_valid)
        
    # Livra memória no que for possível
    gc.collect()  
    
    # Load Model
    id2label = {0: "background", 1: "road"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Freeze Encoder
    model.layers[0].trainable = False
    
    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    
    # Build callbacks
    # Com Early Stopping
    # Early Stopping por Loss
    early_stopping_delta = 0.001
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                  min_delta=early_stopping_delta, verbose=1)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(Path(output_dir) / model_weights_filename),
                                                      monitor='val_loss',
                                                      mode='min',
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      save_freq='epoch',
                                                      save_best_only=True)    
    callbacks = [early_stop, cp_callback]
    
    
    # Fit Model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        callbacks=callbacks,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Retorna resultados de Treino e Validação
    historia = history.history
    
    lista_loss = historia['loss']
    lista_metric = [0]*len(lista_loss)
    lista_val_loss = historia['val_loss']
    lista_val_metric = [0]*len(lista_loss)
    
    history_train = [np.array([dupla]) for dupla in zip(lista_loss, lista_metric)]
    history_valid = [np.array([dupla]) for dupla in zip(lista_val_loss, lista_val_metric)]
    history = [ history_train, history_valid ]
    
    # Imprime e salva história do treinamento
    print('history = \n', history)
    
    # Marca tempo total do treinamento. Encerra Contagem
    end = time.time()
    
    # Salva histórico e inclui no arquivo texto a contagem de tempo gasto
    with open(os.path.join(output_dir, 'history_' + Path(model_weights_filename).stem + '.txt'), 'w') as f:
        f.write('history = \n')
        f.write(str(history))
        f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos')
        
    with open(os.path.join(output_dir, 'history_pickle_' +  Path(model_weights_filename).stem + '.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
        pickle.dump(history, fp)
        
    # Mostra histórico em gráfico
    show_graph_loss_accuracy(history, metric_name='F1-Score', save=True, save_path=output_dir)
    
    # Escreve hiperparâmetreos e modelo usados no Diretório
    with open(os.path.join(output_dir, 'model_configuration_used.txt'), 'w') as f:
        f.writelines([f'Model Type = {type(model)}\n', f'Batch Size = {batch_size}\n', f'Epochs = {epochs}\n',
                      f'Early Stopping Epochs = {early_stopping_epochs}\n', f'Early Stopping Delta = {early_stopping_delta}\n',
                      f'Learning Rate = {learning_rate}\n', 'Função de perda = Loss do Modelo\n', f'Optimizer = {str(type(optimizer))}\n'])
        model.summary(print_fn=lambda x: f.write(x + '\n'))
                
        
    return history

        



