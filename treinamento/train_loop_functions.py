# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:18:54 2024

@author: Marcel
"""

import time, gc
import numpy as np
import tensorflow as tf
from .augment_functions import transform_augment

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