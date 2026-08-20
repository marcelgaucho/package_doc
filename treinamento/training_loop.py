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
from .early_stopping import EarlyStopping
import time
import pdb

# %%

def train_model_loop(model, epochs, early_stopping_epochs, train_dataset, valid_dataset, optimizer, 
                     loss_fn, use_uce=False, metrics_train=[], metrics_val=[], model_path='best_model.keras',
                     early_stopping_delta=0.01, lr_strategy=None, mode='max'):
    
    # 1. Setup Tracking
    history_train, history_valid = [], []
    
    # Use Keras metrics for mean loss to handle varying batch sizes automatically
    train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
    val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
    
    # Instantiate Early Stopping and Reduce on Plateau classes
    early_stopper = EarlyStopping(patience=early_stopping_epochs, min_delta=early_stopping_delta, mode=mode)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} | LR: {optimizer.learning_rate.numpy():.6f}")
        start_time = time.time() # Starts epoch training time

        # --- TRAINING ---
        for step, batches in enumerate(train_dataset):
            # Dynamic unpacking
            x_batch, y_batch = batches[0], batches[1]
            e_batch = batches[2] if len(batches) == 3 else None

            # Execution
            loss_value = train_step(x_batch, y_batch, model, loss_fn, optimizer, metrics_train, e_batch, use_uce)
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

        # =====================================================================
        # BLOCO DE DIAGNÓSTICO DO F1-SCORE (ADAPTADO PARA SOFTMAX)
        # =====================================================================
        # 1. Captura as probabilidades brutas vindas do Softmax (Shape: 15, 256, 256, 2)
        softmax_preds = model(x_v, training=False).numpy()

        # 2. Transforma as probabilidades no mapa de classes final (Shape: 15, 256, 256)
        # O pixel vira '1' se o canal 1 for maior que o canal 0, caso contrário vira '0'
        preds_classes = np.argmax(softmax_preds, axis=-1)

        # 3. Converte as labels reais para numpy e garante o formato correto de classe
        y_true_np = y_v.numpy() if hasattr(y_v, 'numpy') else np.array(y_v)

        # Se o seu gabarito também estiver no formato One-Hot (15, 256, 256, 2), extrai as classes dele também
        if len(y_true_np.shape) == 4 and y_true_np.shape[-1] == 2:
            y_true_classes = np.argmax(y_true_np, axis=-1)
        else:
            y_true_classes = y_true_np 

        # 4. Exibe o Raio-X interpretando os canais do Softmax
        print(f"\n" + "="*50)
        print(f"[RAIO-X DE DIAGNÓSTICO SOFTMAX | ÉPOCA {epoch+1}]")
        print(f" -> Formato original da saída: {softmax_preds.shape}")
        print(f" -> Formato após o Argmax (Classes): {preds_classes.shape}")
        print(f" -> Confiança média do Canal 0 (Fundo): {softmax_preds[..., 0].mean():.4f}")
        print(f" -> Confiança média do Canal 1 (Objeto): {softmax_preds[..., 1].mean():.4f}")
        print("-" * 50)
        print(f" -> Total de pixels da 'Classe 1' REAIS no lote: {np.sum(y_true_classes == 1)}")
        print(f" -> Total de pixels da 'Classe 1' PREVISTOS pelo modelo: {np.sum(preds_classes == 1)}")
        print(f" -> Total de pixels da 'Classe 0' PREVISTOS pelo modelo: {np.sum(preds_classes == 0)}")
        print("="*50 + "\n")
        # =====================================================================

        # --- SCHEDULERS & EARLY STOPPING ---
        monitor_value = current_val_metric if mode == 'max' else current_val_loss
        if lr_strategy:
            lr_strategy.on_epoch_end(monitor_value)

        # Improvement Logic with Early Stopping
        if early_stopper.step(current_val_loss, current_val_metric, model, model_path):
            break

        val_loss_tracker.reset_state()
        for m in metrics_val: m.reset_state()

    # Return a dictionary containing histories AND the scalar best records
    return {
            "history_train": history_train,
            "history_valid": history_valid,
            "best_loss": early_stopper.best_loss,
            "best_metric": early_stopper.best_metric
        }