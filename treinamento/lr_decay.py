#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:58:10 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# %% Step Decay Class

class StepDecay(LearningRateSchedule):
    def __init__(self, initial_lr, steps_per_epoch, drop_rate=0.1, epochs_per_drop=10):
        # Initialize parent class
        super().__init__()
        
        # Define parameters
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.steps_per_epoch = tf.cast(steps_per_epoch, tf.float32)
        self.drop_rate = tf.cast(drop_rate, tf.float32)
        self.epochs_per_drop = tf.cast(epochs_per_drop, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # 1. Determine current epoch
        current_epoch = tf.floor(step / self.steps_per_epoch)
        
        # 2. Determine how many "decay intervals" (e.g., 10-epoch blocks) have passed
        drop_count = tf.floor(current_epoch / self.epochs_per_drop)
        
        # 3. Calculate new LR: initial_lr * (drop_rate ^ drop_count)
        new_lr = self.initial_lr * tf.pow(self.drop_rate, drop_count)
         
        return new_lr
    
# %% Reduce on plateau class

class ReduceOnPlateau:
    def __init__(self, optimizer, decay_factor=0.5,
                 patience=5, min_lr=1e-6, min_delta=0.001,
                 mode='max'):
        self.optimizer = optimizer
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        
    def step(self, current_metric):
        """Call this at the end of every epoch with your validation metric."""
        improved = (current_metric > (self.best_metric + self.min_delta)) if self.mode == 'max' else \
                   (current_metric < (self.best_metric - self.min_delta))
        
        # Check if the new metric is better than the best seen so far
        if self.mode == 'max':
            if current_metric > self.best_metric:
                improved = True
        else:
            if current_metric < self.best_metric:
                improved = True
        
        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0 # Reset counter to give the new LR time to work
    
    def _reduce_lr(self):
        old_lr = self.optimizer.learning_rate.numpy()
        new_lr = max(old_lr * self.factor, self.min_lr)
        
        # Update the optimizer's LR
        self.optimizer.learning_rate.assign(new_lr)
        print(f"\n[LR Decay] No improvement for {self.patience} epochs. "
              f"Reducing LR: {old_lr:.6f} -> {new_lr:.6f}")