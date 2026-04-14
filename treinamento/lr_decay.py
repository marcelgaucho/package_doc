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