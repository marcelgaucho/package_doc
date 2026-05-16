# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:14:19 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np
import tensorflow as tf

# %%

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        # Initialize internal tracking variables
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def check_improvement(self, current_metric):
        """
        Returns True if the current score is a significant improvement 
        based on the mode and min_delta.
        """
        if self.mode == 'max':
            return current_metric > (self.best_metric + self.min_delta)
        else: # mode == 'min'
            return current_metric < (self.best_metric - self.min_delta)        
    
    def step(self, current_metric, model, model_path):
            """
            Updates tracking and determines if training should stop.
            Saves the model on improvement.
            Returns True to trigger a 'break' in the training loop.
            """
            if self.patience is None or self.patience == 0:
                # Early stopping is disabled; always save the absolute best model
                if (self.mode == 'max' and current_metric > self.best_metric) or \
                   (self.mode == 'min' and current_metric < self.best_metric):
                    self.best_metric = current_metric
                    print("✓ Absolute best model updated and saved.")
                    model.save(model_path)
                return False
    
            if self.check_improvement(current_metric):
                self.best_metric = current_metric
                self.wait = 0
                print("✓ Improvement found. Model saved.")
                model.save(model_path)
            else:
                self.wait += 1
                print(f"× No improvement. Patience: {self.wait}/{self.patience}")
                if self.wait >= self.patience:
                    print("!!! Early Stopping Triggered !!!")
                    return True
                    
            return False
            