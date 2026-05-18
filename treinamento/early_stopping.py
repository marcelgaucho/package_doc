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
        
        # Performance Tracking
        self.best_monitored_score = -float('inf') if mode == 'max' else float('inf')
        self.wait = 0
        
        # Store the absolute best metrics achieved
        self.best_loss = None
        self.best_metric = None

    def check_improvement(self, current_monitored, use_delta=True):
        """
        Returns True if the current score is a significant improvement 
        based on the mode and delta.
        """
        delta = self.min_delta if use_delta else 0.0
        if self.mode == 'max':
            return current_monitored > (self.best_monitored_score + delta)
        else: # mode == 'min'
            return current_monitored < (self.best_monitored_score - delta) 
        
    def step(self, current_loss, current_metric, model, model_path):
        """
        Updates tracking and determines if training should stop.
        Saves the model on improvement.
        Returns True to trigger a 'break' in the training loop.
        """
        # Determine what we are actually monitoring (Loss or Metric)
        current_monitored = current_metric if self.mode == 'max' else current_loss
        
        # Case 1: Early stopping is disabled (Patience = 0 or None)
        if self.patience is None or self.patience == 0:
            # Always save the absolute best model if early stopping is disabled
            if self.check_improvement(current_monitored, use_delta=False):
                self.best_monitored_score = current_monitored
                self.best_loss = current_loss      # Save loss snapshot
                self.best_metric = current_metric  # Save metric snapshot
                print("[Early Stopping] ✓ Absolute best model updated and saved.")
                model.save(model_path)
            return False

        # Case 2: Early stopping is active
        if self.check_improvement(current_monitored, use_delta=True):
            self.best_monitored_score = current_monitored
            self.best_loss = current_loss          # Save loss snapshot
            self.best_metric = current_metric      # Save metric snapshot
            self.wait = 0
            print("[Early Stopping] ✓ Significant improvement found. Model saved.")
            model.save(model_path)
        else:
            self.wait += 1
            print(f"[Early Stopping] × No significant improvement. Patience: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                print("[Early Stopping] !!! Early Stopping Triggered !!!")
                return True
                
        return False



'''
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        # Performance Tracking
        self.best_monitored_score = -float('inf') if mode == 'max' else float('inf')
        self.wait = 0
        
        # --- NEW: Storage for the absolute best metrics achieved ---
        self.best_loss = None
        self.best_metric = None
        
    def check_improvement(self, current_monitored, use_delta=True):
        delta = self.min_delta if use_delta else 0.0
        if self.mode == 'max':
            return current_monitored > (self.best_monitored_score + delta)
        else: # mode == 'min'
            return current_monitored < (self.best_monitored_score - delta)        
    
    def step(self, current_loss, current_metric, model, model_path):
        """
        Accepts BOTH current loss and current metric to keep track of the best of both.
        """
        # Determine what we are actually monitoring (Loss or Metric)
        current_monitored = current_metric if self.mode == 'max' else current_loss

        # Case 1: Early stopping is disabled (Patience = 0 or None)
        if self.patience is None or self.patience == 0:
            if self.check_improvement(current_monitored, use_delta=False):
                self.best_monitored_score = current_monitored
                self.best_loss = current_loss      # Save snapshot
                self.best_metric = current_metric  # Save snapshot
                print("✓ Absolute best model updated and saved.")
                model.save(model_path)
            return False
        
        # Case 2: Early stopping is active
        if self.check_improvement(current_monitored, use_delta=True):
            self.best_monitored_score = current_monitored
            self.best_loss = current_loss          # Save snapshot
            self.best_metric = current_metric      # Save snapshot
            self.wait = 0
            print("✓ Significant improvement found. Model saved.")
            model.save(model_path)
        else:
            self.wait += 1
            print(f"× No significant improvement. Patience: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                print("!!! Early Stopping Triggered !!!")
                return True
                
        return False
'''
