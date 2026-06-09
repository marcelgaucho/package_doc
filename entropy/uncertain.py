# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 22:50:19 2025

@author: Marcel
"""

# %% Import Libraries

from pathlib import Path
import numpy as np
from .utils import scale_array, DataGroups

# %%

class UncertaintyCalculator:
    ''' Calculate uncertainty metric for an ensemble of predicted probabilities of shape (B, H, W, 1) '''
    def __init__(self, model_dirs: list[str], data_group: str = DataGroups.Test, scale_result=True):
        self.model_dirs = [Path(model_dir) for model_dir in model_dirs]
        self.data_group = data_group
        self.scale_result = scale_result
        self.epsilon = 1e-7 
        
        # Pre-compute the mean ensemble probabilities
        self.prob_mean_ensemble = self._prob_mean_ensemble()
        
    def _prob_mean_ensemble(self):
        # Stack into shape (N_models, ..., 1)
        probs = np.stack([
            np.load(d / f'prob_{self.data_group.value}.npy') 
            for d in self.model_dirs
        ], axis=0)
        
        # Calculate mean across models: (..., 1)
        mean_p1 = np.mean(probs, axis=0)
        # Create full distribution: (..., 2) where [background, foreground]
        return np.concatenate([1 - mean_p1, mean_p1], axis=-1)

    def _finalize(self, values, min_scale, max_scale, perc_cut):
        """Helper to handle the repeated scaling logic."""
        if self.scale_result:
            return scale_array(
                values, 
                min_target_scale=min_scale, 
                max_target_scale=max_scale, 
                perc_cut=perc_cut
            )
        return values

    def entropy(self, min_target_scale=0, max_target_scale=1, perc_cut=None):
        p = self.prob_mean_ensemble
        # H = -sum(p * log2(p))
        entropy = -np.sum(p * np.log2(p + self.epsilon), axis=-1, keepdims=True)
        return self._finalize(entropy, min_target_scale, max_target_scale, perc_cut)
        
    def surprise(self, min_target_scale=0, max_target_scale=1, perc_cut=None):    
        p1 = self.prob_mean_ensemble[..., 1:2]
        # I(x) = -log2(P(x))
        surprise = -np.log2(p1 + self.epsilon)
        return self._finalize(surprise, min_target_scale, max_target_scale, perc_cut)
    
    def weighted_surprise(self, min_target_scale=0, max_target_scale=1, perc_cut=None):    
        p1 = self.prob_mean_ensemble[..., 1:2]
        # Partial entropy for the object class
        weighted = -p1 * np.log2(p1 + self.epsilon)
        return self._finalize(weighted, min_target_scale, max_target_scale, perc_cut)
    
    def prob_mean(self, min_target_scale=0, max_target_scale=1, perc_cut=None):
        p1 = self.prob_mean_ensemble[..., 1:2]
        return self._finalize(p1, min_target_scale, max_target_scale, perc_cut)
    
    def std_dev(self, min_target_scale=0, max_target_scale=1, perc_cut=None):
        # Load raw foreground probabilities: shape (N_models, B, H, W, 1)
        probs = np.stack([
            np.load(d / f'prob_{self.data_group.value}.npy') 
            for d in self.model_dirs
        ], axis=0)
        
        # Calculate SD across the model axis. Result is (B, H, W, 1)
        sd = np.std(probs, axis=0, ddof=1) 
        
        return self._finalize(sd, min_target_scale, max_target_scale, perc_cut)
        
        


