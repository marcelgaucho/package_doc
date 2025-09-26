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

class Ensemble:
    def __init__(self, model_dirs: list[str], data_group: str = DataGroups.Test):
        self.model_dirs = [Path(model_dir) for model_dir in model_dirs]
        self.data_group = data_group
        
        self.prob_mean_ensemble = self._prob_mean_ensemble()
        
        self.epsilon = 1e-7 # Used to manage log 0
        
    def _prob_mean_ensemble(self):
        prob_ensemble = [np.load(model_dir / f'prob_{self.data_group}.npy') 
                         for model_dir in self.model_dirs] # Load prob arrays in list
        prob_ensemble = np.array(prob_ensemble) # transform list in array
        prob_ensemble = np.concatenate((1-prob_ensemble, prob_ensemble), axis=-1) # insert the background probability
        prob_mean_ensemble = np.mean(prob_ensemble, axis=0)
        
        return prob_mean_ensemble
    
    def entropy(self, min_target_scale=0, max_target_scale=1, perc_cut=None):
        epsilon = self.epsilon
        
        prob_ensemble = self.prob_mean_ensemble
        
        entropy = -np.sum(prob_ensemble * np.log2(prob_ensemble + epsilon), axis=-1, keepdims=True)
        
        entropy_scaled = scale_array(entropy, min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                     perc_cut=perc_cut)
        
        return entropy_scaled
        
    def surprise(self, min_target_scale=0, max_target_scale=1, perc_cut=None):    
        epsilon = self.epsilon
        
        prob_ensemble = self.prob_mean_ensemble[..., 1:2] # Surprise is for object class (class 1)
        
        surprise = -np.log2(prob_ensemble + epsilon)
        
        surprise_scaled = scale_array(surprise, min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                      perc_cut=perc_cut)
        
        return surprise_scaled
    
    def weighted_surprise(self, min_target_scale=0, max_target_scale=1, perc_cut=None):    
        epsilon = self.epsilon
        
        prob_ensemble = self.prob_mean_ensemble[..., 1:2] # Surprise is for object class (class 1)

        weighted_surprise = -prob_ensemble * np.log2(prob_ensemble + epsilon)
        
        weighted_surprise_scaled = scale_array(weighted_surprise, min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                               perc_cut=perc_cut)
        
        return weighted_surprise_scaled
    
    def prob_mean(self, min_target_scale=0, max_target_scale=1, perc_cut=None):
        prob_ensemble = self.prob_mean_ensemble[..., 1:2] # Probability mean is for object class (class 1)
        
        prob_ensemble_scaled = scale_array(prob_ensemble, min_target_scale=min_target_scale, max_target_scale=max_target_scale, 
                                           perc_cut=perc_cut)
        
        return prob_ensemble_scaled