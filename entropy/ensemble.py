# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:48:42 2025

@author: Marcel
"""

# Create a folder with data for building an ensemble


# %% Imports

import shutil
from pathlib import Path
from enum import Enum
import numpy as np
from .utils import scale_array


# %% Enumerated constant for data groups

class DataGroups(str, Enum):
    Train = 'train'
    Valid = 'valid'
    Test = 'test' 
    
# %% Ensemble Directory Class

class EnsembleDir:
    def __init__(self, ensemble_folder: str, model_dirs):
        self.ensemble_folder = Path(ensemble_folder)
        self.model_dirs = model_dirs
        
    def create(self):
        # Create ensemble dir
        if not self.ensemble_folder.exists():
            print("Creating ensemble directory")
            self.ensemble_folder.mkdir()
        elif self.ensemble_folder.exists():
            raise Exception("Ensemble directory already exists")
            
    def paste_data(self, overwrite_arrays=False):
        # Copy and enumerate prob arrays in ascending order
        for i, folder in enumerate(sorted(self.model_dirs)):
            for data_group in iter(DataGroups):
                # Copy Prob Files
                prob_file = self.ensemble_folder / f'prob_{data_group}_{i}.npy'
                if prob_file.exists() and overwrite_arrays is False:
                    print(f"{i}-prob_{data_group} already exists, so it won't be copied")
                else:    
                    print(f"Copying {i}-prob_{data_group}")                  
                    shutil.copy2(Path(folder) / f'prob_{data_group}.npy', prob_file)
                    
# %% Ensemble Class
                
class Ensemble:
    def __init__(self, ensemble_dir: EnsembleDir, data_group: str = DataGroups.Test):
        self.ensemble_dir = ensemble_dir
        self.data_group = data_group
        
        self.prob_mean_ensemble = self._prob_mean_ensemble()
        
        self.epsilon = 1e-7 # Used to manage log 0
        
    def _prob_mean_ensemble(self):
        prob_ensemble = [np.load(self.ensemble_dir.ensemble_folder / f'prob_{self.data_group}_{i}.npy') 
                         for i in range(len(self.ensemble_dir.model_dirs))] # Load prob arrays in list
        prob_ensemble = np.array(prob_ensemble) # transform list in array
        prob_ensemble = np.concatenate((1-prob_ensemble, prob_ensemble), axis=-1) # insert the background probability
        prob_mean_ensemble = np.mean(prob_ensemble, axis=0)
        
        return prob_mean_ensemble
    
    def entropy(self, min_target_scale=0, max_target_scale=1, perc_cut=None, save_result=False):
        epsilon = self.epsilon
        
        prob_ensemble = self.prob_mean_ensemble
        
        entropy = -np.sum(prob_ensemble * np.log2(prob_ensemble + epsilon), axis=-1, keepdims=True)
        
        entropy_scaled = scale_array(entropy, min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                     perc_cut=perc_cut)
        
        if save_result:
            np.save(self.ensemble_dir.ensemble_folder / f'entropy_{self.data_group}.npy', entropy_scaled)           

        return entropy_scaled
        
    def surprise(self, min_target_scale=0, max_target_scale=1, perc_cut=None, save_result=False):    
        epsilon = self.epsilon
        
        prob_ensemble = self.prob_mean_ensemble[..., 1:2] # Surprise is for object class (class 1)
        
        surprise = -np.log2(prob_ensemble + epsilon)
        
        surprise_scaled = scale_array(surprise, min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                      perc_cut=perc_cut)
        
        if save_result:
            np.save(self.ensemble_dir.ensemble_folder / f'surprise_{self.data_group}.npy', surprise_scaled)
            
        return surprise_scaled
    
    def weighted_surprise(self, min_target_scale=0, max_target_scale=1, perc_cut=None, save_result=False):    
        epsilon = self.epsilon
        
        prob_ensemble = self.prob_mean_ensemble[..., 1:2] # Surprise is for object class (class 1)

        weighted_surprise = -prob_ensemble * np.log2(prob_ensemble + epsilon)
        
        weighted_surprise_scaled = scale_array(weighted_surprise, min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                               perc_cut=perc_cut)
        
        if save_result:
            np.save(self.ensemble_dir.ensemble_folder / f'weightedsurprise_{self.data_group}.npy', weighted_surprise_scaled)
            
        return weighted_surprise_scaled
            
    def prob_mean(self, min_target_scale=0, max_target_scale=1, perc_cut=None, save_result=False):
        prob_ensemble = self.prob_mean_ensemble[..., 1:2] # Probability mean is for object class (class 1)
        
        prob_ensemble_scaled = scale_array(prob_ensemble, min_target_scale=min_target_scale, max_target_scale=max_target_scale, 
                                           perc_cut=perc_cut)
        
        if save_result:
            np.save(self.ensemble_dir.ensemble_folder / f'probmean_{self.data_group}.npy', prob_ensemble_scaled)
            
        return prob_ensemble_scaled