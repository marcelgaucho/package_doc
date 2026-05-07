# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:46:21 2025

@author: Marcel
"""

# Functions used in other modules inside entropy

# %% Imports

import numpy as np
from enum import Enum 
import matplotlib.pyplot as plt


# %% Function to scale array 

def scale_array(array: np.ndarray, min_target_scale=0, max_target_scale=1, 
                perc_cut=None):
    assert max_target_scale > min_target_scale, "Maximum value must be greater than minimum value in target scale"
    
    if all(minmax is None for minmax in [min_target_scale, max_target_scale]):
        return array
    
    min_input = array.min()
    max_input = array.max()
    
    if perc_cut is not None:
        pinf = np.percentile(array, perc_cut)
        psup = np.percentile(array, 100-perc_cut)
        array = np.clip(array, pinf, psup)
        min_input, max_input = pinf, psup

    array_0_1 = (array - min_input) / (max_input - min_input) # normalize to [0, 1]
    array_scaled = array_0_1 * (max_target_scale - min_target_scale) + min_target_scale # scale to [min_target_scale, max_target_scale]

    return array_scaled   

# %% Enumerated constant for data groups

class DataGroups(str, Enum):
    Train = 'train'
    Valid = 'valid'
    Test = 'test' 

# %% Enumerated constant for uncertainty metrics

class UncertaintyMetric(str, Enum):
    Entropy = 'entropy'
    Surprise = 'surprise'
    WeightedSurprise = 'weightedsurprise'
    ProbMean = 'probmean'
    
# %%

def plot_uncertainty_histogram(data, title="Entropy Distribution", log_scale=True):
    flat_data = data.flatten()
    
    plt.figure(figsize=(10, 6))
    nbins = 50
    
    # added range=(0, 1) to keep the x-axis consistent across different tiles
    n, bins, patches = plt.hist(flat_data, bins=nbins, range=(0, 1), 
                                color='royalblue', edgecolor='white', 
                                linewidth=0.5, alpha=0.8)
    
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Frequency (Log Scale)")
    else:
        plt.ylabel("Frequency")
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f"Entropy (bits) \n [0 = Certain, 1 = Maximum Uncertainty] \n bins = {nbins} ")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adding Median alongside Mean for better skewness intuition
    mean_val = flat_data.mean()
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()