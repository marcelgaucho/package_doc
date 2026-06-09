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
from matplotlib.ticker import PercentFormatter, ScalarFormatter

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


# %%

class UncertaintyMetric(str, Enum):
    Entropy = 'entropy'
    Surprise = 'surprise'
    WeightedSurprise = 'weighted_surprise'
    ProbMean = 'prob_mean'
    StdDev = 'std_dev'
    
    @property
    def display_name(self) -> str:
        """Returns a nicely formatted string for plot titles."""
        mapping = {
            'entropy': 'Entropy',
            'surprise': 'Surprise',
            'weighted_surprise': 'Weighted Surprise',
            'prob_mean': 'Probability Mean',
            'std_dev': 'Standard Deviation'
        }
        return mapping[self.value]

# %%

def plot_uncertainty_histogram(data, title="Entropy Distribution", log_scale=True, save_path=None):
    flat_data = data.flatten()
    
    plt.figure(figsize=(10, 6))
    nbins = 50
    
    # Create weights for Percentages (for Normalized Frequency)
    weights = (np.ones_like(flat_data, dtype=np.float32) / len(flat_data)) * 100
    
    n, bins, patches = plt.hist(flat_data, bins=nbins, range=(0, 1), 
                                color='royalblue', edgecolor='white', 
                                weights=weights, 
                                linewidth=0.5, alpha=1)
    
    if log_scale:
        plt.yscale('log')
        
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())

        v_min = n[n > 0].min() # lower limit (escape 0 because of log)
        
        # Manually set ticks to useful percentage intervals
        v_min_ceil = np.ceil(v_min * 10) / 10 # round up with 1 decimal 
        ticks_pos = [v_min_ceil, 1, 5, 10, 50, 100]
        ticks_labels = [f'{t}%' for t in ticks_pos]
        
        plt.yticks(ticks_pos, ticks_labels) # set y ticks
        
        plt.ylim(bottom=v_min, top=110) # set y range
        
        plt.ylabel("Percentage of Pixels (%) - Log Scale")
    else:
        plt.gca().yaxis.set_major_formatter(PercentFormatter())
        
        plt.ylabel("Percentage of Pixels (%)")
        
    plt.title(title, fontsize=14, fontweight='bold')
    # plt.xlabel(f"Shannon Entropy (bits) \n [0 = Certain, 1 = Maximum Uncertainty] \n bins = {nbins} ")
    plt.xlabel(f"Uncertainty \n bins = {nbins} ")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adding Mean 
    mean_val = flat_data.mean()
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    
    plt.legend()
    plt.tight_layout()
    
    # Show or save plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)
    