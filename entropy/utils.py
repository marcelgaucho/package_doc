# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:46:21 2025

@author: Marcel
"""

# Functions used in other modules inside entropy

# %% Imports

import numpy as np

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