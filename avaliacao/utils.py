# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:40:23 2025

@author: marce
"""

# Functions used in other modules inside avaliacao

# %% Imports

import numpy as np

# %% Function for "one-hot codification" of binary probability

def adiciona_prob_0(prob_1):
    ''' Add first channel (background) to a probability array with only road channel (prob_1).
        It is a kind of one-hot codification for binary probability '''   
    zeros = np.zeros(prob_1.shape, dtype=np.uint8)
    prob_1 = np.concatenate((zeros, prob_1), axis=-1)
    
    # Percorre dimensão dos batches e adiciona imagens (arrays) que são o complemento
    # da probabilidade de estradas
    for i in range(prob_1.shape[0]):
        prob_1[i, :, :, 0] = 1 - prob_1[i, :, :, 1]
        
    return prob_1

# %% Function for stacking arrays of different sizes

# Source: https://stackoverflow.com/questions/44951624/numpy-stack-with-unequal-shapes
def stack_uneven(arrays, fill_value=0):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result