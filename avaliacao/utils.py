# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:40:23 2025

@author: marce
"""

# Functions used in other modules inside avaliacao

# %% Imports

from osgeo import gdal
from typing import Union
from pathlib import Path
import numpy as np
from skimage import morphology

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

# %% Function to remove small areas from an array

def ignore_small_areas(array: np.ndarray, min_area_px: int, ignore_index: int=255) -> np.ndarray:
    if len(array.shape) != 4 or array.shape[-1] != 1:
        raise ValueError('Array shape must be in [B, H, W, 1] format.')
    
    # Mask valid pixels
    masked_array = array * (array != ignore_index)
    
    cleaned_array = np.zeros_like(array)
    
    # Define padding with min area (for edge objects)
    pad_size = min_area_px
    pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    
    # Remove and ignore areas with a loop
    for b in range(array.shape[0]):
        img_slice = masked_array[b, :, :, 0]
        padded_slice = np.pad(img_slice, pad_width=pad_width, mode='symmetric') # pad slice
        
        cleaned_slice = morphology.area_opening(padded_slice, min_area_px, connectivity=1)
        cleaned_slice = cleaned_slice[pad_size:-pad_size, pad_size:-pad_size] # crop back
        
        removed_pixels = (img_slice == 1) & (cleaned_slice == 0) # removed pixels 
       
        # Ignore small areas
        cleaned_array[b, :, :, 0] = cleaned_slice
        cleaned_array[b, :, :, 0][removed_pixels] = ignore_index        
        
    # Return ignore index to array
    cleaned_array[array == ignore_index] = ignore_index
        
    return cleaned_array 

# %% Function to load reference mosaics

def load_reference_mosaics(label_tiles_dir: Union[str, Path]) -> np.ndarray:
    # Enforce Path
    label_tiles_dir = Path(label_tiles_dir)
    
    # Tiles in dir
    labels_paths = sorted([
        str(p) for p in label_tiles_dir.iterdir() 
        if p.suffix in ('.tiff', '.tif')
    ])
    
    # Load tiles
    y_mosaics = [gdal.Open(path).ReadAsArray() for path in labels_paths]
    
    # Return stacked array
    return stack_uneven(y_mosaics)[..., np.newaxis]

# %% Function to decode one-hot array in dataset

def decode_onehot(y_onehot: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Reverses one-hot encoding, converting (B, H, W, C) back to sparse (B, H, W, 1).
    Restores the ignore_index for pixels where all class channels are 0.
    """
    # Get the predicted class indices
    y_sparse = np.argmax(y_onehot, axis=-1)[..., np.newaxis]
    
    # Identify pixels that were ignored (all channels sum to 0)
    if ignore_index is not None:
        ignore_mask = np.sum(y_onehot, axis=-1)[..., np.newaxis] == 0
        y_sparse[ignore_mask] = ignore_index
        
    return y_sparse.astype(np.uint8)


