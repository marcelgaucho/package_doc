# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:53:52 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np
from skimage import morphology


# %% Other approach

def ignore_small_areas1(array: np.ndarray, min_area_px: int, ignore_index: int = 255) -> np.ndarray:
    if len(array.shape) != 4 or array.shape[-1] != 1:
        raise ValueError('Array shape must be in [B, H, W, 1] format.')
    
    # Start with a clean copy of the original array to preserve 0s, 1s, and 255s perfectly
    cleaned_array = array.copy()
    
    # Pad size should ideally be a small buffer or based on sqrt(min_area) to save memory/time,
    # but keeping your pad_size logic here safely:
    pad_size = int(np.ceil(np.sqrt(min_area_px))) # Optimization: padding by sqrt is enough for edge effects
    pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    
    for b in range(array.shape[0]):
        img_slice = array[b, :, :, 0]
        
        # FIX 1: Isolate ONLY the objects (1s) into a binary mask. 
        # This completely ignores 0 and 255 background interference.
        object_mask = (img_slice == 1)
        
        # Pad the binary mask
        padded_slice = np.pad(object_mask, pad_width=pad_width, mode='symmetric')
        
        # Run area opening on the binary mask (removes small component islands of 1s)
        cleaned_slice = morphology.remove_small_objects(padded_slice, min_area_px, connectivity=1)
        cleaned_slice = cleaned_slice[pad_size:-pad_size, pad_size:-pad_size] # crop back
        
        # FIX 2: Now we can accurately find which pixels were flipped from 1 to 0
        removed_pixels = (object_mask == 1) & (cleaned_slice == 0)
        
        # FIX 3: Update the output array directly using the mask
        cleaned_array[b, :, :, 0][removed_pixels] = ignore_index
        
    return cleaned_array

# %% Function to remove small areas from an array

def remove_small_areas_generic(
    array: np.ndarray, 
    min_area_px: int, 
    ignore_index: int = 255, 
    background_index: int = 0
) -> np.ndarray:
    if len(array.shape) != 4 or array.shape[-1] != 1:
        raise ValueError('Array shape must be in [B, H, W, 1] format.')
    
    # Start with a clean copy of the original array.
    # This ensures ignore_index and all valid classes are preserved by default.
    cleaned_array = array.copy()
    
    # Find all unique classes that are NOT the background or the ignore index
    unique_classes = np.unique(array)
    unique_classes = unique_classes[(unique_classes != ignore_index) & (unique_classes != background_index)]
    
    for b in range(array.shape[0]):
        img_slice = array[b, :, :, 0]
        
        for cls in unique_classes:
            # 1. Isolate ONLY the pixels belonging to the current class
            class_mask = (img_slice == cls)
            
            # 2. Find connected components of this class and remove small ones
            cleaned_mask = morphology.area_opening(class_mask, min_area_px, connectivity=1)
            
            # 3. Identify exactly which pixels were removed
            removed_pixels = class_mask & ~cleaned_mask
            
            # 4. Safely turn ONLY those removed pixels into the designated background
            cleaned_array[b, :, :, 0][removed_pixels] = background_index
            
    return cleaned_array