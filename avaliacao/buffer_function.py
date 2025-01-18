# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:36:26 2024

@author: Marcel
"""

# Buffer function to buffer 4D array

# %% Imports

import numpy as np
from skimage.morphology import disk
import cv2

# %% Function to buffer patches (use morphological dilation to buffer)

def buffer_patches_array(patches: np.ndarray, radius_px=3, print_interval=None):
    assert len(patches.shape) == 4 and patches.shape[-1] == 1, 'Patches must be in shape (B, H, W, 1)'
    
    patches = patches.squeeze(axis=3) # Squeeze patches in last dimension (channel dimension)
    
    # Build structuring element
    struct_elem = disk(radius_px)

    size = len(patches) # Total number of patches 
    result = [] # Result list
    
    for i, patch in enumerate(patches):
        if print_interval:
            if i % print_interval == 0:
                print(f'Buffering patch {i:>6d}/{size:>6d}')
            
        buffered_patch = cv2.dilate(patch.astype(np.uint8), struct_elem)
        result.append(buffered_patch)
        
    result = np.array(result)[..., np.newaxis] # Aggregate list and expand to shape (B, H, W, 1)
    
    return result


