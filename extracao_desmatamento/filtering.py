# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:03:53 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np



# %%

class IndexesFinder:
    """
    Return the specified indexes of data, for patch filtering.
    
    Args:
        data: Numpy array (N, H, W, C)        
    """
    def __init__(self, data):
        self.data = data
        
    def object_patches(self, threshold=0.01, target_class=1):
        # 1. Calculate total pixels per patch (Height * Width)
        pixels_per_patch = self.data.shape[1] * self.data.shape[2]
        
        # 2. Convert threshold to pixel count
        threshold_count = pixels_per_patch * threshold
        
        # 3. Mask object values
        is_object = self.data == target_class
        
        # 4. Sum object pixels per patch
        object_counts = np.sum(is_object, axis=(1, 2, 3)) # Sum over Height, Width, and Channels
            
        # 5. Get patch indices where count exceeds threshold
        object_patches_indices = np.where(object_counts >= threshold_count)[0].tolist()
        
        return object_patches_indices
    
    def nodataless_patches(self, nodata_value=255, nodata_tolerance=0):
        # 1. Mask nodata pixels (nodata_value across all channels)
        is_nodata = (self.data == nodata_value).all(axis=-1)
        
        # 2. Mask patches that exceeds nodata count
        nodataless_mask = is_nodata.sum(axis=(1, 2)) <= nodata_tolerance
        
        # 3. Return mask indexes
        return np.where(nodataless_mask)[0].tolist()  
    
# %%

def filter_nodataless_patches(x_patches, y_patches, nodata_value=(255, 255, 255),
                              nodata_tolerance=0):
    """
    Filters out patches that exceed a threshold count of nodata pixels.
    
    Args:
        x_patches: Numpy array (N, H, W, C)
        y_patches: Numpy array (N, H, W, 1)
        threshold: Minimum percentage (0.0 to 1.0) of target_class pixels required.
        target_class: The value representing deforestation/change.
    """
    # 1. Mask nodata pixels
    is_nodata = (x_patches == nodata_value).all(axis=-1)
    
    # 2. Mask patches that exceeds nodata count
    nodataless_mask = is_nodata.sum(axis=(1, 2)) <= nodata_tolerance
    
    # 3. Return x and y filtered patches
    return x_patches[nodataless_mask], y_patches[nodataless_mask] 

# %%

def filter_object_patches(x_patches, y_patches, threshold=0.01, target_class=1):
    """
    Filters out patches that don't have enough foreground pixels.
    
    Args:
        x_patches: Numpy array (N, H, W, C)
        y_patches: Numpy array (N, H, W, 1)
        threshold: Minimum percentage (0.0 to 1.0) of target_class pixels required.
        target_class: The value representing deforestation/change.
    """
    # 1. Calculate total pixels per patch (Height * Width)
    pixels_per_patch = x_patches.shape[1] * x_patches.shape[2]
    
    # 2. Convert threshold to pixel count
    threshold_count = pixels_per_patch * threshold
    
    # 3. Mask object values
    is_object = y_patches == target_class
    
    # 4. Sum object pixels per patch
    object_counts = np.sum(is_object, axis=(1, 2, 3)) # Sum over Height, Width, and Channels
        
    # 5. Get patch indices where count exceeds threshold
    object_patches_indices = np.where(object_counts >= threshold_count)[0].tolist()

    # 6. Return x and y filtered patches
    return x_patches[object_patches_indices], y_patches[object_patches_indices]