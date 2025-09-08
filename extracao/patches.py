# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 13:08:41 2025

@author: Marcel
"""

# Patches classes for patches extraction

# %% Import Libraries

import numpy as np

# %% Parent class

class Patches:
    def __init__(self, patches):
        assert len(patches.shape) == 4, 'Array must be in shape (Patches Length, Patch Height, Patch Width, Patch Channels)' # Array shape restriction
        self.patches = patches
        
    def nodata_indexes(self, nodata_value=(255, 255, 255), nodata_tolerance=0):
        ''' Select indexes of patches without nodata pixels or within a tolerance '''
        assert len(nodata_value) == self.patches.shape[-1], "Nodata dimension length has to match channels length"
        
        # Indexes of patches without nodata values (or with nodata values within tolerance)
        # Pixel values are verified to match nodata along channels-last dimension        
        nodata_indexes = [i for i in range(len(self.patches)) if 
                                          not np.sum( (self.patches[i] == nodata_value).all(axis=-1) ) > 
                                          nodata_tolerance]
                
        return nodata_indexes
    
# %% Child class of X Patches

class XPatches(Patches):
    def __init__(self, patches):
        super().__init__(patches)
        
    def normalize(self):
        min_value = np.min(self.patches)        
        max_value = np.max(self.patches)
                
        self.patches = (self.patches - min_value) / (max_value - min_value)
        
        return self
    
    def save_nparray(self, output_path):
        # Cast array to occupy less space
        patches = self.patches.astype(np.float16)

        np.save(output_path, patches)
        
        return True
    
# %% Child class of Y Patches

class YPatches(Patches):
    def __init__(self, patches):
        super().__init__(patches)
        assert patches.shape[-1] == 1, 'Array must be a mask, so the channel length must be equal to 1'
        
    def object_indexes(self, threshold_percentage=1, object_value=1):
        ''' Select indexes of patches with object pixels percentage above a threshold '''
        assert threshold_percentage >= 0 and threshold_percentage <= 100, 'Threshold percentage of object pixels in the patch must be between 0 and 100'
        
        # Number of pixels of a patch and of the threshold
        pixels_patch = self.patches.shape[1] * self.patches.shape[2]
        pixels_threshold = pixels_patch*(threshold_percentage/100)
        
        # Indexes of patches above threshold (object indexes)
        object_indexes = [i for i in range(len(self.patches)) if
                                           np.sum(self.patches[i] == object_value) > pixels_threshold]
        
        return object_indexes
    
    def onehot(self):
        patches = self.patches.squeeze(axis=3) # Squeeze patches in last dimension (channel dimension)
        
        n_values = np.max(patches) + 1 # number of classes
        
        patches = np.eye(n_values, dtype=np.uint8)[patches] # One-Hot codification
        
        return patches
    
    def save_nparray(self, output_path):
        # Cast array to occupy less space
        patches = self.patches.astype(np.uint8)

        np.save(output_path, patches)
        
        return True
