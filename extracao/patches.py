# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 13:08:41 2025

@author: Marcel
"""

# Patches classes for patches extraction

# %% Import Libraries

import numpy as np
from .tile_type import TileType

from typing import Type

# %% Parent class

class Patches:
    def __init__(self, array):
        assert len(array.shape) == 4, 'Array must be in shape (Patches Length, Patch Height, Patch Width, Patch Channels)' # Array shape restriction
        self.array = array
        
    def nodata_indexes(self, nodata_value=(255, 255, 255), nodata_tolerance=0):
        ''' Select indexes of patches without nodata pixels or within a tolerance '''
        assert len(nodata_value) == self.array.shape[-1], "Nodata dimension length has to match channels length"
        
        # Indexes of patches without nodata values (or with nodata values within tolerance)
        # Pixel values are verified to match nodata along channels-last dimension        
        nodata_indexes = [i for i in range(len(self.array)) if 
                                          not np.sum( (self.array[i] == nodata_value).all(axis=-1) ) > 
                                          nodata_tolerance]
                
        return nodata_indexes
    
    def __repr__(self):
        return f'{self.__class__.__name__} {self.array.shape}'
    
# %% Child class of X Patches

class XPatches(Patches):
    def __init__(self, array):
        super().__init__(array)
        
    def normalize(self):
        '''Normalize image for each band'''
        min_value = [np.min(self.array[..., i]) for i in range(self.array.shape[-1])]        
        max_value = [np.max(self.array[..., i]) for i in range(self.array.shape[-1])]
        
        self.array = self.array.astype(np.float32) # transform array in float
                
        for i in range(self.array.shape[-1]):
            self.array[..., i] = (self.array[..., i] - min_value[i]) / (max_value[i] - min_value[i])
        
        return self.array
    
    def concatenate(self, x_patches: 'XPatches'):
        ''' Concatenate other XPatches array to the present XPatches array '''
        xpatches = XPatches(np.concatenate((self.array, x_patches.array), axis=-1))
        
        return xpatches
    
    def save_nparray(self, output_path):
        # Cast array to occupy less space
        array = self.array.astype(np.float16)

        np.save(output_path, array)
        
        return True
    
# %% Child class of Y Patches

class YPatches(Patches):
    def __init__(self, array):
        super().__init__(array)
        assert array.shape[-1] == 1, 'Array must be a mask, so the channel length must be equal to 1'
        
    def object_indexes(self, threshold_percentage=1, object_value=1):
        ''' Select indexes of patches with object pixels percentage above a threshold '''
        assert threshold_percentage >= 0 and threshold_percentage <= 100, 'Threshold percentage of object pixels in the patch must be between 0 and 100'
        
        # Number of pixels of a patch and of the threshold
        pixels_patch = self.array.shape[1] * self.array.shape[2]
        pixels_threshold = pixels_patch*(threshold_percentage/100)
        
        # Indexes of patches above threshold (object indexes)
        object_indexes = [i for i in range(len(self.array)) if
                                           np.sum(self.array[i] == object_value) > pixels_threshold]
        
        return object_indexes
    
    def onehot(self, num_classes=2, ignore_index=255):
        array = self.array.squeeze(axis=3) # Squeeze patches in last dimension (channel dimension)
        
        # If ignore_index is set, first change ignore_index to the last value and 
        # increment the number of classes
        if ignore_index is not None:
            mask = (array == ignore_index)
            array = array.copy()
            array[mask] = num_classes
            num_classes = num_classes + 1            
            
        array = np.eye(num_classes, dtype=np.uint8)[array] # One-Hot codification
        
        # Consider classes only up to the ignore index
        if ignore_index is not None:
            array = array[..., :num_classes-1]
        
        return array
    
    def save_nparray(self, output_path):
        # Cast array to occupy less space
        array = self.array.astype(np.uint8)

        np.save(output_path, array)
        
        return True
    
# %% Composite class of X and Y Patches

class XYPatches(Patches):
    def __init__(self, patches_x: XPatches, patches_y: YPatches):
        self.patches_x = patches_x
        self.patches_y = patches_y
        
    def filter_nodata(self, tile_type: TileType, nodata_value=(255, 255, 255), nodata_tolerance=0):
        if tile_type == TileType.X:
            nodata_indexes = self.patches_x.nodata_indexes(nodata_value=nodata_value, 
                                                           nodata_tolerance=nodata_tolerance)
        elif tile_type == TileType.Y:
            nodata_indexes = self.patches_y.nodata_indexes(nodata_value=nodata_value, 
                                                           nodata_tolerance=nodata_tolerance)
            
        self.patches_x.array = self.patches_x.array[nodata_indexes]
        self.patches_y.array = self.patches_y.array[nodata_indexes]
        
        return self.patches_x.array, self.patches_y.array
            
        
    def filter_object(self, threshold_percentage=1, object_value=1):
        object_indexes = self.patches_y.object_indexes(threshold_percentage=threshold_percentage, 
                                                       object_value=object_value)
        
        self.patches_x.array = self.patches_x.array[object_indexes]
        self.patches_y.array = self.patches_y.array[object_indexes]
        
        return self.patches_x.array, self.patches_y.array
        
        
        
        