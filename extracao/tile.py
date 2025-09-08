# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 16:56:04 2025

@author: Marcel
"""

# Tiles classes and function for patches extraction

# %% Import Libraries

from osgeo import gdal
import numpy as np

from .patches import XPatches, YPatches

# %% Parent class

class Tile:
    def __init__(self, tile_path: str):
        self.tile_path = tile_path
        
        self.dataset = gdal.Open(tile_path)
        
        self.array = self._array()
        
        self.tile_patches = None
        
    def _array(self) -> np.ndarray:
        return self.dataset.ReadAsArray()
    
    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        '''
        Extract patches from numpy image in format (Height, Width, Channels), with a squared patch of informed patch_size,
        with a specified overlap (in range [0, 1[).
        Patches are extracted by row, from left to right, optionally with border patches. If border patches are marked,
        at most only one patch is extracted for each line or column and only when necessary to complete the image.
        '''
        
        array = self.array
        
        assert len(array.shape) == 3, 'Image must be in shape (Height, Width, Channels)' # Image shape restriction
        
        # Calculate stride
        assert 0  <= overlap < 1, "Overlap must be in range [0, 1[. 0 means no overlap and 1 is no stride."
        
        stride = patch_size - int(patch_size * overlap)
        
        # Dimensions of input and output
        h_input, w_input, c_input = array.shape
        
        c_output = c_input
        
        # If division is inexact and there is border patches, add one border patch
        if ( (h_input - patch_size) % stride != 0 ) and border_patches:
            h_output = int(  (h_input - patch_size)/stride + 1 ) + 1
            w_output = int(  (w_input - patch_size)/stride + 1 ) + 1
        # Else the normal operation is executed
        else:
            h_output = int(  (h_input - patch_size)/stride + 1 )
            w_output = int(  (w_input - patch_size)/stride + 1 )
       
        if (h_output <= 0 or w_output <= 0) and border_patches == False:
            raise Exception('Could not generate output with zero height or width')
            
        # m loop through rows and n through columns
        # for ease, the last is stored in case of border_patches
        last_m = h_output - 1
        last_n = w_output - 1
        
        # List with image patches
        patch_img = []
        
        # Extract patches row by row
        for m in range(0, h_output):
            for n in range(0, w_output):
                # Indexes relative to input image
                i_h = m*stride
                i_w = n*stride
                
                # Image is overflowed
                if (m == last_m or n == last_n) and border_patches:
                    # Border Patch initially has zeros
                    border_patch_img = np.zeros((patch_size, patch_size, c_output), dtype=array.dtype)
                    
                    # If patch goes beyond image height,
                    # border_mmax is what goes from beginning of the patch up to bottom border of image
                    if (i_h + patch_size > h_input):
                        border_mmax = array.shape[0] - i_h
                    # Otherwise, the patch size is maintained   
                    else:
                        border_mmax = patch_size
                        
                    # If patch goes beyond image width,
                    # border_nmax is what goes from beginning of the patch up to right border of image    
                    if (i_w + patch_size > w_input):
                        border_nmax = array.shape[1] - i_w
                    else:
                        border_nmax = patch_size                    
                       
                    # Fill patches
                    border_patch_img[0:border_mmax, 0:border_nmax, :] = array[i_h : i_h+border_mmax, i_w : i_w+border_nmax, :]
                    
                    # Add patches to list 
                    patch_img.append( border_patch_img )
                
                # Patch is inside image
                else:
                    patch_img.append( array[i_h : i_h+patch_size, i_w : i_w+patch_size, :] )          
        
                    
        # Output array        
        patches_array = np.array(patch_img)
        # patches_array_reshaped = patches_array.reshape((h_output, w_output, patch_size, patch_size, c_output)) # reshape to (rows, cols, patch size, patch size, channel size)
        
        # Store output in object
        self.tile_patches = patches_array
        
        # Return patches array or reshaped patches array
        return self
    
# %% Child class of X Tile

class XTile(Tile):
    def _array(self) -> np.ndarray:
        array = self.dataset.ReadAsArray()

        assert len(array.shape) == 3, 'Image must have channel dimension'

        # Move to channel-last because the image is multiband
        array = np.moveaxis(array, 0, 2)
        
        return array
    
    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        super().extract_patches(patch_size=patch_size, overlap=overlap, border_patches=border_patches)
        self.tile_patches = XPatches(self.tile_patches) # Transform in custom object
        
        return self
    
# %% Child class of Y Tile

class YTile(Tile):
    def _array(self):
        array = self.dataset.ReadAsArray()
        
        assert len(array.shape) == 2, 'Image must not have channel dimension'

        # Create channel dimension because the image is monochromatic 
        array = np.expand_dims(array, 2)
        
        return array
    
    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        super().extract_patches(patch_size=patch_size, overlap=overlap, border_patches=border_patches)
        self.tile_patches = YPatches(self.tile_patches) # Transform in custom object
        
        return self
    
# %% Enumerated constant for tile types

from enum import Enum

class TileType(str, Enum):
    X = 'X'
    Y = 'Y'
    
# %% Factory function to create tile object (XTile or YTile)

def tile(tile_path: str, tile_type: TileType):
    if tile_type == TileType.X:
        return XTile(tile_path)
    elif tile_type == TileType.Y:
        return YTile(tile_path)
    
    raise Exception(f'Tile type must be in: {list(TileType)}')