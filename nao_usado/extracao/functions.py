# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:08:15 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
import numpy as np
from .tile import TileType

# %%

def extract_patches(array, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
    '''
    Extract patches from numpy image in format (Height, Width, Channels), with a squared patch of informed patch_size,
    with a specified overlap (in range [0, 1[).
    Patches are extracted by row, from left to right, optionally with border patches. If border patches are marked,
    at most only one patch is extracted for each line or column and only when necessary to complete the image.
    '''
    
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
    
    # Return patches array or reshaped patches array
    return patches_array

# %%

def open_tile(tile_path: str, tile_type: TileType):
    # Open dataset and read array
    dataset = gdal.Open(tile_path)
    array = dataset.ReadAsArray()
    
    
    if tile_type == TileType.X:
        assert len(array.shape) == 3, 'Image must have channel dimension'
        # Move to channel-last because the image is multiband
        array = np.moveaxis(array, 0, 2)
        
        return array
    
    elif tile_type == TileType.Y:
        assert len(array.shape) == 2, 'Image must not have channel dimension'
        # Create channel dimension because the image is monochromatic 
        array = np.expand_dims(array, 2)
        
        return array
    
# %% 

def filter_object_xy(patches_x: np.ndarray, patches_y: np.ndarray, threshold_percentage=1, object_value=1):
    pass
    