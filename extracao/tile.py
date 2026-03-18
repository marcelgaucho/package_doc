# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 16:56:04 2025

@author: Marcel
"""

# Tiles classes and function for patches extraction

# %% Import Libraries

from osgeo import gdal, gdal_array
import numpy as np

from .patches import XPatches, YPatches
from .tile_type import TileType

from skimage.morphology import disk, dilation, erosion
# from scipy.ndimage import binary_dilation

from abc import ABC, abstractmethod

# %% Parent class

class Tile(ABC):
    def __init__(self, tile_path: str):
        self.tile_path = tile_path
        
        self.dataset = gdal.Open(tile_path)
        
        self.array = self._array()
        
        self.patches = None
        
    def _array(self) -> np.ndarray:
        return self.dataset.ReadAsArray()
    
    def nodata_indexes(self, nodata_value=(255, 255, 255), nodata_tolerance=0):
        return self.patches.nodata_indexes(nodata_value=nodata_value,
                                           nodata_tolerance=nodata_tolerance)
    
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
        self.patches = patches_array
        
        # Return patches array or reshaped patches array
        return self.patches
    
    @abstractmethod
    def export_to_geotiff(self, output_path):
        # Dimensions of output        
        x_pixels = self.array.shape[1]
        y_pixels = self.array.shape[0]
        
        # Find numpy respective type on GDAL 
        code_type = gdal_array.NumericTypeCodeToGDALTypeCode(self.array.dtype)
        
        # Create output dataset
        driver = gdal.GetDriverByName('GTiff')
        nbands = self.array.shape[2]
        out_ds = driver.Create(output_path, xsize=x_pixels, ysize=y_pixels,
                               bands=nbands, eType=code_type)
        
        # Write arrays
        for b in range(nbands):
            out_ds.GetRasterBand(b+1).WriteArray(self.array[..., b])
            
        # Set geotransform and projection from input
        out_ds.SetGeoTransform(self.dataset.GetGeoTransform())
        out_ds.SetProjection(self.dataset.GetProjection())
        
        # Write on disk
        out_ds.FlushCache()
        
        return True
    
    def __repr__(self):
        return f'{self.__class__.__name__} {self.array.shape}'
    
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
        self.patches = XPatches(self.patches) # Transform in custom object
        
        return self.patches
    
    def export_to_geotiff(self, output_path):
        return super().export_to_geotiff(output_path=output_path)
    
    # def normalize(self):
    #     self.tile_patches.normalize()
    
# %% Child class of Y Tile

class YTile(Tile):
    def _array(self):
        array = self.dataset.ReadAsArray()
        
        assert len(array.shape) == 2, 'Image must not have channel dimension'

        # Create channel dimension because the image is monochromatic 
        array = np.expand_dims(array, 2)
        
        return array
        
    def object_indexes(self, threshold_percentage=1, object_value=1):
        return self.patches.object_indexes(threshold_percentage=threshold_percentage,
                                           object_value=object_value)      
    
    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        super().extract_patches(patch_size=patch_size, overlap=overlap, border_patches=border_patches)
        self.patches = YPatches(self.patches) # Transform in custom object
        
        return self.patches
    
    def preprocess_reference_t2(self, ytile_t2: 'YTile', dilation_px=2, 
                              erosion_px=0):
        ''' This object has reference T1, while the reference T2 is passed in a parameter  '''
        # Structuring element
        disk_dil = disk(dilation_px)
        disk_ero = disk(erosion_px)
        
        # 1st operation with Reference T1 - Dilation
        dilated_ref_t1 = dilation(self.array[..., 0], footprint=disk_dil)[..., np.newaxis]
        
        # Operations with Reference T2 - Dilation, Erosion and Ring calculation
        dilated_ref_t2 = dilation(ytile_t2.array[..., 0], footprint=disk_dil)[..., np.newaxis]
        dilation_ring_ref_t2 = dilated_ref_t2 - ytile_t2.array
        eroded_ref_t2 = erosion(ytile_t2.array[..., 0], footprint=disk_ero)[..., np.newaxis]
        erosion_ring_ref_t2 = ytile_t2.array - eroded_ref_t2
        dilation_erosion_ring_ref_t2 = dilation_ring_ref_t2 + erosion_ring_ref_t2
        
        # First update Reference T2 by removing the erosion ring
        new_ref_t2 = ytile_t2.array - erosion_ring_ref_t2
        
        # Update the value of the ring resulted from dilation-erosion by 255s (mask value)
        # and add the updated result to Reference T2
        dilation_erosion_ring_ref_t2[dilation_erosion_ring_ref_t2 == 1] = 255
        new_ref_t2 = new_ref_t2 + dilation_erosion_ring_ref_t2
        
        # Now mask, in ref T2, the object pixels from ref T1
        new_ref_t2[dilated_ref_t1 == 1] = 255
        ytile_t2.array = new_ref_t2
        
        return ytile_t2.array
    
    def export_to_geotiff(self, output_path):
        return super().export_to_geotiff(output_path=output_path)
        

    
# %% Factory function to create tile object (XTile or YTile)

def tile(tile_path: str, tile_type: TileType):
    if tile_type == TileType.X:
        return XTile(tile_path)
    elif tile_type == TileType.Y:
        return YTile(tile_path)
    
    raise Exception(f'Tile type must be in: {list(TileType)}')