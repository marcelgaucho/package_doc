# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:21:02 2024

@author: Marcel
"""

# %% Import libraries

from osgeo import gdal
from pathlib import Path
from icecream import ic
import numpy as np
import shutil
import tensorflow as tf
import json



# %% PatchExtractor class definition

class DirPatchExtractor:
    def __init__(self, in_x_dir: str, in_y_dir: str, out_x_dir: str, out_y_dir: str,
                 patch_size=256, overlap=0.25, border_patches=False,
                 filter_nodata=False, nodata_value=(255, 255, 255), nodata_tolerance=0,
                 filter_object=False, object_value=1, threshold_percentage=1,
                 group='train'
                 ):
        assert group in ('train', 'valid', 'test'), "Parameter group must be 'train', 'valid' or 'test'"
        
        # Directories
        self.in_x_dir = in_x_dir
        self.in_y_dir = in_y_dir
        self.out_x_dir = out_x_dir
        self.out_y_dir = out_y_dir
        
        # Data for extraction
        self.patch_size = patch_size
        
        self.overlap = overlap        
        self.stride = self.patch_size - int(self.patch_size * self.overlap)
        
        self.border_patches = border_patches
        
        self.filter_nodata = filter_nodata
        self.nodata_value = nodata_value
        self.nodata_tolerance = nodata_tolerance
        
        self.filter_object = filter_object
        self.object_value = object_value
        self.threshold_percentage = threshold_percentage
        
        self.group = group
        
        # List of tiles
        x_tiles = [str(path) for path in Path(in_x_dir).iterdir() 
                   if path.suffix=='.tiff' or path.suffix=='.tif']
        x_tiles.sort()
        self.x_tiles = [DirPatchExtractor.load_tile(path) for path in x_tiles]
        
        
        y_tiles = [str(path) for path in Path(in_y_dir).iterdir() 
                   if path.suffix=='.tiff' or path.suffix=='.tif']
        y_tiles.sort()        
        self.y_tiles = [DirPatchExtractor.load_tile(path) for path in y_tiles]
        
        # Patches extracted
        self.x_patches = None
        self.y_patches = None
        
        # Extraction metadata
        self.len_tiles = None
        self.shape_tiles = None
        self.stride_tiles = None
        
    def extract_patches_from_tiles(self):        
        # Lists to store x and y patches and its metadata
        x_patches, y_patches  = [], []
        len_tiles, shape_tiles, stride_tiles =  [], [], []
        
        # Loop through the lists with tiles paths
        for x_tile, y_tile in zip(self.x_tiles, self.y_tiles):
            ic(x_tile, y_tile)
            
            # Extract patches
            p_x = self._extract_patches(img_array=x_tile)
            p_y = self._extract_patches(img_array=y_tile)
            ic(p_x, p_y)
            
            # If marked, filter patches for without nodata and that contains object
            if self.filter_nodata:
                ids_without_nodata = self.ids_without_nodata(x_patches=p_x)
                p_x = p_x[ids_without_nodata]
                p_y = p_y[ids_without_nodata]
                
            if self.filter_object:
                ids_with_object = self.ids_with_object(y_patches=p_y)
                p_x = p_x[ids_with_object]
                p_y = p_y[ids_with_object]              

                
            # Add to list
            x_patches.append(p_x)
            y_patches.append(p_y)
            
            len_tiles.append(len(p_y))
            shape_tiles.append(y_tile.shape[:2])
            stride_tiles.append(self.stride)
            
        # Concatenate patches in list to array
        x_patches = np.concatenate(x_patches, axis=0)
        y_patches = np.concatenate(y_patches, axis=0)
        
        # Save inside object
        self.x_patches = x_patches
        self.y_patches = y_patches
        
        self.len_tiles = len_tiles
        self.shape_tiles = shape_tiles
        self.stride_tiles = stride_tiles
        
        return True    

    def _extract_patches(self, img_array: np.ndarray):
        '''
        Extract patches from numpy image in format (Height, Width, Channels), with a squared patch of informed patch_size,
        with a specified overlap (normally a decimal in range [0, 1[).
        Patches are extracted by row, from left to right, optionally with border patches. If border patches are marked,
        at most only one patch is extracted for each line or column and only when necessary to complete the image.
        '''
        assert self.overlap < 1, "Overlap must be lower than 1 for the stride to be positive" # Overlap restriction
        assert len(img_array.shape) == 3, 'Image must be in shape (Height, Width, Channels)' # Image shape restriction
        
        # Dimensions of input and output
        h_input, w_input, c_input = img_array.shape
        
        c_output = c_input
        
        # If division is inexact and there is border patches, add one border patch
        if ( (h_input - self.patch_size) % self.stride != 0 ) and self.border_patches:
            h_output = int(  (h_input - self.patch_size)/self.stride + 1 ) + 1
            w_output = int(  (w_input - self.patch_size)/self.stride + 1 ) + 1
        # Else the normal operation is executed
        else:
            h_output = int(  (h_input - self.patch_size)/self.stride + 1 )
            w_output = int(  (w_input - self.patch_size)/self.stride + 1 )
       
        ic(h_output, w_output)
        
        if (h_output <= 0 or w_output <= 0) and self.border_patches == False:
            raise Exception('Could not generate output with zero height or width')
            
        # m loop through rows and n through columns
        # for ease, the last is stored in case of border_patches
        last_m = h_output - 1
        last_n = w_output - 1
        
        ic(last_m, last_n)
        
        # List with image patches
        patch_img = []
        
        # Extract patches row by row
        for m in range(0, h_output):
            for n in range(0, w_output):
                # Indexes relative to input image
                i_h = m*self.stride
                i_w = n*self.stride
                
                # Image is overflowed
                if (m == h_output - 1 or n == w_output - 1) and self.border_patches:
                    # Border Patch initially has zeros
                    border_patch_img = np.zeros((self.patch_size, self.patch_size, c_output), dtype=img_array.dtype)
                    
                    # If patch goes beyond image height,
                    # border_mmax is what goes from beginning of the patch up to bottom border of image
                    if (i_h + self.patch_size > h_input):
                        border_mmax = img_array.shape[0] - i_h
                    # Otherwise, the patch size is maintained   
                    else:
                        border_mmax = self.patch_size
                        
                    # If patch goes beyond image width,
                    # border_nmax is what goes from beginning of the patch up to right border of image    
                    if (i_w + self.patch_size > w_input):
                        border_nmax = img_array.shape[1] - i_w
                    else:
                        border_nmax = self.patch_size                    
                       
                    # Fill patches
                    border_patch_img[0:border_mmax, 0:border_nmax, :] = img_array[i_h : i_h+border_mmax, i_w : i_w+border_nmax, :]
                    
                    # Add patches to list 
                    patch_img.append( border_patch_img )
                
                # Patch is inside image
                else:
                    patch_img.append( img_array[i_h : i_h+self.patch_size, i_w : i_w+self.patch_size, :] )          
        
                    
        # Store in object to give as an output          
        patches_array = np.array(patch_img)
        # patches_array_reshaped = patches_array.reshape((h_output, w_output, patch_size, patch_size, c_output)) # reshape to (rows, cols, patch size, patch size, channel size)
        
        # Return patches array or reshaped patches array
        return patches_array
    
    def normalize_x_patches(self):
        min_value = np.min(self.x_patches)        
        max_value = np.max(self.x_patches)
                
        self.x_patches = (self.x_patches - min_value) / (max_value - min_value)
        
        return True
    
    def _onehot_y_patches(self):
        ''' One-hot y patches with the n classes numbered from 0 to n-1 in the array '''
        assert len(self.y_patches.shape) == 4 and self.y_patches.shape[-1] == 1, 'Y Patches must be in shape (B, H, W, 1)'
        
        n_classes = np.max(self.y_patches) + 1 # number of classes
        
        y_patches = np.eye(n_classes, dtype=np.uint8)[self.y_patches.squeeze(axis=3)] 
        
        return y_patches
    
    def save_np_arrays(self, overwrite_arrays=False):
        assert self.group in ('train', 'valid', 'test'), "Parameter group must be 'train', 'valid' or 'test'"
        
        # Cast arrays to types to occupy few space
        x_patches = self.x_patches.astype(np.float16)
        y_patches = self.y_patches.astype(np.uint8)
        
        x_path = Path(self.out_x_dir) / f'x_{self.group}.npy'
        y_path = Path(self.out_y_dir) / f'y_{self.group}.npy'
        
        if (x_path.exists() or y_path.exists()) and overwrite_arrays==False:
            raise Exception('X array or Y array already exists in output directory')
            
        np.save(x_path, x_patches)
        np.save(y_path, y_patches)
        
    def save_tf_dataset(self, overwrite_dir=False, onehot_y_patches=True):
        assert self.group in ('train', 'valid', 'test'), "Parameter group must be 'train', 'valid' or 'test'"

        # Create directory
        dataset_path = Path(self.out_x_dir) / f'{self.group}_dataset'
        if overwrite_dir:
            if dataset_path.exists(): 
                shutil.rmtree(dataset_path)
        dataset_path.mkdir()
        
        # Cast arrays (to occupy few space) and one-hot y patches if marked
        x_patches = self.x_patches.astype(np.float16)
        if onehot_y_patches:
            y_patches = self._onehot_y_patches() # already in uint8
        else:
            y_patches = self.y_patches.astype(np.uint8)
        
        # Create and save dataset
        dataset =  tf.data.Dataset.from_tensor_slices((x_patches, y_patches))
        dataset.save(str(dataset_path))            
        
        return True   

    def save_info_tiles(self, overwrite_info=False):
        # Dict to export to JSON
        info_tiles = {'len_tiles': self.len_tiles, 'shape_tiles': self.shape_tiles, 'stride_tiles': self.stride_tiles}
        
        info_tiles_path = Path(self.out_y_dir) / f'info_tiles_{self.group}.json'
        
        if info_tiles_path.exists() and overwrite_info==False:
            raise Exception('Info file already exists')
            
        with open(info_tiles_path, mode='w', encoding='utf-8') as f:
            json.dump(info_tiles, f, sort_keys=True, ensure_ascii=False, indent=4)
        
    
    @staticmethod
    def load_tile(tile_path):
        dataset = gdal.Open(tile_path)
        array = dataset.ReadAsArray()
        # Move to channel-last if the image is multiband
        if len(array.shape) > 2:
            array = np.moveaxis(array, 0, 2)
        # Create channel dimension for monochromatic images 
        else:
            array = np.expand_dims(array, 2)
        
        return array
    
    def ids_without_nodata(self, x_patches):
        ''' Select indexes of patches without nodata pixels or within a tolerance '''
        assert len(x_patches.shape) == 4, 'Array must be in shape (Patches Length, Patch Height, Patch Width, Patch Channels)' # Array shape restriction
        assert len(self.nodata_value) == x_patches.shape[-1], "Nodata dimension length has to match channels length"
        
        # Indexes of patches without nodata values (or with nodata values within tolerance)
        # Pixel values are verified to match nodata along channels-last dimension        
        patches_without_nodata_indexes = [i for i in range(len(x_patches)) if 
                                          not np.sum( (x_patches[i] == self.nodata_value).all(axis=-1) ) > self.nodata_tolerance]
        
        return patches_without_nodata_indexes
    

    def ids_with_object(self, y_patches):
        ''' Select indexes of patches with object pixels percentage above a threshold '''
        assert len(y_patches.shape) == 4, 'Array must be in shape (Patches Length, Patch Height, Patch Width, Patch Channels)' # Array shape restriction
        assert y_patches.shape[-1] == 1, 'Array must be a mask, so the channel length must be equal to 1'
        assert self.threshold_percentage >= 0 and self.threshold_percentage <= 100, 'Threshold percentage of object pixels in the patch must be between 0 and 100'
        
        # Number of pixels of a patch and of the threshold
        pixels_patch = y_patches.shape[1] * y_patches.shape[2]
        pixels_threshold = pixels_patch*(self.threshold_percentage/100)
        
        # Indexes of patches above threshold
        patches_above_threshold_indexes = [i for i in range(len(y_patches)) if
                                           np.sum(y_patches[i] == self.object_value) > pixels_threshold]
        
        return patches_above_threshold_indexes