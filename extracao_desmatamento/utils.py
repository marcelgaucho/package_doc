# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:15:35 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal, gdal_array
import numpy as np
import shutil
from skimage.morphology import disk, dilation, erosion
import tensorflow as tf
from pathlib import Path

# %%

def load_geo_file(path_obj):
    """Loads a geographic file via pathlib and returns a (H, W, C) array."""
    # GDAL requires strings, so we cast the Path object
    ds = gdal.Open(str(path_obj))
    if ds is None:
        raise FileNotFoundError(f"Could not open {path_obj}")
    
    data = ds.ReadAsArray()
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    
    # GDAL (C, H, W) -> (H, W, C)
    return np.transpose(data, (1, 2, 0))

# %%

def preprocess_reference_t2(y_t1, y_t2, 
                            dil_px=2, ero_px=0):
        """Handle the 'ignore' (255) logic for T2."""
        # Structuring elements
        d_foot, e_foot = disk(dil_px), disk(ero_px)
        
        # T1 Dilation
        t1_dilated = dilation(y_t1[..., 0], footprint=d_foot)
        
        # T2 Rings logic
        t2_base = y_t2[..., 0]
        t2_dilated = dilation(t2_base, footprint=d_foot)
        t2_eroded = erosion(t2_base, footprint=e_foot)
        t2_ignore_ring = t2_dilated - t2_eroded # T2 'ignore' ring
        
        # Mark as 255 (ignore value) in T2:
        # the ring where deforestation is uncertain and 
        # the area already deforested (where T1 exists)
        new_t2 = t2_eroded.copy().astype(np.uint16)
        new_t2[t2_ignore_ring == 1] = 255
        new_t2[t1_dilated == 1] = 255
        
        return new_t2[..., np.newaxis]
    
# %%

def preprocess_reference_t2_nobuffer(y_t1, y_t2):
        """Handle the 'ignore' (255) logic for T2 without buffers."""
        # Mark as 255 (ignore value) in T2:
        # the area already deforested (where T1 exists)
        new_t2 = y_t2.copy()
        new_t2[y_t1 == 1] = 255
        
        return new_t2
    
# %%

def minmax_normalize(data, min_value=None, max_value=None):
    """Min-max normalization to [0, 1] for concatenated patches. 
       Use input min and max values if passed."""
    # 1. Force math to be float32 for accuracy
    data = data.astype(np.float32)
    if min_value is None:
        # Min/Max per channel across the whole dataset
        min_value = np.min(data, axis=(0, 1, 2))
        max_value = np.max(data, axis=(0, 1, 2))
        
    # 2. Perform the math
    normalized = (data - min_value) / (max_value - min_value + np.float32(1e-7))
    
    # 3. Return normalized and max and min values used in normalization
    return normalized, min_value, max_value

# %%

def export_to_geotiff(array, base_geotiff, out_path):
    ''' Export to geotiff an array (H, W, C) '''
    # Open base geotiff
    ds_base = gdal.Open(base_geotiff)
    if ds_base is None:
        raise FileNotFoundError(f"Could not open {base_geotiff}")
    
    # Array dimensions
    y_pixels, x_pixels, nbands = array.shape
    
    # Map numpy dtype to GDAL type 
    code_type = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    
    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, xsize=x_pixels, ysize=y_pixels,
                           bands=nbands, eType=code_type)
    
    # Write arrays
    for b in range(nbands):
        out_ds.GetRasterBand(b+1).WriteArray(array[..., b])
        
    # Sync Georeferencing
    out_ds.SetProjection(ds_base.GetProjection())
    out_ds.SetGeoTransform(ds_base.GetGeoTransform())
    
    # Write on disk and close files
    out_ds.FlushCache()
    out_ds = None 
    ds_base = None
    
# %%

def onehot(array, num_classes=2, ignore_index=255):
    ''' One-hot encoding of array (B, H, W, C), setting ignored index to all 0s '''
    array = array.squeeze(axis=3) # Squeeze patches in channel dimension
    
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

# %%

def save_dataset(dataset_path, x_patches, y_patches):
    ''' Save x and y patches to dataset '''
    # Ensure dataset_path is a Path object
    dataset_path = Path(dataset_path)
    
    # Clean and recreate the directory
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(exist_ok=True)
    
    # Create and save dataset
    dataset =  tf.data.Dataset.from_tensor_slices((x_patches, y_patches))
    dataset.save(str(dataset_path)) 
    
    
    

    
    
    
    