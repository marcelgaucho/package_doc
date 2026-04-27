# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:15:35 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
import numpy as np
from skimage.morphology import disk, dilation, erosion

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
    normalized = (data - min_value) / (max_value - min_value + 1e-7)
    
    # 3. Cast to float16 for memory efficiency (Storage)
    return normalized.astype(np.float16), min_value, max_value