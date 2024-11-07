# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:32:45 2024

@author: Marcel
"""

from osgeo import gdal
from pathlib import Path

def replace_value_y_files(y_dir: str, old_value: int, new_value: int):
    ''' Change original tiff y images in directory, replacing value '''
    
    # Paths of the y tiff files
    y_paths = [path for path in Path(y_dir).iterdir() 
               if path.suffix=='.tiff' or path.suffix=='.tif']
    
    for y_path in y_paths:
        # Open as dataset and get metadata
        y_dataset = gdal.Open(y_path, gdal.GA_Update)
        y_dataset_geotransform = y_dataset.GetGeoTransform()
        y_dataset_projection = y_dataset.GetProjection()
        
        # Read dataset as array and replace value
        y_array = y_dataset.ReadAsArray()
        y_array[ y_array == old_value ] = new_value
        
        # Write output
        y_dataset.GetRasterBand(1).WriteArray(y_array)
        y_dataset.SetGeoTransform(y_dataset_geotransform)
        y_dataset.SetProjection(y_dataset_projection)
        y_dataset.FlushCache() # Save result
        
    return True    



def replace_value_y_tiles_nplist(y_tiles: list, old_value: int, new_value: int):
    ''' Replace value in a list of arrays '''
    for array in y_tiles:
        array[ array == old_value ] = new_value
        
    return True 