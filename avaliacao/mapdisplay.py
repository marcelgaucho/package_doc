# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 18:46:19 2025

@author: marcel.rotunno
"""

# Classes and function to export map with TP, TN, FP, FN

# %% Imports

from osgeo import gdal, gdal_array
from .buffer_function import buffer_patches_array
import numpy as np
from enum import Enum

# %% Classes for result map

class ResultMap:
    def __init__(self, pred_path: str, y_path: str, 
                 out_path: str, radius_px: int=3):
        self.pred_path = pred_path
        self.y_path = y_path
        self.out_path = out_path
        
        self.pred = self._array(self.pred_path)
        self.y = self._array(self.y_path)
        
        self.pred_buffer = buffer_patches_array(patches=self.pred, radius_px=radius_px)
        self.y_buffer = buffer_patches_array(patches=self.y, radius_px=radius_px)
        
        self.tp, self.tn, self.fp, self.fn = self._calculate()
        
        self.result = self._result()
        
   
    @staticmethod
    def _array(str_path: str) -> np.ndarray:
        dataset = gdal.Open(str_path)
        array = dataset.ReadAsArray()
        # Modify shape to (batch, height, width, channels) - First and last dimension have length 1
        array = array[np.newaxis, :, :, np.newaxis] 
        
        dataset = None
        
        return array
    
    def _result(self):
        # Result array
        result =  np.stack([self.tp, self.tn, self.fp, self.fn], axis=4)
        result = result.squeeze()
        result = np.argmax(result, axis=2).astype(np.uint8) # Convert to 1-band array
        
        return result       
    
    def _calculate(self):
        # Default result is all True Negatives
        tp = np.zeros(self.y_path.shape, dtype=np.uint8)
        tn = np.ones(self.y_path.shape, dtype=np.uint8)
        fp = np.zeros(self.y_path.shape, dtype=np.uint8)
        fn = np.zeros(self.y_path.shape, dtype=np.uint8)
                
        return tp, tn, fp, fn
    
    def export(self):
        # Export result
        result = self.result
        ysize = result.shape[0]
        xsize = result.shape[1] 
        bands = 1 
        
        driver = gdal.GetDriverByName('GTiff') # Use Geotiff
        code_type = gdal_array.NumericTypeCodeToGDALTypeCode(result.dtype)
        
        file = driver.Create(self.out_path, xsize, ysize, bands=bands, eType=code_type)

        ds = gdal.Open(self.pred_path)
        file.SetGeoTransform(ds.GetGeoTransform())
        file.SetProjection(ds.GetProjection())

        band = file.GetRasterBand(1)
        band.WriteArray(result)
        
        file.FlushCache()
        
        file = None
        ds = None
        band = None
        
    
class ResultMapPrec(ResultMap):
    def _calculate(self):
        # TP e FP
        tp = self.y_buffer*self.pred
        fp = self.pred - tp
        
        # TN e FN
        pred_neg = 1 - self.pred
        y_buffer_neg = 1 - self.y_buffer
        
        tn = y_buffer_neg*pred_neg
        fn = pred_neg - tn
        
        return tp, tn, fp, fn
        
    
class ResultMapRecall(ResultMap):
    def _calculate(self):
        # TP e FN
        tp = self.y*self.pred_buffer
        fn = self.y - tp
        
        # TN e FP
        pred_buffer_neg = 1 - self.pred_buffer
        y_neg = 1 - self.y
        
        tn = y_neg*pred_buffer_neg
        fp = self.pred_buffer - tp
        
        return tp, tn, fp, fn

    
# %% Enumerator and convenience function for object generation

class MapType(str, Enum):
    Precision = "Precision"
    Recall = "Recall"
    
def resultmap(pred_path: str, y_path: str, out_path: str, maptype: MapType, radius_px: int=3):
    if maptype == MapType.Precision:
        return ResultMapPrec(pred_path=pred_path, y_path=y_path, out_path=out_path, radius_px=radius_px)
                
    elif maptype == MapType.Recall:
        return ResultMapRecall(pred_path=pred_path, y_path=y_path, out_path=out_path, radius_px=radius_px)            
    
    raise Exception(f'Map type must be in: {list(MapType)}')

