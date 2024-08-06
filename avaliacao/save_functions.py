# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:27:52 2024

@author: Marcel
"""

from osgeo import gdal, gdal_array
import os
import numpy as np

def salva_arrays(folder, **kwargs):
    '''
    

    Parameters
    ----------
    folder : str
        The string of the folder to save the files.
    **kwargs : keywords and values
        Keyword (name of the file) and the respective value (numpy array)

    Returns
    -------
    None.

    '''
    for kwarg in kwargs:
        if not os.path.exists(os.path.join(folder, kwarg) + '.npy'):
            np.save(os.path.join(folder, kwarg) + '.npy', kwargs[kwarg])
        else:
            print(kwarg, 'não foi salvo pois já existe')
            
            
# Função para salvar mosaico como tiff georreferenciado a partir de outra imagem do qual extraímos as informações geoespaciais
def save_raster_reference(in_raster_path, out_raster_path, array_exported):
    # Copia dados da imagem de labels
    ds_raster = gdal.Open(in_raster_path)
    xsize = ds_raster.RasterXSize
    ysize = ds_raster.RasterYSize
    
    # Exporta imagem para visualização
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver
    
    # Find respective type on GDAL and export
    code_type = gdal_array.NumericTypeCodeToGDALTypeCode(array_exported.dtype)
    if code_type:
        file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=code_type)
    else:
        # Export floating type to GDAL Float32 and unsigned and integer types to GDAL Int32, 
        # in case a correspondent type on GDAL is not found
        if array_exported.dtype.kind == 'f':
            file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Float32)
        elif array_exported.dtype.kind == 'i' or array_exported.dtype.kind == 'u':
            file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Int32)
        else:           
            raise Exception('Could not find correspondent type in GDAL to export array')

    file_band = file.GetRasterBand(1) 
    file_band.WriteArray(array_exported)

    file.SetGeoTransform(ds_raster.GetGeoTransform())
    file.SetProjection(ds_raster.GetProjection())    
    
    file.FlushCache()