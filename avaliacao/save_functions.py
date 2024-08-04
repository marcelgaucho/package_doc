# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:27:52 2024

@author: Marcel
"""

from osgeo import gdal
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
def save_raster_reference(in_raster_path, out_raster_path, array_exported, is_float=False):
    # Copia dados da imagem de labels
    ds_raster = gdal.Open(in_raster_path)
    xsize = ds_raster.RasterXSize
    ysize = ds_raster.RasterYSize
    
    # Exporta imagem para visualização
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver
    if is_float:
        file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Float32) # Tipo float para números entre 0 e 1
    else:        
        file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Byte) # Tipo Byte para somente números 0 e 1

    file_band = file.GetRasterBand(1) 
    file_band.WriteArray(array_exported)

    file.SetGeoTransform(ds_raster.GetGeoTransform())
    file.SetProjection(ds_raster.GetProjection())    
    
    file.FlushCache()