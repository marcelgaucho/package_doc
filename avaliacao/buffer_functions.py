# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:36:26 2024

@author: Marcel
"""
from osgeo import gdal
import cv2
import numpy as np
from skimage.morphology import disk

# Faz um buffer em um array binário
def array_buffer(array, dist_cells=3):
    # Inicializa matriz resultado
    out_array = np.zeros_like(array)
    
    # Dimensões da imagem
    row = array.shape[0]
    col = array.shape[1]    
    
    # i designará o índice correspondete à distância horizontal (olunas) percorrida no buffer para cada célula  
    # j designará o índice correspondete à distância vertical (linhas) percorrida no buffer para cada célula  
    # h percorre colunas
    # k percorre linhas
    i, j, h, k = 0, 0, 0, 0
    
    # Array bidimensional
    if len(out_array.shape) < 3: 
        # Percorre matriz coluna a coluna
        # Percorre colunas
        while(h < col):
            k = 0
            # Percorre linhas
            while (k < row):
                # Teste se célula é maior ou igual a 1
                if (array[k][h] == 1):
                    i = h - dist_cells
                    while(i <= h + dist_cells and i < col):
                        if i < 0:
                            i+=1
                            continue
                            
                        j = k - dist_cells
                        while(j <= k + dist_cells and j < row):
                            if j < 0:
                                j+=1
                                continue
                            
                            # Testa se distância euclidiana do pixel está dentro do buffer
                            if ((i - h)**2 + (j - k)**2 <= dist_cells**2):
                                # Atualiza célula 
                                out_array[j][i] = array[k][h]
    
                            j+=1
                        i+=1
                k+=1
            h+=1
           
    # Array tridimensional disfarçado (1 na última dimensão)
    else:
        # Percorre matriz coluna a coluna
        # Percorre colunas
        while(h < col):
            k = 0
            # Percorre linhas
            while (k < row):
                # Teste se célula é maior ou igual a 1
                if (array[k][h][0] == 1):
                    i = h - dist_cells
                    while(i <= h + dist_cells and i < col):
                        if i < 0:
                            i+=1
                            continue
                            
                        j = k - dist_cells
                        while(j <= k + dist_cells and j < row):
                            if j < 0:
                                j+=1
                                continue
                            
                            # Testa se distância euclidiana do pixel está dentro do buffer
                            if ((i - h)**2 + (j - k)**2 <= dist_cells**2):
                                # Atualiza célula 
                                out_array[j][i][0] = array[k][h][0]
    
                            j+=1
                        i+=1
                k+=1
            h+=1
    
    # Retorna resultado
    return out_array


# Faz buffer em um raster .tif e salva o resultado como outro arquivo .tif
def buffer_binary_raster(in_raster_path, out_raster_path, dist_cells=3):
    # Lê raster como array numpy
    ds_raster = gdal.Open(in_raster_path)
    array = ds_raster.ReadAsArray()
    
    # Faz buffer de 3 pixels
    buffer_array_3px = array_buffer(array)
    
    # Exporta buffer para visualização
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver

    xsize = ds_raster.RasterXSize
    ysize = ds_raster.RasterYSize
    buffer_file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Byte)

    buffer_band = buffer_file.GetRasterBand(1) 
    buffer_band.WriteArray(buffer_array_3px)

    buffer_file.SetGeoTransform(ds_raster.GetGeoTransform())
    buffer_file.SetProjection(ds_raster.GetProjection())    
    
    buffer_file.FlushCache()


# Faz buffer nos patches, sendo que o buffer é feito percorrendo os patches e fazendo o buffer,
# um por um
def buffer_patches(patch_test, dist_cells=3, print_interval=200):
    result = []
    
    for i in range(len(patch_test)):
        if print_interval:
            if i % print_interval == 0:
                print('Buffering patch {}/{}'.format(i+1, len(patch_test))) 
                
        # Patch being buffered 
        patch_batch = patch_test[i, ..., 0]
        
        # Do buffer
        patch_batch_r_new = array_buffer(patch_batch, dist_cells=dist_cells)[np.newaxis, ..., np.newaxis]

        # Append to result list       
        result.append(patch_batch_r_new)
        
    # Concatenate result patches to form result
    result = np.concatenate(result, axis=0)
            
    return result



# Alternative function to buffer_patches
def buffer_patches_array(patches: np.ndarray, radius_px=3, print_interval=None):
    assert len(patches.shape) == 4 and patches.shape[-1] == 1, 'Patches must be in shape (B, H, W, 1)'
    
    patches = patches.squeeze(axis=3) # Squeeze patches in last dimension (channel dimension)
    
    # Build structuring element
    struct_elem = disk(radius_px)

    size = len(patches) # Total number of patches 
    result = [] # Result list
    
    for i, patch in enumerate(patches):
        if print_interval:
            if i % print_interval == 0:
                print(f'Buffering patch {i:>6d}/{size:>6d}')
            
        buffered_patch = cv2.dilate(patch.astype(np.uint8), struct_elem)
        result.append(buffered_patch)
        
    result = np.array(result)[..., np.newaxis] # Aggregate list and expand to shape (B, H, W, 1)
    
    return result


