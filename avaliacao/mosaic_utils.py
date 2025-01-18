# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:53:47 2025

@author: marce
"""

# Functions to be used in mosaic building

# %% Imports

from osgeo import gdal, gdal_array
import numpy as np

# %% Calculate patch limits in x and y, to be used in border patches

def calculate_xp_yp_limits(p_size, x, y, xmax, ymax, left_half, up_half, right_shift, down_shift):
    # y == 0
    if y == 0:
        if x == 0:
            if y + p_size >= ymax:
                yp_limit = ymax - y 
            else:
                yp_limit = p_size
                
            if x + p_size >= xmax:
                xp_limit = xmax - x
            else:
                xp_limit = p_size
                
        else:
            if y + p_size >= ymax:
                yp_limit = ymax - y
            else:
                yp_limit = p_size
                
            if x + right_shift >= xmax:
                xp_limit = xmax - x + left_half
            else:
                xp_limit = p_size
    # y != 0
    else:
        if x == 0:
            if y + down_shift >= ymax:
                yp_limit = ymax - y + up_half
            else:
                yp_limit = p_size
                
            if x + p_size >= xmax:
                xp_limit = xmax - x
            else:
                xp_limit = p_size
        
        else:
            if y + down_shift >= ymax:
                yp_limit = ymax - y + up_half
            else:
                yp_limit = p_size
                
            if x + right_shift >= xmax:
                xp_limit = xmax - x + left_half
            else:
                xp_limit = p_size

    return xp_limit, yp_limit


# %% Function to create mosaic with test patches

def unpatch_reference(patches, stride, reference_shape, border_patches=False, dtype=None):
    '''
    Function: unpatch_reference
    -------------------------
    Unpatch the patches of a reference to form a mosaic
    
    Input parameters:
      patches  = array containing the patches (patches, h, w)
      stride = stride used to extract the patches
      reference_shape = shape of the target mosaic. Is the same shape from the labels image. (h, w)
      border_patches = include patches overlaping image borders, as extracted with the function extract_patches
    
    Returns: 
      mosaic = array in shape (h, w) with the mosaic build from the patches.
               Array is builded so that, in overlap part, half of the patch is from the left patch,
               and the other half is from the right patch. This is done to lessen border effects.
               This happens also with the patches in the vertical.
    '''
    
    # The objective is reconstruct the image so that, in the overlapping part,
    # the left half belongs the left patch and the right half belongs to the right patch.
    # This is to lesses the border effect

    # We use a square patch (patches.shape[1] == patches.shape[2])
    # overlap and notoverlap are the lengths, in pixels, of the overlapping and non-overlapping parts of the patch 
    # This depends on the stride used to extract the patches
    p_size = patches.shape[1]
    overlap = p_size - stride
    notoverlap = stride
    
    # Half overlap calculation
    # If the number is fractional, the left half is rounded and the right half
    # gets what remains
    half_overlap = overlap/2
    left_half = round(half_overlap)
    right_half = overlap - left_half
    up_half = left_half
    down_half = right_half
    
    # Left and right advancements in pixels for writing mosaic
    # up and down advancements are equal (square patch)
    left_shift = notoverlap + left_half
    right_shift = notoverlap + right_half
    up_shift = left_shift 
    down_shift = right_shift
    
    # Create empty mosaic with its shape
    if dtype:
        pred_test_mosaic = np.zeros(reference_shape, dtype=dtype)
    else:
        pred_test_mosaic = np.zeros(reference_shape, dtype=patches.dtype)

    # Mosaic dimensions along vertical and horizontal axes
    ymax, xmax = pred_test_mosaic.shape
    
    # Line and column to update in loop
    y = 0 
    x = 0 
    
    # Loop to write mosaic  
    for patch in patches:
        # Mosaic part to be updated (debug)
        # mosaic_updated_part = pred_test_mosaic[y : y + p_size, x : x + p_size]
        # print(mosaic_updated_part) 
        
        # If border patches are included (border_patches=True, 
        # with the maximum of one border patch in a patch line/column)
        # and if the patch is a border patch, we use only the necessary space in the patch
        # limited by yp_limit e xp_limit
        if border_patches:
            xp_limit, yp_limit = calculate_xp_yp_limits(p_size, x, y, xmax, ymax, left_half, up_half, right_shift, down_shift)
        # border_patches=False, patches are considered as a whole         
        else:
            yp_limit = p_size
            xp_limit = p_size                   
                    
        # If it is the first patch, the whole patch is used
        # From the second patch onwards, only the right part of the patch will be used, 
        # overwriting the previous patch in the corresponding area
        # This also happens vertically, with those patches below overwriting those immediately above
        if y == 0:
            if x == 0:
                pred_test_mosaic[y : y + p_size, x : x + p_size] = patch[0 : yp_limit, 0 : xp_limit]
            else:
                pred_test_mosaic[y : y + p_size, x : x + right_shift] = patch[0 : yp_limit, left_half : xp_limit]
        # y != 0
        else:
            if x == 0:
                pred_test_mosaic[y : y + down_shift, x : x + p_size] = patch[up_half : yp_limit, 0 : xp_limit]
            else:
                pred_test_mosaic[y : y + down_shift, x : x + right_shift] = patch[up_half : yp_limit, left_half : xp_limit]
            
        # print(pred_test_mosaic) # debug
        
        # Increments reference line after exhausting the reference columns
        
        # If there are no border patches, it is necessary to test if 
        # the next patch will go over the border 
        if not border_patches:
            if x == 0 and x + left_shift + right_shift > xmax:
                x = 0
        
                if y == 0:
                    y = y + up_shift
                else:
                    y = y + notoverlap
        
                continue
            
            else:
                if x + right_shift + right_shift > xmax:
                    x = 0
            
                    if y == 0:
                        y = y + up_shift
                    else:
                        y = y + notoverlap
            
                    continue
                
        
        
        if x == 0 and x + left_shift >= xmax:
            x = 0
            
            if y == 0:
                y = y + up_shift
            else:
                y = y + notoverlap
            
            continue
            
        elif x + right_shift >= xmax:
            x = 0
            
            if y == 0:
                y = y + up_shift
            else:
                y = y + notoverlap
            
            continue
        
        # Incrementa coluna de referência
        # Se ela for o primeiro patch (o mais a esquerda), então será considerada como patch da esquerda
        # Do segundo patch em diante, eles serão considerados como patch da direita
        if x == 0:
            x = x + left_shift
        else:
            x = x + notoverlap
            
    
    return pred_test_mosaic

# %% Function to save a raster with georreference from other raster

def save_raster_reference(in_raster_path, out_raster_path, array_exported):
    # Open reference raster 
    ds_raster = gdal.Open(in_raster_path)
    xsize = ds_raster.RasterXSize
    ysize = ds_raster.RasterYSize
    
    driver = gdal.GetDriverByName('GTiff') # Use Geotiff
    
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

    # Export Raster
    file_band = file.GetRasterBand(1) 
    file_band.WriteArray(array_exported)

    file.SetGeoTransform(ds_raster.GetGeoTransform())
    file.SetProjection(ds_raster.GetProjection())    
    
    file.FlushCache() # Write to disk

