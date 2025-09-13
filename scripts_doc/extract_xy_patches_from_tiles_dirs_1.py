# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:01:06 2024

@author: Marcel
"""

# Extract x and y patches from directories of tiles

# This is done according to parameters

# The patch_size is the size of the patch

# The overlap is the overlap between patches (in range [0, 1[), e.g.,
# 0.25 means 25% of overlap between patches (in horizontal and vertical directions)

# border patches extraction (with a maximum of one border patch per line/column)
# are regulated by border_patches parameter, that is set set as False for 
# train and validation groups, and True for test group, in order to build the mosaics 

# The filter_nodata is used to only extract patches without 
# pixels with nodata_value,
# with a certain tolerance set by nodata_tolerance, e.g.,
# nodata_tolerance = 10 and nodata_value = (255, 255, 255)
# only allows patches with the maximum of 10 pixels 
# with value (255, 255, 255). 
# The filter_nodata is set to True in train and validation groups, 
# to filter these patches, 
# and to False in test group, to not filter
# these patches, in order to build the mosaics 

# The filter_object is used to only extract patches with a
# certain amount of pixels of the object (road in this case).
# This inhibits the extraction of patches with only background.
# The object_value is the pixel value of the object class.
# The threshold_percentage is the maximum percentage, 
# relative to the patch's total pixels, 
# that excludes patches from the set. 
# Therefore, only patches with object percentage 
# above the threshold_percentage are selected.  
# The filter_object is set to True in train and validation groups, 
# to filter these patches, 
# and to False in test group, to not filter
# these patches, in order to build the mosaics 

# The exported Y arrays in .npy format have only one channel
# But the exported tensorflow dataset has the Y array in one-hot format 

# %% Import Libraries

from pathlib import Path
import shutil
import json

import tensorflow as tf
import numpy as np

from package_doc.extracao.tile_dir import TileDir, XYTileDir
from package_doc.extracao.tile import TileType

# %% X and Y Input and Output Directories

group = 'train' # train, valid or test group
in_x_dir = fr'dataset_massachusetts_mnih_exp/{group}/input/'
in_y_dir = fr'dataset_massachusetts_mnih_exp/{group}/maps/'
out_x_dir = 'testes/teste_out_x_dir/'
out_y_dir = 'testes/teste_out_y_dir/'

# %% Create output directories if they don't exist

if not Path(out_x_dir).exists():
    Path(out_x_dir).mkdir()
    
if not Path(out_y_dir).exists():
    Path(out_y_dir).mkdir()

# %% Parameters for extraction

patch_size=256

overlap=0.25 

# For train and valid groups, border patches aren't included
# For test group, the border patches are included to make the mosaic
if group == 'train' or group == 'valid':
    border_patches = False
else: # group == 'test'
    border_patches = True

# For train and valid groups, filter for patches without nodata and 
# filter for patches with road objects 
if group == 'train' or group == 'valid':
    filter_nodata = True
    nodata_value = (255, 255, 255)
    nodata_tolerance = 10
    
    filter_object = True
    object_value = 1
    threshold_percentage = 1
else: # group == 'test'
    filter_nodata = False
    nodata_value = (255, 255, 255)
    nodata_tolerance = 10
    
    filter_object = False
    object_value = 1
    threshold_percentage = 1

# %% Create tile dir objects

x_tiledir = TileDir(in_x_dir, TileType.X)
y_tiledir = TileDir(in_y_dir, TileType.Y)

xy_tiledir = XYTileDir(x_tiledir=x_tiledir, y_tiledir=y_tiledir)

# %% Extract patches from tiles

xy_tiledir.extract_patches(patch_size=patch_size, overlap=overlap, 
                           border_patches=border_patches)

# %% Filter against X patches with nodata
 
xy_tiledir.filter_nodata(tile_type=TileType.X,
                             nodata_value=nodata_value,
                             nodata_tolerance=nodata_tolerance
                         )

# %% Filter against Y patches with object

xy_tiledir.filter_object(threshold_percentage=threshold_percentage,
                             object_value=object_value)

# %% Return objects of patches

x_patches_obj, y_patches_obj = xy_tiledir.xy_concat_patches()

# %% Normalize X patches

x_patches_obj.normalize()

# %% Get patches in numpy format

x_patches_np, y_patches_np = x_patches_obj.patches, y_patches_obj.patches

# %% Export arrays to files .npy

x_patches_obj.save_nparray(Path(out_x_dir) / f'x_{group}.npy')
y_patches_obj.save_nparray(Path(out_y_dir) / f'y_{group}.npy')

# %% Export arrays to tensorflow dataset format

# Create directory
dataset_path = Path(out_x_dir) / f'{group}_dataset'
try:
    shutil.rmtree(dataset_path)
except FileNotFoundError:
    dataset_path.mkdir()

# One-hot Y patches 
y_patches_onehot_np = y_patches_obj.onehot()

# Create and save dataset
dataset =  tf.data.Dataset.from_tensor_slices((x_patches_np.astype(np.float16), y_patches_onehot_np))
dataset.save(str(dataset_path))   

# %% Save metadata dictionary (info tiles) to JSON

# Get length and shape of tiles by X Tile Dir
len_tiles = [len(tile_patches.patches) for tile_patches in x_tiledir.tiles_patches]

shape_tiles = [tile.array.shape[:2] for tile in x_tiledir.tiles]

# Repeat stride used in this script to get to all tiles
stride = patch_size - int(patch_size * overlap)
stride_tiles = [stride]*len(shape_tiles)

# Info tiles dict
info_tiles = {'len_tiles': len_tiles, 'shape_tiles': shape_tiles, 'stride_tiles': stride_tiles}

# Export dict
info_tiles_path = Path(out_y_dir) / f'info_tiles_{group}.json'

with open(info_tiles_path, mode='w', encoding='utf-8') as f:
    json.dump(info_tiles, f, sort_keys=True, ensure_ascii=False, indent=4)

 





