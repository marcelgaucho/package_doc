# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:04:07 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
from pathlib import Path
import shutil
import json

import tensorflow as tf
import numpy as np

from package_doc.extracao.tile_dir import TileDir, XsYsTileDir
from package_doc.extracao.tile import TileType
from package_doc.extracao.patches import XYPatches, XPatches


# %% X and Y Input and Output Directories

group = 'train' # train, valid or test group

in_x_dir_t1 = fr'deforestation_dataset/PA/{group}/image/t1/'
in_x_dir_t2 = fr'deforestation_dataset/PA/{group}/image/t2/'

in_y_dir_t1 = fr'deforestation_dataset/PA/{group}/label/t1/'
in_y_dir_t2 = fr'deforestation_dataset/PA/{group}/label/t2/'

out_x_dir = 'experimentos_deforestation/x_dir/'
out_y_dir = 'experimentos_deforestation/y_dir/'

# %% Create output directories if they don't exist

if not Path(out_x_dir).exists():
    Path(out_x_dir).mkdir()
    
if not Path(out_y_dir).exists():
    Path(out_y_dir).mkdir()
    
# %% Parameters for extraction

patch_size=64

overlap=0.9 

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
    nodata_value = (0, 0, 0, 0, 0, 0, 0)
    nodata_tolerance = 0
    
    filter_object = True
    object_value = 1
    threshold_percentage = 2
else: # group == 'test'
    filter_nodata = False
    nodata_value = (0, 0, 0, 0, 0, 0, 0)
    nodata_tolerance = 0
    
    filter_object = False
    object_value = 1
    threshold_percentage = 2


# %% X and Y Tile Directories

# X Tile Dirs
x_tiledir_t1 = TileDir(in_x_dir_t1, TileType.X)
x_tiledir_t2 = TileDir(in_x_dir_t2, TileType.X)

# Y Tile Dirs
y_tiledir_t1 = TileDir(in_y_dir_t1, TileType.Y)
y_tiledir_t2 = TileDir(in_y_dir_t2, TileType.Y)


# %% Preprocess T2 references

ytiledir_t2 = y_tiledir_t2
dilation_px = 2
erosion_px = 0

y_t2_tiles_preprocessed = y_tiledir_t1.preprocess_reference_t2(ytiledir_t2=ytiledir_t2,
                                                               dilation_px=dilation_px,
                                                               erosion_px=erosion_px)

# %%
output_dir = r'testes1/'

for i, tile in enumerate(y_t2_tiles_preprocessed):
    tile.export_to_geotiff(output_dir + str(i) + '.tif')

# %% Create XsYs Tile Directory object

xsys_tiledir = XsYsTileDir(x_tiledirs=[x_tiledir_t1, x_tiledir_t2], 
                           y_tiledirs=[y_tiledir_t1, y_tiledir_t2])


# %% Extract Patches from Xs and Ys directories


(x_t1_patches, x_t2_patches), (y_t1_patches, y_t2_patches) = xsys_tiledir.extract_patches(patch_size=patch_size, overlap=overlap, border_patches=border_patches)


# %% Normalize Patches of Xs directories 

x_t1_patches = x_tiledir_t1.normalize_patches()
x_t2_patches = x_tiledir_t2.normalize_patches()

# %% Filter Patches with object

if filter_object:
    (x_t1_patches, x_t2_patches), (y_t1_patches, y_t2_patches) =  \
    xsys_tiledir.filter_object(y_tiledir_t2, 
                               threshold_percentage=threshold_percentage, 
                               object_value=object_value)

# %% Filter Patches with nodata in X T2

if filter_nodata:
    (x_t1_patches, x_t2_patches), (y_t1_patches, y_t2_patches) =  \
    xsys_tiledir.filter_nodata(x_tiledir_t2, 
                               nodata_value=nodata_value, 
                               nodata_tolerance=nodata_tolerance)

# %% Patches extracted

x_t1_patches = x_tiledir_t1.concat_patches() 
x_t2_patches = x_tiledir_t2.concat_patches()

y_t1_patches = y_tiledir_t1.concat_patches()
y_t2_patches = y_tiledir_t2.concat_patches()

# %% Concatenate X Patches in T1 and T2

x_t1t2_patches = x_t1_patches.concatenate(x_t2_patches)

# %% Adjust variables to export

x_patches_obj = x_t1t2_patches
y_patches_obj = y_t2_patches

# %% Get patches in numpy format

x_patches_np, y_patches_np = x_patches_obj.array, y_patches_obj.array

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
y_patches_onehot_np = y_patches_obj.onehot(num_classes=2, ignore_index=255)

# Create and save dataset
dataset =  tf.data.Dataset.from_tensor_slices((x_patches_np.astype(np.float16), y_patches_onehot_np))
dataset.save(str(dataset_path)) 

# %% Save metadata dictionary (info tiles) to JSON

# Get length of patches in tiles of X Tile Dir
len_tiles = [len(tile.patches.array) for tile in x_tiledir_t2.tiles]

# Get shape of tiles of X Tile Dir
shape_tiles = [tile.array.shape[:2] for tile in x_tiledir_t2.tiles]

# Repeat stride used in this script to get to all tiles
stride = patch_size - int(patch_size * overlap)
stride_tiles = [stride]*len(shape_tiles)

# Info tiles dict
info_tiles = {'len_tiles': len_tiles, 'shape_tiles': shape_tiles, 'stride_tiles': stride_tiles}

# Export dict
info_tiles_path = Path(out_y_dir) / f'info_tiles_{group}.json'

with open(info_tiles_path, mode='w', encoding='utf-8') as f:
    json.dump(info_tiles, f, sort_keys=True, ensure_ascii=False, indent=4)
