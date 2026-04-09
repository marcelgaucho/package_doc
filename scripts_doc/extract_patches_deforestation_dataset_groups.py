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


# %% X and Y Output Directories of Data

out_x_dir = 'experimentos_deforestation/x_dir/'
out_y_dir = 'experimentos_deforestation/y_dir/'

# %%

groups = ['train', 'valid', 'test'] # train, valid or test group

in_x_dir_t1 = {group : fr'deforestation_dataset/PA/{group}/image/t1/' for group in groups}
in_x_dir_t2 = {group: fr'deforestation_dataset/PA/{group}/image/t2/' for group in groups}

in_y_dir_t1 = {group : fr'deforestation_dataset/PA/{group}/label/t1/' for group in groups}
in_y_dir_t2 = {group : fr'deforestation_dataset/PA/{group}/label/t2/' for group in groups}

# %% Create output directories if they don't exist

if not Path(out_x_dir).exists():
    Path(out_x_dir).mkdir()
    
if not Path(out_y_dir).exists():
    Path(out_y_dir).mkdir()
    
# %% Parameters for extraction

patch_size=64

overlap =  {'train': 0.9, 'valid': 0.9, 'test': 0.75}

# For train and valid groups, border patches aren't included
# For test group, the border patches are included to make the mosaic
border_patches = {'train': False, 'valid': False, 'test': True}

# For train and valid groups, filter for patches without nodata and 
# filter for patches with road objects 
nodata_value = (0, 0, 0, 0, 0, 0, 0)
nodata_tolerance = 0

object_value = 1
threshold_percentage = 2

filter_object = {'train': True, 'valid': True, 'test': False}

filter_nodata = {'train': True, 'valid': True, 'test': False}

# %% X and Y Tile Directories

# X Tile Dirs
x_tiledir_t1 = {group: TileDir(in_x_dir_t1[group], TileType.X)
                for group in groups}

x_tiledir_t2 = {group: TileDir(in_x_dir_t2[group], TileType.X)
                for group in groups}

# Y Tile Dirs
y_tiledir_t1 = {group: TileDir(in_y_dir_t1[group], TileType.Y)
                for group in groups}

y_tiledir_t2 = {group: TileDir(in_y_dir_t2[group], TileType.Y)
                for group in groups}

# %% Preprocess T2 references

dilation_px = 2
erosion_px = 0

y_t2_tiles_preprocessed = {}

for group in groups:
    y_t2_tiles_preprocessed[group] = y_tiledir_t2[group].preprocess_reference(ytiledir_t1=y_tiledir_t1[group],
                                                                              dilation_px=dilation_px,
                                                                              erosion_px=erosion_px)

# %% Export preprocessed references

# Uncomment to export to geotiffs
'''
output_dir = r'tiles_t2_preprocessed/'
Path(output_dir).mkdir(exist_ok=True)

for group in groups:
    output_dir_group = Path(fr'{output_dir}{group}')
    output_dir_group.mkdir(exist_ok=True)
    
    for tile in y_t2_tiles_preprocessed[group]:
        filename_ext = Path(tile.tile_path).stem + '_prep.tif'
        tile_path = output_dir_group / filename_ext
        tile.export_to_geotiff(str(tile_path))
'''      



# %% Create XsYs Tile Directory object

xsys_tiledir = {group: XsYsTileDir(x_tiledirs=[x_tiledir_t1[group], 
                                               x_tiledir_t2[group]], 
                                   y_tiledirs=[y_tiledir_t1[group], 
                                               y_tiledir_t2[group]])
                for group in groups}

# %% Extract Patches from Xs and Ys directories

x_t1_patches, x_t2_patches, y_t1_patches, y_t2_patches = {}, {}, {}, {}

for group in groups:
    (x_t1_patches[group], x_t2_patches[group]), \
    (y_t1_patches[group], y_t2_patches[group]) = xsys_tiledir[group].extract_patches(patch_size=patch_size, 
                                                                                     overlap=overlap[group], 
                                                                                     border_patches=border_patches[group])   
    
# %% Normalize Train Patches of Xs directories 

(x_t1_patches['train'], x_t2_patches['train']), \
min_value_train, max_value_train = xsys_tiledir['train'].normalize_patches(min_value=None,
                                                                           max_value=None)

# %% Normalize Valid Patches of Xs directories

(x_t1_patches['valid'], x_t2_patches['valid']), \
min_value_valid, max_value_valid = xsys_tiledir['valid'].normalize_patches(min_value=min_value_train,
                                                                           max_value=max_value_train)

# %% Normalize Test Patches of Xs directories

(x_t1_patches['test'], x_t2_patches['test']), \
min_value_test, max_value_test = xsys_tiledir['test'].normalize_patches(min_value=min_value_train,
                                                                        max_value=max_value_train)

# %% Filter Patches with object

for group, f_obj in filter_object.items():
    if f_obj:
        (x_t1_patches[group], x_t2_patches[group]), \
        (y_t1_patches[group], y_t2_patches[group]) = \
        xsys_tiledir[group].filter_object(y_tiledir_t2[group], 
                                          threshold_percentage=threshold_percentage, 
                                          object_value=object_value)
        
# %% Filter Patches with nodata in X T2

for group, f_nodata in filter_nodata.items():
    if f_nodata:
        (x_t1_patches[group], x_t2_patches[group]), \
        (y_t1_patches[group], y_t2_patches[group]) = xsys_tiledir[group].filter_nodata(x_tiledir_t2[group], 
                                                                                       nodata_value=nodata_value, 
                                                                                       nodata_tolerance=nodata_tolerance)



# %% Concatenate X Patches in T1 and T2

x_t1t2_patches = {}

for group in groups:
    x_t1t2_patches[group] = x_t1_patches[group].concatenate(x_t2_patches[group])

# %% Adjust variables to export

x_patches_obj = x_t1t2_patches
y_patches_obj = y_t2_patches

# %% Get patches in numpy format

x_patches_np, y_patches_np = {}, {}

for group in groups:
    x_patches_np[group], y_patches_np[group] = x_patches_obj[group].array, y_patches_obj[group].array

# %% Export arrays to files .npy

for group in groups:
    x_patches_obj[group].save_nparray(Path(out_x_dir) / f'x_{group}.npy')
    y_patches_obj[group].save_nparray(Path(out_y_dir) / f'y_{group}.npy')

# %% Export Train and Valid arrays to tensorflow dataset format

for group in ['train', 'valid']:
    # Create directory
    dataset_path = Path(out_x_dir) / f'{group}_dataset'
    try:
        shutil.rmtree(dataset_path)
    except FileNotFoundError:
        dataset_path.mkdir()
    
    # One-hot Y patches 
    y_patches_onehot_np = y_patches_obj[group].onehot(num_classes=2, ignore_index=255)
    
    # Create and save dataset
    dataset =  tf.data.Dataset.from_tensor_slices((x_patches_np[group].astype(np.float16), 
                                                   y_patches_onehot_np))
    dataset.save(str(dataset_path)) 

# %% Save metadata dictionary (info tiles) to JSON

for group in groups:
    # Get length of patches in tiles of X Tile Dir
    len_tiles = [len(tile.patches.array) for tile in x_tiledir_t2[group].tiles]
    
    # Get shape of tiles of X Tile Dir
    shape_tiles = [tile.array.shape[:2] for tile in x_tiledir_t2[group].tiles]
    
    # Repeat stride used in this script to get to all tiles
    stride = patch_size - int(patch_size * overlap[group])
    stride_tiles = [stride]*len(shape_tiles)
    
    # Info tiles dict
    info_tiles = {'len_tiles': len_tiles, 'shape_tiles': shape_tiles, 'stride_tiles': stride_tiles}
    
    # Export dict
    info_tiles_path = Path(out_y_dir) / f'info_tiles_{group}.json'
    
    with open(info_tiles_path, mode='w', encoding='utf-8') as f:
        json.dump(info_tiles, f, sort_keys=True, ensure_ascii=False, indent=4)
