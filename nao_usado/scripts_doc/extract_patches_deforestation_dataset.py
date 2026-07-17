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

# %% Parameters for extraction

patch_size = 64

object_value = 1
threshold_percentage = 2

nodata_value = (0, 0, 0, 0, 0, 0, 0)
nodata_tolerance = 0

# --- Configuration ---
CONFIGS = {
    'train': {'border': True, 'filter': True, 'overlap': 0.9, 'coords': False,
              'norm_with_train': False, 'export_dataset': True},
    'valid': {'border': True, 'filter': True, 'overlap': 0.9, 'coords': False,
              'norm_with_train': True, 'export_dataset': True},
    'test':  {'border': True,  'filter': False, 'overlap': 0.75, 'coords': True,
              'norm_with_train': True, 'export_dataset': False}
}

# %% X and Y Input and Output Directories

group = 'test' # train, valid or test group
cfg = CONFIGS[group]

in_x_dir_t1 = fr'deforestation_dataset/PA/{group}/image/t1/'
in_x_dir_t2 = fr'deforestation_dataset/PA/{group}/image/t2/'

in_y_dir_t1 = fr'deforestation_dataset/PA/{group}/label/t1/'
in_y_dir_t2 = fr'deforestation_dataset/PA/{group}/label/t2/'

out_x_dir = 'experimentos_deforestation/x_dir/'
out_y_dir = 'experimentos_deforestation/y_dir/'

# %% Load train parameters for normalization

if cfg['norm_with_train']:
    with open(Path(out_y_dir) / 'info_tiles_train.json') as fp:   
        info_tiles_train = json.load(fp)
        
    min_value_train = np.array(info_tiles_train['min_value_train'])
    max_value_train = np.array(info_tiles_train['max_value_train'])
    
# %% Create output directories if they don't exist

if not Path(out_x_dir).exists():
    Path(out_x_dir).mkdir()
    
if not Path(out_y_dir).exists():
    Path(out_y_dir).mkdir()
    
# %% X and Y Tile Directories

# X Tile Dirs
x_tiledir_t1 = TileDir(in_x_dir_t1, TileType.X)
x_tiledir_t2 = TileDir(in_x_dir_t2, TileType.X)

# Y Tile Dirs
y_tiledir_t1 = TileDir(in_y_dir_t1, TileType.Y)
y_tiledir_t2 = TileDir(in_y_dir_t2, TileType.Y)


# %% Preprocess T2 references

ytiledir_t1 = y_tiledir_t1
dilation_px = 2
erosion_px = 0

y_t2_tiles_preprocessed = y_tiledir_t2.preprocess_reference(ytiledir_t1=ytiledir_t1,
                                                            dilation_px=dilation_px,
                                                            erosion_px=erosion_px)

# %% Export preprocessed references
'''
output_dir = Path(r'testes1/')
output_dir.mkdir(exist_ok=True)

for tile in y_t2_tiles_preprocessed:
    filename_ext = Path(tile.tile_path).stem + '_prep.tif'
    tile_path = output_dir / filename_ext
    tile.export_to_geotiff(str(tile_path))
'''

# %% Create XsYs Tile Directory object

xsys_tiledir = XsYsTileDir(x_tiledirs=[x_tiledir_t1, x_tiledir_t2], 
                           y_tiledirs=[y_tiledir_t1, y_tiledir_t2])


# %% Extract Patches from Xs and Ys directories


(x_t1_patches, x_t2_patches), (y_t1_patches, y_t2_patches) = xsys_tiledir.extract_patches(patch_size=patch_size, overlap=cfg['overlap'], 
                             border_patches=cfg['border'])


# %% Normalize Patches of Xs directories 

if cfg['norm_with_train']: # Valid and Test
    (x_t1_patches, x_t2_patches), \
    min_value, max_value = xsys_tiledir.normalize_patches(min_value=min_value_train,
                                                                   max_value=max_value_train)    
else: # Train
    (x_t1_patches, x_t2_patches), \
    min_value_train, max_value_train = xsys_tiledir.normalize_patches(min_value=None,
                                                                      max_value=None)

# %% Filter Patches with object

if cfg['filter']:
    (x_t1_patches, x_t2_patches), (y_t1_patches, y_t2_patches) =  \
    xsys_tiledir.filter_object(y_tiledir_t2, 
                               threshold_percentage=threshold_percentage, 
                               object_value=object_value)

# %% Filter Patches with nodata in X T2

if cfg['filter']:
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

if cfg['export_dataset']:
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
shape_tiles = [tile.array.shape for tile in y_tiledir_t2.tiles]

# Repeat stride used in this script to get to all tiles
stride = patch_size - int(patch_size * cfg['overlap'])
stride_tiles = [stride]*len(shape_tiles)

# Overlap to store
overlap = [cfg['overlap']]*len(shape_tiles)

# Get patch size of the extraction
patch_sizes = [patch_size]*len(shape_tiles)

# Info tiles dict
if cfg['norm_with_train']:
    # Store coords if group is test
    if cfg['coords']: # Test
        coords_path = Path(out_y_dir) / f'coords_{group}.npz'
        coords = [tile.coords for tile in x_tiledir_t2.tiles]
        info_tiles = {'len_tiles': len_tiles, 'shape_tiles': shape_tiles, 
                      'stride_tiles': stride_tiles, 'overlap': overlap, 
                      'patch_sizes': patch_sizes, 'coords_path': str(coords_path)}
        np.savez_compressed(coords_path, coords=coords)
    else: # Valid
        info_tiles = {'len_tiles': len_tiles, 'shape_tiles': shape_tiles, 
                      'stride_tiles': stride_tiles, 'overlap': overlap,
                      'patch_sizes': patch_sizes}
else: # Train
    info_tiles = {'len_tiles': len_tiles, 'shape_tiles': shape_tiles, 
                  'stride_tiles': stride_tiles, 'overlap': overlap,
                  'patch_sizes': patch_sizes,
                  'min_value_train': min_value_train.tolist(),
                  'max_value_train': max_value_train.tolist()}

# Export dict
info_tiles_path = Path(out_y_dir) / f'info_tiles_{group}.json'

with open(info_tiles_path, mode='w', encoding='utf-8') as f:
    json.dump(info_tiles, f, sort_keys=True, ensure_ascii=False, indent=4)
