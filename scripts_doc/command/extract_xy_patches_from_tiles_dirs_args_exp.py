# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 18:03:03 2025

@author: Marcel
"""

# Extract x and y patches from two directories of tiles (X and Y)
# and puts the output in two directories (X and Y)
# The outputs are in .npy and dataset tensorflow formats
# Saves the metadata of the extracted patches as well, that will be used later 

# %% Import Libraries

from osgeo import gdal
from pathlib import Path
import shutil
import json
import argparse

import tensorflow as tf
import numpy as np

from package_doc.extracao.tile_dir import TileDir, XYTileDir
from package_doc.extracao.tile import TileType

from icecream import ic

# %% Create parser

parser = argparse.ArgumentParser(description='Extract patches from X and Y directories of tiles')

# %% Add arguments

# Input dirs
parser.add_argument('x_tile_dir', type=str, help='Enter the X directory of tiles')
parser.add_argument('y_tile_dir', type=str, help='Enter the Y directory of tiles')

# Output dirs
parser.add_argument('-ox', '--out_x_dir', metavar='output_x_directory', type=str, help='Enter the output X directory')
parser.add_argument('-oy', '--out_y_dir', metavar='output_y_directory', type=str, help='Enter the output Y directory')

# Patch size and overlap
parser.add_argument('-p', '--patch', metavar='patch_size', type=int, help='Enter the patch size for extraction')
parser.add_argument('-o', '--overlap', metavar='overlap_percentage', type=float, help='Enter the overlap percentage, in decimal form, of the patches in extraction')



# %% Parse args 

args = parser.parse_args()
# args = parser.parse_args(['dataset_massachusetts_mnih_exp/test/input',
#                           'dataset_massachusetts_mnih_exp/test/maps',
#                           '-ox', 'testes/teste_out_x_dir',
#                           '-oy', 'testes/teste_out_y_dir',
#                           '-p', '256',
#                           '-o', '0.25'])


# X and Y tile directories
x_tile_dir = args.x_tile_dir
y_tile_dir = args.y_tile_dir

# X and Y output directories
out_x_dir = args.out_x_dir
out_y_dir = args.out_y_dir

# Patch size and overlap
patch_size = args.patch
overlap = args.overlap

# print('x_tile_directory', x_tile_directory)
# print('y_tile_directory', y_tile_directory)
# print('out_x_dir', out_x_dir)
# print('out_y_dir', out_y_dir)
ic(x_tile_dir)
ic(y_tile_dir)
ic(out_x_dir)
ic(out_y_dir)
ic(patch_size)
ic(overlap)

# %% Get group

x_tile_dir = Path(x_tile_dir)
y_tile_dir = Path(y_tile_dir)

x_parent_dir = x_tile_dir.parent.stem
y_parent_dir = y_tile_dir.parent.stem

assert x_parent_dir in ('train', 'valid', 'test'), "Parent directory must be 'train', 'valid' or 'test'"
assert y_parent_dir in ('train', 'valid', 'test'), "Parent directory must be 'train', 'valid' or 'test'"
assert x_parent_dir == y_parent_dir, 'Parent directories of x and y tile directories must have equal name'

group = x_parent_dir

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

x_tiledir = TileDir(x_tile_dir, TileType.X)
y_tiledir = TileDir(y_tile_dir, TileType.Y)

xy_tiledir = XYTileDir(x_tiledir=x_tiledir, y_tiledir=y_tiledir)

# %% Extract patches from tiles

xy_tiledir.extract_patches(patch_size=patch_size, overlap=overlap, 
                           border_patches=border_patches)

# %% Filter against X patches with nodata

if filter_nodata: 
    xy_tiledir.filter_nodata(tile_type=TileType.X,
                                 nodata_value=nodata_value,
                                 nodata_tolerance=nodata_tolerance
                             )

# %% Filter against Y patches with object

if filter_object: 
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
