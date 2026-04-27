#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:15:01 2026

@author: rotunno
"""

# %% Import libraries

import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from package_doc.extracao.patch_extraction import PatchExtractor
from package_doc.extracao_desmatamento.filtering import IndexesFinder
from package_doc.extracao_desmatamento.utils import (load_geo_file, 
                                                     preprocess_reference_t2,
                                                     minmax_normalize)

from skimage.morphology import disk, dilation, erosion


    
     

# %%

class PatchProcessor:
    def __init__(self, base_path, patch_size=64, overlap=0.9):
        self.base_path = base_path
        self.patch_size = patch_size
        self.overlap = overlap
        # Initialize extractor 
        self.extractor = PatchExtractor(
            patch_size=self.patch_size, 
            overlap_percent=self.overlap, 
            padding=True
        )    
        
        self.info_tiles_train = None
        self.info_tiles_valid = None
        self.info_tiles_test = None
        
    def export_info(self, split_name='test', out_dir='./'):
        with open(Path(out_dir) / f'info_tiles_{split_name}.json', 'w') as f:
            json.dump( self.info_tiles_test, f, indent=4)

    def process_split(self, split_name='train'):
        # Initialize the base path object
        base = Path(self.base_path) / split_name
        
        # Define directories using pathlib / operator
        img_t1_dir = base / "image" / "t1"
        img_t2_dir = base / "image" / "t2"
        lbl_t1_dir = base / "label" / "t1"
        lbl_t2_dir = base / "label" / "t2"
        
        x_all, y_all = [], []
        lbl_t2_paths = []
        coords_all = []
        meta_tiles = []
        
        # Iterate through t1 files and find matches in other folders
        for t1_path in sorted(img_t1_dir.glob("*.tif")):
            fname = t1_path.name  # Get the filename string
            yt2_path = lbl_t2_dir / fname # Get label path as reference
            
            # Load Images
            xt1 = load_geo_file(t1_path)
            xt2 = load_geo_file(img_t2_dir / fname)
            
            # Load Labels
            yt1 = load_geo_file(lbl_t1_dir / fname)
            yt2 = load_geo_file(lbl_t2_dir / fname)
            
            # 1. Preprocess Y: Early Fusion logic
            # Combine labels (T1 info injected into T2)
            y_combined = preprocess_reference_t2(yt1, yt2, dil_px=2, ero_px=0)
    
            # 2. Concatenate X: Early Fusion (Stacking T1 and T2 channels)
            x_combined = np.concatenate([xt1, xt2], axis=-1)
    
            # 3. Extract Patches
            x_p, coords = self.extractor(x_combined, get_coords=True)
            y_p         = self.extractor(y_combined, get_coords=False)
            
            # 4. Filter Patches
            nodataless_indexes = IndexesFinder(x_p).nodataless_patches(nodata_value=0,
                                                                       nodata_tolerance=0)
            x_p, y_p, coords = x_p[nodataless_indexes], y_p[nodataless_indexes], coords[nodataless_indexes] # By Nodataless
            
            object_indexes = IndexesFinder(y_p).object_patches(threshold=0.02,
                                                               target_class=1)
            x_p, y_p, coords = x_p[object_indexes], y_p[object_indexes], coords[object_indexes]    
            
            # Store these specifically for this file
            tile_info = {
                "filepath": str(yt2_path),
                "shape": yt2.shape,
                "len_patches": len(coords)
            }
            meta_tiles.append(tile_info)
    
            x_all.append(x_p)
            y_all.append(y_p)
            lbl_t2_paths.append(str(lbl_t2_dir / fname))
            coords_all.append(coords)
            
        with open(f'testes2/info_tiles_{split_name}.json', 'w') as f:
            json.dump(meta_tiles, f, indent=4)
    
        return (np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0), 
               lbl_t2_paths, coords_all)    
        
    def __repr__(self):
        return f'PatchProcessor (base_path={self.base_path}, ' \
               f'patch_size={self.patch_size}, overlap={self.overlap})'




# %% 


def process_geographic_split(base_path, split_name, patch_size=256, 
                             overlap=0.25):
    # Create extractor
    extractor = PatchExtractor(patch_size=patch_size, overlap_percent=overlap, 
                               padding=True)
    
    # Initialize the base path object
    base = Path(base_path) / split_name
    
    # Define directories using pathlib / operator
    img_t1_dir = base / "image" / "t1"
    img_t2_dir = base / "image" / "t2"
    lbl_t1_dir = base / "label" / "t1"
    lbl_t2_dir = base / "label" / "t2"

    x_all, y_all = [], []
    lbl_t2_paths = []
    coords_all = []
    meta_tiles = []

    # Iterate through t1 files and find matches in other folders
    for t1_path in sorted(img_t1_dir.glob("*.tif")):
        fname = t1_path.name  # Get the filename string
        yt2_path = lbl_t2_dir / fname # Get label path as reference
        
        # Load Images
        xt1 = load_geo_file(t1_path)
        xt2 = load_geo_file(img_t2_dir / fname)
        
        # Load Labels
        yt1 = load_geo_file(lbl_t1_dir / fname)
        yt2 = load_geo_file(lbl_t2_dir / fname)
        
        # 1. Preprocess Y: Early Fusion logic
        # Combine labels (T1 info injected into T2)
        y_combined = preprocess_reference_t2(yt1, yt2, dil_px=2, ero_px=0)

        # 2. Concatenate X: Early Fusion (Stacking T1 and T2 channels)
        x_combined = np.concatenate([xt1, xt2], axis=-1)

        # 3. Extract Patches
        x_p, coords = extractor(x_combined, get_coords=True)
        y_p         = extractor(y_combined, get_coords=False)
        
        # 4. Filter Patches
        nodataless_indexes = IndexesFinder(x_p).nodataless_patches(nodata_value=0,
                                                                   nodata_tolerance=0)
        x_p, y_p, coords = x_p[nodataless_indexes], y_p[nodataless_indexes], coords[nodataless_indexes] # By Nodataless
        
        object_indexes = IndexesFinder(y_p).object_patches(threshold=0.02,
                                                           target_class=1)
        x_p, y_p, coords = x_p[object_indexes], y_p[object_indexes], coords[object_indexes]    
        
        # Store these specifically for this file
        tile_info = {
            "filepath": str(yt2_path),
            "shape": yt2.shape,
            "len_patches": len(coords)
        }
        meta_tiles.append(tile_info)

        x_all.append(x_p)
        y_all.append(y_p)
        lbl_t2_paths.append(str(lbl_t2_dir / fname))
        coords_all.append(coords)
        
    with open(f'testes2/info_tiles_{split_name}.json', 'w') as f:
        json.dump(meta_tiles, f, indent=4)

    return (np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0), 
           lbl_t2_paths, coords_all)

# %%

patch_size = 64
overlap = 0.9

# --- Example Workflow ---
# Process train
root = "./deforestation_dataset/PA"
x_train_raw, y_train, yt2_paths, coords_train = process_geographic_split(root, "train", 
                                                                         patch_size=patch_size,
                                                                         overlap=overlap)
x_train, min_value_train, max_value_train = minmax_normalize(x_train_raw)

# Process valid
x_val_raw, y_val = process_geographic_split(root, "valid")
x_val, _, _ = minmax_normalize(x_val_raw, min_value_train, max_value_train)

# Process test
