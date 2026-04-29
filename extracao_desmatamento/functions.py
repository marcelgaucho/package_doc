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
       
        self.global_min_train = None
        self.global_max_train = None
        
        # Meta-information storage
        self.metadata = {'train': {}, 'valid': {}, 'test': {}}
        self.coords = {'train': {}, 'valid': {}, 'test': {}}
        
    def _find_train_global_stats(self, nodata_value):
        """
        Pass 1: Iterates through the split to find global Min/Max per channel.
        """
        print("--- Calculating global stats for train ---")
        base = Path(self.base_path) / 'train'
        img_t1_dir = base / "image" / "t1"
        img_t2_dir = base / "image" / "t2"
        
        mins, maxs = [], []
        
        for t1_path in sorted(img_t1_dir.glob("*.tif")):
            xt1 = load_geo_file(t1_path)
            xt2 = load_geo_file(img_t2_dir / t1_path.name)
            x_combined = np.concatenate([xt1, xt2], axis=-1)
            
            # Mask valid pixels (that aren't nodata)
            is_valid = (x_combined != nodata_value).any(axis=-1) 
            
            # Get min/max for each channel in this specific tile
            mins.append(np.min(x_combined[is_valid], axis=0))
            maxs.append(np.max(x_combined[is_valid], axis=0))
        
        # Aggregate to find the global min/max across all tiles
        self.global_min_train = np.min(mins, axis=0)
        self.global_max_train = np.max(maxs, axis=0)
        print(f"Global Stats - Min: {self.global_min_train}, Max: {self.global_max_train}")
        
    def process_split(self, split_name='train', nodata_value=0):
        """Pass 2: Preprocess, extract patches, normalize and filter."""
        # First calculate global min and max in training set
        if self.global_min_train is None:
            self._find_train_global_stats(nodata_value)
        
        # Initialize extractor 
        extractor = PatchExtractor(
            patch_size=self.patch_size, 
            overlap_percent=self.overlap, 
            padding=True
        ) 
        
        # Initialize the base path object
        base = Path(self.base_path) / split_name
        
        # Define directories using pathlib / operator
        img_t1_dir = base / "image" / "t1"
        img_t2_dir = base / "image" / "t2"
        lbl_t1_dir = base / "label" / "t1"
        lbl_t2_dir = base / "label" / "t2"
        
        x_all, y_all = [], []
        meta = {'patch_size': self.patch_size, 'overlap': self.overlap, 'tile_info': []}
        
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
            if split_name in ['train', 'valid']:
                object_indexes = IndexesFinder(y_p).object_patches(threshold=0.02,
                                                                   target_class=1)
                x_p, y_p, coords = x_p[object_indexes], y_p[object_indexes], coords[object_indexes] # By Nodataless
                
                nodataless_indexes = IndexesFinder(x_p).nodataless_patches(nodata_value=0,
                                                                           nodata_tolerance=0)
                x_p, y_p, coords = x_p[nodataless_indexes], y_p[nodataless_indexes], coords[nodataless_indexes] # By Object
            
            # 5. Normalize Patches
            x_p, _, _ = minmax_normalize(x_p, self.global_min_train, self.global_max_train)
            
            # Store these specifically for this file
            tile_info = {
                "filepath": str(yt2_path),
                "shape": yt2.shape,
                "len_patches": len(x_p)
            }
            meta['tile_info'].append(tile_info)
            
            self.coords[split_name][str(yt2_path)] = coords
            
            x_all.append(x_p)
            y_all.append(y_p)
        
        # Save tile info in object
        self.metadata[split_name] = meta
        
        return (np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0))

    def export_info(self, out_path, split_name='test', coords_path=None):
        self.metadata[split_name]['coords_path'] = coords_path
        
        with open(out_path, 'w') as f:
            json.dump(self.metadata[split_name], f, indent=4)
            
        if coords_path:
            np.savez_compressed(coords_path, **self.coords[split_name])
            
    def __repr__(self):
        return f'PatchProcessor (base_path={self.base_path}, ' \
               f'patch_size={self.patch_size}, overlap={self.overlap})'

# %%

patch_size = 64
overlap = 0.9
test_tileinfo_path = 'testes2/info_tiles_test.json'
test_coords_path = 'testes2/coords_test.npz'

# --- Example Workflow ---
root = "./deforestation_dataset/PA"
patch_processor = PatchProcessor(root, patch_size, overlap)

# Process train
x_train, y_train = patch_processor.process_split("train")

# Process valid
x_val, y_val = patch_processor.process_split("valid")

# Process test and export info and coords
patch_processor.overlap = 0.75
x_test, y_test = patch_processor.process_split("test")
patch_processor.export_info(test_tileinfo_path, 'test', test_coords_path)



