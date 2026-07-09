#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:15:01 2026

@author: rotunno
"""
# %% Disable GPU and limit number of CPU threads (and thus the number of cpu cores) to use in evaluation

from osgeo import gdal # First import gdal when it gives error
import os
import pdb
#os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use Keras 2

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Print 0 gpus (gpu disabled)

cpu_threads = 4

print(f"Number of CPU threads used in evaluation: {cpu_threads}") 

tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)

# %% Import libraries

import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from package_doc.extracao.patch_extraction import PatchExtractor
from package_doc.extracao_desmatamento.filtering import IndexesFinder
from package_doc.extracao_desmatamento.utils import (load_geo_file, 
                                                     preprocess_reference_t2,
                                                     preprocess_reference_t2_nobuffer,
                                                     minmax_normalize,
                                                     export_to_geotiff)
from package_doc.extracao_desmatamento.utils import save_dataset, onehot

from skimage.morphology import disk, dilation, erosion

# %%

class RoadPatchProcessor:
    def __init__(self, base_path, patch_size=64, overlap=0.9):
        self.base_path = Path(base_path)
        self.patch_size = patch_size
        self.overlap = overlap
        
        self.global_min_train = None
        self.global_max_train = None
        
        # Meta-information storage
        self.metadata = {'train': {}, 'valid': {}, 'test': {}}
        self.coords = {'train': {}, 'valid': {}, 'test': {}}
        
        # Global min and max are known for the Massachusetts dataset
        self.global_min_train = np.array([0.0, 0.0, 0.0], dtype=np.uint8)
        self.global_max_train = np.array([255.0, 255.0, 255.0], dtype=np.uint8)
        
    def _find_train_global_stats(self, nodata_value):
        """
        Pass 1: Iterates through the train split to find global Min/Max per channel.
        """
        print("--- Calculating global stats for train ---")
        train_input_dir = self.base_path / 'train' / 'input'
        
        mins, maxs = [], []
        
        # Checking for both .tif and .tiff
        for img_path in sorted(train_input_dir.glob("*.tif*")):
            x_img = load_geo_file(img_path)
            
            # Mask valid pixels (that aren't nodata)
            is_valid = (x_img != nodata_value).any(axis=-1) 
            
            if is_valid.any():
                # Get min/max for each channel in this specific tile
                mins.append(np.min(x_img[is_valid], axis=0))
                maxs.append(np.max(x_img[is_valid], axis=0))
        
        # Aggregate to find the global min/max across all tiles
        self.global_min_train = np.min(mins, axis=0)
        self.global_max_train = np.max(maxs, axis=0)
        print(f"Global Stats - Min: {self.global_min_train}, Max: {self.global_max_train}")
        
    def process_split(self, split_name='train', nodata_value=0, normalize=True, min_road_ratio=0.05, max_patches=None):
        """Pass 2: Extract patches, normalize and filter."""
        if self.global_min_train is None: 
            self._find_train_global_stats(nodata_value)
        
        # Initialize extractor 
        extractor = PatchExtractor(
            patch_size=self.patch_size, 
            overlap_percent=self.overlap, 
            padding=True
        ) 
        
        base = self.base_path / split_name
        input_dir = base / "input"
        maps_dir = base / "maps"
        
        x_all, y_all = [], []
        meta = {'patch_size': self.patch_size, 
                'overlap': self.overlap, 
                'global_min_train': self.global_min_train,
                'global_max_train': self.global_max_train,
                'tile_info': []}
        
        for img_path in sorted(input_dir.glob("*.tiff")):
            fname = img_path.name
            
            # The mask might have a different extension (.tif vs .tiff). Ensure exact match.
            map_path = maps_dir / fname
            if not map_path.exists():
                raise ValueError(f"Warning: Mask not found for {fname}, skipping...")
            
            # Load Image and Mask
            x_img = load_geo_file(img_path)
            y_mask = load_geo_file(map_path)
            
            # Extract Patches
            x_p, coords = extractor(x_img, get_coords=True)
            y_p         = extractor(y_mask, get_coords=False)
            
            # Filter Patches (Optional: adjust target_class/threshold based on roads)
            if split_name in ['train', 'valid']:
                # Filter out patches with no roads if desired
                object_indexes = IndexesFinder(y_p).object_patches(threshold=min_road_ratio, target_class=1)
                x_p, y_p, coords = x_p[object_indexes], y_p[object_indexes], coords[object_indexes]
                
                # Filter out no-data patches
                nodataless_indexes = IndexesFinder(x_p).nodataless_patches(nodata_value=nodata_value, nodata_tolerance=0)
                x_p, y_p, coords = x_p[nodataless_indexes], y_p[nodataless_indexes], coords[nodataless_indexes]
            
            # Normalize Patches
            if normalize:
                x_p, _, _ = minmax_normalize(x_p, self.global_min_train, self.global_max_train)
                x_p = x_p.astype(np.float16) # Float 16 is sufficient for np.uint8 data of Massachusetts dataset
            
            # Store tile info
            tile_info = {
                "filepath": str(map_path),
                "shape": y_mask.shape,
                "len_patches": len(x_p)
            }
            meta['tile_info'].append(tile_info)
            self.coords[split_name][str(map_path)] = coords
            
            x_all.append(x_p)
            y_all.append(y_p)
        
        self.metadata[split_name] = meta
        
        # Concatenate into final numpy arrays
        x_stacked = np.concatenate(x_all, axis=0)
        y_stacked = np.concatenate(y_all, axis=0)
        
        # Subsampling to max_patches
        if max_patches is not None and len(x_stacked) > max_patches:
            print(f"--- Subsampling {split_name} split from {len(x_stacked)} down to {max_patches} patches ---")
            rng = np.random.default_rng(seed=42) # Fixed seed ensures reproducible datasets
            selected_indices = rng.choice(len(x_stacked), size=max_patches, replace=False)
            
            x_stacked = x_stacked[selected_indices]
            y_stacked = y_stacked[selected_indices]
            
        return x_stacked, y_stacked
        
        return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)

    def export_info(self, out_path, split_name='test', coords_path=None):
        self.metadata[split_name]['coords_path'] = str(coords_path) 
        
        with open(out_path, 'w') as f:
            json.dump(self.metadata[split_name], f, indent=4)
            
        if coords_path:
            np.savez_compressed(coords_path, **self.coords[split_name])
            
    def __repr__(self):
        return f'RoadPatchProcessor(base_path={self.base_path}, patch_size={self.patch_size}, overlap={self.overlap})'

# %%

# =============================================================================
# patch_size = 64
# overlap = 0.9
# test_tileinfo_path = 'testes2/info_tiles_test.json'
# test_coords_path = 'testes2/coords_test.npz'
# 
# # --- Example Workflow ---
# root = "./deforestation_dataset/PA"
# patch_processor = PatchProcessor(root, patch_size, overlap)
# 
# # Process train
# x_train, y_train = patch_processor.process_split("train")
# 
# # Process valid
# x_val, y_val = patch_processor.process_split("valid")
# 
# # Process test and export info and coords
# patch_processor.overlap = 0.75
# x_test, y_test = patch_processor.process_split("test")
# patch_processor.export_info(test_tileinfo_path, 'test', test_coords_path)
# =============================================================================


# %%
pdb.set_trace()
patch_size = 256
overlap_train = 0.25
overlap_valid = 0
overlap_test = 0.25
x_dir = 'experimentos_massachusetts/x_dir'
y_dir = 'experimentos_massachusetts/y_dir'
nodata_value = 255
min_road_ratio = 0.02
max_patches = 4000

# %%

x_dir = Path(x_dir)

# %%

y_dir= Path(y_dir)
test_tileinfo_path = y_dir / 'info_tiles_test.json'
test_coords_path = y_dir / 'coords_test.npz'

# %% --- Workflow ---

root = "./dataset_massachusetts_mnih_mod/"
patch_processor = RoadPatchProcessor(root, patch_size, overlap_train)

# Process train
x_train, y_train = patch_processor.process_split("train", 
                                                 nodata_value, 
                                                 normalize=False, 
                                                 min_road_ratio=min_road_ratio, 
                                                 max_patches=max_patches) # X train will be normalized (divided by 255) when parsed to training
# np.save(x_dir / 'x_train.npy', x_train)
# np.save(y_dir / 'y_train.npy', y_train)
save_dataset(x_dir / 'train_dataset', x_train, onehot(y_train))

# Process valid
patch_processor.overlap = overlap_valid
x_valid, y_valid = patch_processor.process_split("valid", nodata_value, normalize=False, min_road_ratio=min_road_ratio) # X valid will be normalized (divided by 255) when parsed to training
# np.save(x_dir / 'x_valid.npy', x_valid)
# np.save(y_dir / 'y_valid.npy', y_valid)
save_dataset(x_dir / 'valid_dataset', x_valid, onehot(y_valid))

# Process test and export info and coords
patch_processor.overlap = overlap_test
x_test, y_test = patch_processor.process_split("test", nodata_value, normalize=True) # Test is already normalized
np.save(x_dir / 'x_test.npy', x_test) 
np.save(y_dir / 'y_test.npy', y_test)
patch_processor.export_info(test_tileinfo_path, "test", str(test_coords_path))


