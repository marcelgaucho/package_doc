#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:26:45 2026

@author: rotunno
"""

from osgeo import gdal
from pathlib import Path

# %%

split = 'train'
base = Path('dataset_massachusetts_mnih_mod/') / split
input_dir = base / "input"
maps_dir = base / "maps"

# %%



for img_path in sorted(input_dir.glob("*.tif*")):
    # Respective map file
    fname = img_path.name
    map_path = maps_dir / fname
    
    if not map_path.exists():
        raise ValueError(f"Respective mask not found for {fname} ...")
    
    # Open image file and get geo information    
    img_dataset = gdal.Open(str(img_path))
    img_gt = img_dataset.GetGeoTransform()
    img_proj = img_dataset.GetProjection()

    # Open map and apply image geo information to it
    map_dataset = gdal.Open(str(map_path), gdal.GA_Update)
    map_dataset.SetGeoTransform(img_gt)
    map_dataset.SetProjection(img_proj)
    map_dataset.FlushCache() # Save result