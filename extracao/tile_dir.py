# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 18:05:12 2025

@author: Marcel
"""

# %% Import Libraries

import numpy as np
from pathlib import Path

from .patches import XPatches, YPatches
from .tile import TileType, tile

# %% Aggregate class of tiles of certain type (XTile or YTile)

class TileDir:
    def __init__(self, dir_path: str, tile_type: TileType):
        dir_path = Path(dir_path)
        self.tiles = [str(tile_path)
                      for tile_path in dir_path.iterdir()
                      if tile_path.suffix == '.tiff' or
                      tile_path.suffix == '.tif']
        self.tiles.sort()
        self.tiles = [tile(tile_path, tile_type) for 
                      tile_path in self.tiles]
        
        self.tiles_patches = None 
        
        self.dir_path = dir_path
        self.tile_type = tile_type

    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        tiles_patches = []
        for tile_obj in self.tiles:
            if self.tile_type == TileType.X:
                tiles_patches.append( tile_obj.extract_patches(patch_size=patch_size, overlap=overlap,
                                                          border_patches=border_patches).tile_patches ) 
            elif self.tile_type == TileType.Y:
                tiles_patches.append( tile_obj.extract_patches(patch_size=patch_size, overlap=overlap,
                                                          border_patches=border_patches).tile_patches ) 
       
        self.tiles_patches = tiles_patches # Store in object
        
        return self

    def concat_patches(self):
        patches = np.concatenate([p.patches for p in self.tiles_patches], axis=0)
        
        if self.tile_type == TileType.X:
            return XPatches(patches)
        elif self.tile_type == TileType.Y:
            return YPatches(patches)
        
# %% Composite Class of X and Y tile dirs

class XYTileDir:
    def __init__(self, x_tiledir: TileDir, y_tiledir: TileDir):
        self.x_tiledir = x_tiledir
        self.y_tiledir = y_tiledir
        
    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        self.x_tiledir.extract_patches(patch_size=patch_size, overlap=overlap,
                                  border_patches=border_patches)
        self.y_tiledir.extract_patches(patch_size=patch_size, overlap=overlap,
                                       border_patches=border_patches)
        
        return self
        
    def filter_nodata(self, tile_type: TileType, nodata_value=(255, 255, 255), nodata_tolerance=0):
        x_tiles_patches_filtered, y_tiles_patches_filtered = [], []        
        
        for x_tile_patches, y_tile_patches in zip(self.x_tiledir.tiles_patches, self.y_tiledir.tiles_patches):
            if tile_type == TileType.X:
                nodata_indexes = x_tile_patches.nodata_indexes(nodata_value=nodata_value, nodata_tolerance=nodata_tolerance)
            elif tile_type == TileType.Y:
                nodata_indexes = y_tile_patches.nodata_indexes(nodata_value=nodata_value, nodata_tolerance=nodata_tolerance)
                
            x_tile_patches_filtered = XPatches(x_tile_patches.patches[nodata_indexes])
            y_tile_patches_filtered = YPatches(y_tile_patches.patches[nodata_indexes])
            
            x_tiles_patches_filtered.append(x_tile_patches_filtered)
            y_tiles_patches_filtered.append(y_tile_patches_filtered)
            
        self.x_tiledir.tiles_patches = x_tiles_patches_filtered
        self.y_tiledir.tiles_patches = y_tiles_patches_filtered
        
        return self
        
    def filter_object(self, threshold_percentage=1, object_value=1):
        x_tiles_patches_filtered, y_tiles_patches_filtered = [], []  
        
        for x_tile_patches, y_tile_patches in zip(self.x_tiledir.tiles_patches, self.y_tiledir.tiles_patches):
            object_indexes = y_tile_patches.object_indexes(threshold_percentage=threshold_percentage, object_value=object_value)

            x_tile_patches_filtered = XPatches(x_tile_patches.patches[object_indexes])
            y_tile_patches_filtered = YPatches(y_tile_patches.patches[object_indexes])
                
            x_tiles_patches_filtered.append(x_tile_patches_filtered)
            y_tiles_patches_filtered.append(y_tile_patches_filtered)
            
        self.x_tiledir.tiles_patches = x_tiles_patches_filtered
        self.y_tiledir.tiles_patches = y_tiles_patches_filtered
        
        return self
        
    def xy_concat_patches(self):
        return self.x_tiledir.concat_patches(), self.y_tiledir.concat_patches()