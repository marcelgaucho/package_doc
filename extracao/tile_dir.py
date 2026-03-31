# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 18:05:12 2025

@author: Marcel
"""

# %% Import Libraries

import numpy as np
from pathlib import Path

from .patches import XPatches, YPatches
from .tile import tile
from .tile_type import TileType

from typing import List

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
        
        self.dir_path = dir_path
        self.tile_type = tile_type

    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        for tile_obj in self.tiles:
            tile_obj.extract_patches(patch_size=patch_size, overlap=overlap,
                                     border_patches=border_patches) 
       
        return [tile_obj.patches for tile_obj in self.tiles]
    
    def normalize_patches(self, min_value=None, max_value=None):
        ''' Normalization according to all tiles in the directory '''
        assert all(t.patches is not None for t in self.tiles), 'Patches must first be extracted with extrac_patches method'
        assert self.tile_type == TileType.X, 'Only X tiles can be normalized'
        
        # Concatenate and normalize patches
        all_patches = self.concat_patches()
        
        all_patches, min_value, max_value = all_patches.normalize(min_value=min_value, max_value=max_value)
        
        # Update patches in tiles
        patch_index = 0 # initial patch index to extract patches of a tile 
        for t in self.tiles:
            t.patches = XPatches(all_patches[patch_index : patch_index+len(t.patches.array), ...])
            patch_index += len(t.patches.array)
            
        return [t.patches for t in self.tiles], min_value, max_value          

    def concat_patches(self):
        patches = np.concatenate([t.patches.array for t in self.tiles], axis=0)
        
        if self.tile_type == TileType.X:
            return XPatches(patches)
        elif self.tile_type == TileType.Y:
            return YPatches(patches)
        
    def nodata_indexes(self, nodata_value=(255, 255, 255), nodata_tolerance=0):
        return [tile.nodata_indexes(nodata_value=nodata_value, nodata_tolerance=nodata_tolerance)
                for tile in self.tiles]
        
    def object_indexes(self, threshold_percentage=1, object_value=1):
        assert self.tile_type == TileType.Y, 'Only Y tiles can be filtered by object'
        return [tile.object_indexes(threshold_percentage=threshold_percentage, object_value=object_value) 
                for tile in self.tiles]
    
    def preprocess_reference_t2(self, ytiledir_t2: 'TileDir', dilation_px=2, 
                                erosion_px=0):
        ''' This object has reference T1, while the reference T2 is passed in a parameter  '''
        assert ytiledir_t2.tile_type == TileType.Y, 'T2 Tile Dir must be of Y type'
        assert self.tile_type == TileType.Y, 'T1 Tile Dir must be of Y type'
        
        for ytile_t1, ytile_t2 in zip(self.tiles, ytiledir_t2.tiles):
            ytile_t1.preprocess_reference_t2(ytile_t2=ytile_t2, dilation_px=dilation_px,
                                             erosion_px=erosion_px)
            
        return [ytile_t2 for ytile_t2 in ytiledir_t2.tiles]
            
        
        
    def __repr__(self):
        return f'Tile Dir {self.tile_type}: {self.dir_path}'
        
# %% Composite Class of Xs and Ys tile dirs

class XsYsTileDir:
    def __init__(self, x_tiledirs: List[TileDir], y_tiledirs: List[TileDir]):
        self.x_tiledirs = x_tiledirs
        self.y_tiledirs = y_tiledirs
        
        self.x_tiledirs_patches = []
        self.y_tiledirs_patches = []
        
    def extract_patches(self, patch_size: int=256, overlap: float=0.25, border_patches: bool=False):
        for x_tiledir in self.x_tiledirs:
            x_tiledir.extract_patches(patch_size=patch_size, overlap=overlap,
                                      border_patches=border_patches)
        
        for y_tiledir in self.y_tiledirs:
            y_tiledir.extract_patches(patch_size=patch_size, overlap=overlap,
                                      border_patches=border_patches)
            
        return [x_tiledir.concat_patches() for x_tiledir in self.x_tiledirs], [y_tiledir.concat_patches() for y_tiledir in self.y_tiledirs]
        
    def filter_nodata(self, tiledir_base: TileDir, nodata_value=(255, 255, 255), nodata_tolerance=0):
        # Nodata indexes to filter
        nodata_indexes = tiledir_base.nodata_indexes(nodata_value=nodata_value, nodata_tolerance=nodata_tolerance)

        # Update tiles patches inside tiles of directories
        for x_tiledir in self.x_tiledirs:
            for i, x_tile in enumerate(x_tiledir.tiles):
                x_tile.patches = XPatches(x_tile.patches.array[nodata_indexes[i]])
                
        for y_tiledir in self.y_tiledirs:
            for i, y_tile in enumerate(y_tiledir.tiles):
                y_tile.patches = YPatches(y_tile.patches.array[nodata_indexes[i]])
        
        return [x_tiledir.concat_patches() for x_tiledir in self.x_tiledirs], [y_tiledir.concat_patches() for y_tiledir in self.y_tiledirs]
        
    def filter_object(self, tiledir_base: TileDir, threshold_percentage=1, object_value=1):
        assert tiledir_base.tile_type == TileType.Y, 'Tile directory must be of Y Type' 
        
        # Object indexes to filter
        object_indexes = tiledir_base.object_indexes(threshold_percentage=threshold_percentage, object_value=object_value)
        
        # Update tiles patches inside tiles of directories
        for x_tiledir in self.x_tiledirs:
            for i, x_tile in enumerate(x_tiledir.tiles):
                x_tile.patches = XPatches(x_tile.patches.array[object_indexes[i]])
                
        for y_tiledir in self.y_tiledirs:
            for i, y_tile in enumerate(y_tiledir.tiles):
                y_tile.patches = YPatches(y_tile.patches.array[object_indexes[i]])
       
        return [x_tiledir.concat_patches() for x_tiledir in self.x_tiledirs], [y_tiledir.concat_patches() for y_tiledir in self.y_tiledirs]
        
    def xy_concat_patches(self):
        return self.x_tiledir.concat_patches(), self.y_tiledir.concat_patches()