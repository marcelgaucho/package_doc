# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:44:56 2024

@author: Marcel
"""

# Class to be used to generate mosaics in test

# %% Imports

import numpy as np
from .mosaic_utils import unpatch_reference, save_raster_reference
from .utils import stack_uneven
from pathlib import Path

# %% Class to generate mosaics

class MosaicGenerator:
    def __init__(self, test_array, info_tiles, tiles_dir, output_dir):
        self.test_array = test_array
        
        self.len_tiles = info_tiles['len_tiles']
        self.shape_tiles = info_tiles['shape_tiles']
        self.stride_tiles = info_tiles['stride_tiles']
        
        self.tiles_dir = tiles_dir
        
        self.output_dir = output_dir
        
        self.mosaics = None
        
    def _set_labels_paths(self):
        self.labels_paths = [str(path) for path in Path(self.tiles_dir).iterdir() 
                             if path.suffix=='.tiff' or path.suffix=='.tif']
        self.labels_paths.sort()
        
    def build_mosaics(self):
        # Total number of mosaics
        n_mosaics = len(self.shape_tiles)
        
        # Index where mosaic begins
        i_tile_start = 0
        
        # List with the predicted mosaics
        mosaics = []
        
        # Build mosaics
        for i_mosaic in range(n_mosaics):
            print(f'Building Mosaic {i_mosaic+1:>5d}/{n_mosaics:>5d}')
            
            patches_mosaic = self.test_array[i_tile_start:i_tile_start+self.len_tiles[i_mosaic],
                                            :, :, 0]
            
            mosaic = unpatch_reference(patches=patches_mosaic, 
                                       stride=self.stride_tiles[i_mosaic], 
                                       reference_shape=self.shape_tiles[i_mosaic],
                                       border_patches=True)
            
            mosaics.append(mosaic)
            
            i_tile_start += self.len_tiles[i_mosaic] # Update index where tile starts
            
        self.mosaics = mosaics
            
        return mosaics
    
    def save_mosaics(self, prefix='pred'):
        mosaics = stack_uneven(self.mosaics)[..., np.newaxis] # Transform mosaics list to array
        
        np.save(Path(self.output_dir)/f'{prefix}_mosaics.npy', mosaics)
        
    def export_mosaics(self, prefix='outmosaic'):
        self._set_labels_paths() # Set list of paths of reference tiles
        
        # Export predictions mosaics
        for mosaic, label_path in zip(self.mosaics, self.labels_paths):
            outmosaic_basename = prefix + '_' + Path(label_path).stem + '.tif'
            outmosaic_path = str(Path(self.output_dir)/outmosaic_basename)

            save_raster_reference(in_raster_path=label_path,
                                  out_raster_path=outmosaic_path, 
                                  array_exported=mosaic)