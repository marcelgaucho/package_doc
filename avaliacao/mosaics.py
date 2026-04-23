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
from ..extracao.patch_merging import MidpointPatchMerger
from pathlib import Path

# %% Class to generate mosaics

class MosaicGenerator:
    def __init__(self, test_array, info_tiles, tiles_dir, output_dir):
        self.test_array = test_array
        
        self.len_tiles = info_tiles['len_tiles']
        self.shape_tiles = info_tiles['shape_tiles']
        self.patch_sizes = info_tiles['patch_sizes']
        self.overlap = info_tiles['overlap']
        
        self.coords = np.load(info_tiles['coords_path'])['coords']
        
        self.tiles_dir = Path(tiles_dir)
        
        self.output_dir = Path(output_dir)
        
        self.mosaics = None        
        
    def _set_labels_paths(self):
        self.labels_paths = [path for path in self.tiles_dir.iterdir() 
                             if path.suffix=='.tiff' or path.suffix=='.tif']
        self.labels_paths.sort()
        
    def build_mosaics(self, dtype=np.uint8):
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
                                            :, :, :]
            
            merger = MidpointPatchMerger(target_shape=self.shape_tiles[i_mosaic], 
                                         patch_size=self.patch_sizes[i_mosaic], 
                                         overlap_percent=self.overlap[i_mosaic])
            
            mosaic = merger(patches=patches_mosaic, coords=self.coords[i_mosaic],
                            dtype=dtype)
            
            mosaics.append(mosaic)
            
            i_tile_start += self.len_tiles[i_mosaic] # Update index where tile starts
            
        self.mosaics = stack_uneven(mosaics) # Transform mosaics list to array
            
        return mosaics
    
    def save_mosaics(self, prefix='pred'):
        np.save(self.output_dir / f'{prefix}_mosaics.npy', self.mosaics)
        
    def export_mosaics(self, prefix='outmosaic'):
        self._set_labels_paths() # Set list of paths of reference tiles
        
        # Export predictions mosaics
        for mosaic, label_path in zip(self.mosaics, self.labels_paths):
            mosaic = mosaic[..., 0] # Suppress channel dimension
            outmosaic_basename = prefix + '_' + label_path.stem + '.tif'
            outmosaic_path = str(self.output_dir / outmosaic_basename)

            save_raster_reference(in_raster_path=label_path,
                                  out_raster_path=outmosaic_path, 
                                  array_exported=mosaic)