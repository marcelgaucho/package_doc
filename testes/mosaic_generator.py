# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:01:52 2024

@author: marcel.rotunno
"""

# %% Import Libraries

from osgeo import gdal
import numpy as np
import pickle
from pathlib import Path
import os
from package_doc.avaliacao.mosaic_functions import unpatch_reference
from package_doc.avaliacao.save_functions import save_raster_reference
from package_doc.avaliacao.compute_functions import stack_uneven

# %% Directories   

ensemble_dir = r'ensemble_dir/'
labels_dir = r'dataset_massachusetts_mnih_mod/test/maps'   


# %% Load entropy

test_entropy = np.load(ensemble_dir+'entropy_test.npy')

# %% Load info pickle

with open(ensemble_dir + 'info_tiles_test.pickle', "rb") as fp:   
    info = pickle.load(fp)

# %%

class MosaicGenerator:
    def __init__(self, test_array, info_tiles, tiles_dir, output_dir):
        self.test_array = test_array
        
        self.length_tiles = info_tiles['len_tiles_test']
        self.shape_tiles = info_tiles['shape_tiles_test']
        self.stride = info_tiles['patch_stride_test']
        
        self.tiles_dir = tiles_dir
        
        self.output_dir = output_dir
        
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
        pred_mosaics = []
        
        # Build mosaics
        for i_mosaic in range(n_mosaics):
            print(f'Building Mosaic {i_mosaic+1:>5d}/{n_mosaics:>5d}')
            
            patches_mosaic = self.test_array[i_tile_start:i_tile_start+self.length_tiles[i_mosaic],
                                            :, :, 0]
            
            pred_mosaic = unpatch_reference(reference_batches=patches_mosaic, 
                                            stride=self.stride, 
                                            reference_shape=self.shape_tiles[i_mosaic],
                                            border_patches=True)
            
            pred_mosaics.append(pred_mosaic)
            
            i_tile_start += self.length_tiles[i_mosaic] # Update index where tile starts
            
        self.pred_mosaics = pred_mosaics
            
        return pred_mosaics
    
    def save_mosaics(self):
        pred_mosaics = stack_uneven(self.pred_mosaics)[..., np.newaxis] # Transform mosaics list to array
        
        np.save(Path(self.output_dir)/'pred_mosaics.npy', pred_mosaics)
        
    def export_mosaics(self, prefix='outmosaic'):
        self._set_labels_paths() # Set list of paths of reference tiles
        
        # Export predictions mosaics
        for mosaic, label_path in zip(self.pred_mosaics, self.labels_paths):
            outmosaic_basename = prefix + '_' + Path(label_path).stem + '.tif'
            outmosaic_path = str(Path(self.output_dir)/outmosaic_basename)

            save_raster_reference(in_raster_path=label_path,
                                  out_raster_path=outmosaic_path, 
                                  array_exported=mosaic)
            

        
            
        
            
            
        

# %%
p = list(Path(labels_dir).glob('*.tiff'))
labels_paths = [os.path.join(labels_dir, arq) for arq in os.listdir(labels_dir)]

labels_paths = [str(path) for path in Path(labels_dir).iterdir() if path.suffix=='.tiff' or path.suffix=='.tif']
labels_paths.sort()

# %% Testa classe

mosaic_generator = MosaicGenerator(test_array=test_entropy, info_tiles=info, tiles_dir=labels_dir,                                  
                                   output_dir=ensemble_dir)
mosaics = mosaic_generator.build_mosaics()
mosaic_generator.save_mosaics()

# %%

mosaic_generator.export_mosaics(prefix='outmosaic_entropy_back')