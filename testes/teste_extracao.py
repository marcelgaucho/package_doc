# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:19:43 2024

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
import numpy as np
from pathlib import Path


# %% Directories

x_dir_extracao = 'teste_x_extracao/'
y_dir_extracao = 'teste_y_extracao/'

img_tiles_dir_train = r'dataset_massachusetts_mnih_exp/train/input/'
img_tiles_dir_valid = r'dataset_massachusetts_mnih_exp/validation/input/'
img_tiles_dir_test = r'dataset_massachusetts_mnih_exp/test/input/'


label_tiles_dir_train = r'dataset_massachusetts_mnih_exp/train/maps/'
label_tiles_dir_valid = r'dataset_massachusetts_mnih_exp/validation/maps/'
label_tiles_dir_test = r'dataset_massachusetts_mnih_exp/test/maps/'

# %%

class PatchExtractor:
    def __init__(self, img_tiles_dir_train, label_tiles_dir_train,
                       img_tiles_dir_valid, label_tiles_dir_valid,
                       img_tiles_dir_test, label_tiles_dir_test):
        # Train tiles paths
        self.img_paths_train = [str(path) for path in Path(img_tiles_dir_train).iterdir() 
                                if path.suffix=='.tiff' or path.suffix=='.tif']
        self.label_paths_train = [str(path) for path in Path(label_tiles_dir_train).iterdir() 
                                  if path.suffix=='.tiff' or path.suffix=='.tif']
        self.img_paths_train.sort()
        self.label_paths_train.sort()
        
        # Valid tiles paths
        self.img_paths_valid = [str(path) for path in Path(img_tiles_dir_valid).iterdir() 
                                if path.suffix=='.tiff' or path.suffix=='.tif']        
        self.label_paths_valid = [str(path) for path in Path(label_tiles_dir_valid).iterdir() 
                                  if path.suffix=='.tiff' or path.suffix=='.tif']
        self.img_paths_valid.sort()
        self.label_paths_valid.sort()
        
        # Test tiles paths
        self.img_paths_test = [str(path) for path in Path(img_tiles_dir_test).iterdir() 
                               if path.suffix=='.tiff' or path.suffix=='.tif']        
        self.label_paths_test = [str(path) for path in Path(label_tiles_dir_test).iterdir() 
                                 if path.suffix=='.tiff' or path.suffix=='.tif']
        self.img_paths_test.sort()
        self.label_paths_test.sort()
    
    @staticmethod
    def load_tile(tile_path):
        dataset = gdal.Open(tile_path)
        image = dataset.ReadAsArray()
        # Move to channel-last if the image is multiband
        if len(image.shape) > 2:
            image = np.moveaxis(image, 0, 2)
        # Create channel dimension for monochromatic images 
        else:
            image = np.expand_dims(image, 2)
        
        return image
    
    def extract_patches(tile):
        pass
        
# %% Testa classe

extractor = PatchExtractor(img_tiles_dir_train=img_tiles_dir_train, label_tiles_dir_train=label_tiles_dir_train,
                           img_tiles_dir_valid=img_tiles_dir_valid, label_tiles_dir_valid=label_tiles_dir_valid,
                           img_tiles_dir_test=img_tiles_dir_test, label_tiles_dir_test=label_tiles_dir_test)
img0 = extractor.img_paths_train[0]
label0 = extractor.label_paths_train[0]

# %%

img0_np = PatchExtractor.load_tile(img0)
label0_np = PatchExtractor.load_tile(label0)






        
    