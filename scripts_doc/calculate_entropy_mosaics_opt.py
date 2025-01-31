# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:53:10 2025

@author: marce
"""

# Build mosaics for calculated entropy

# %% Imports

from package_doc.entropy.entropy_calc import EntropyClasses, EntropyGroups
from package_doc.avaliacao.mosaics import MosaicGenerator
import numpy as np
import json
from pathlib import Path

# %% Directories and data necessary to calculate mosaics

ensemble_dir = r'ensemble_dir1/'

y_dir = 'teste_y1/'

label_tiles_dir = 'dataset_massachusetts_mnih_exp/test/maps'

entropy_classes = EntropyClasses.Road

entropy_group = EntropyGroups.Test

prefix = f'mosaic_entropy_{entropy_classes}_'


# %% Create mosaics

# Load entropy array
entropy_array = np.load(ensemble_dir + f'entropy_{entropy_group}_{entropy_classes}.npy')

# Load information
with open(Path(y_dir) / 'info_tiles_test.json') as fp:   
    info_tiles_test = json.load(fp)
    
# Build and export mosaics for entropy
mosaics = MosaicGenerator(test_array=entropy_array, 
                          info_tiles=info_tiles_test, 
                          tiles_dir=label_tiles_dir,
                          output_dir=ensemble_dir)
mosaics.build_mosaics()
mosaics.export_mosaics(prefix=prefix)