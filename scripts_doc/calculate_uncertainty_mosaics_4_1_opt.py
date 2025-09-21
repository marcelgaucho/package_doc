# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:53:10 2025

@author: marce
"""

# Build mosaics for calculated entropy

# %% Imports

from package_doc.entropy.dir_uncertain import UncertaintyMetric 
from package_doc.entropy.ensemb import DataGroups
from package_doc.avaliacao.mosaics import MosaicGenerator
import numpy as np
import json
from pathlib import Path

# %% Directories and data necessary to calculate mosaics

folder_uncertainty = Path(r'experimentos1/saidas1/uncertainty/')

metric = UncertaintyMetric.Entropy

y_dir = Path('experimentos1/y_dir')

label_tiles_dir = 'dataset_massachusetts_mnih_exp/test/maps'

prefix = f'mosaic_{metric}_'

data_group = DataGroups.Test

model_name = 'resunet'


# %% Create mosaics

# Unceratinty array
uncertainty_array = np.load(folder_uncertainty / f'{model_name}_{metric}_{data_group}.npy')

# Load information
with open(y_dir / 'info_tiles_test.json') as fp:   
    info_tiles_test = json.load(fp)
    
# Build and export mosaics for entropy
mosaics = MosaicGenerator(test_array=uncertainty_array, 
                          info_tiles=info_tiles_test, 
                          tiles_dir=label_tiles_dir,
                          output_dir=folder_uncertainty)
mosaics.build_mosaics()
mosaics.export_mosaics(prefix=prefix)