# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:37:07 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np

from package_doc.extracao_desmatamento.functions import PatchProcessor
from package_doc.extracao_desmatamento.utils import save_dataset, onehot
from pathlib import Path

# %%

patch_size = 64
overlap = 0.9
overlap_test = 0.75
x_dir = 'experimentos_deforestation/x_dir'
y_dir = 'experimentos_deforestation/y_dir'

# %%

x_dir = Path(x_dir)

# %%

y_dir= Path(y_dir)
test_tileinfo_path = y_dir / 'info_tiles_test.json'
test_coords_path = y_dir / 'coords_test.npz'

# %% --- Workflow ---

root = "./deforestation_dataset/PA"
patch_processor = PatchProcessor(root, patch_size, overlap)

# Process train
x_train, y_train = patch_processor.process_split("train")
np.save(x_dir / 'x_train.npy', x_train)
np.save(y_dir / 'y_train.npy', y_train)
save_dataset(x_dir / 'train_dataset', x_train, onehot(y_train))

# Process valid
x_valid, y_valid = patch_processor.process_split("valid")
np.save(x_dir / 'x_valid.npy', x_valid)
np.save(y_dir / 'y_valid.npy', y_valid)
save_dataset(x_dir / 'valid_dataset', x_valid, onehot(y_valid))

# Process test and export info and coords
patch_processor.overlap = overlap_test
x_test, y_test = patch_processor.process_split("test", preprocessed_path='t2tiles_prep/')
np.save(x_dir / 'x_test.npy', x_test)
np.save(y_dir / 'y_test.npy', y_test)
patch_processor.export_info(test_tileinfo_path, "test", str(test_coords_path))

