# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:52:24 2024

@author: Marcel
"""

# Replace values in tiff files in a y directory

# %% Import Libraries

from package_doc.extracao.replace_value import replace_value_y_files

# %% Y Directory

y_dir = r'dataset_massachusetts_mnih_exp/train/maps/'

# %% Value to replace

old_value = 255
new_value = 1

# %% Replace value

replace_value_y_files(y_dir=y_dir, old_value=old_value, new_value=new_value)