# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:44:53 2024

@author: Marcel
"""

import numpy as np
from tensorflow.image import extract_patches
# from package_doc.functions_extract import extract_patches


# %%

x = np.moveaxis(np.arange(48).reshape((4, 4, 3)), 0, 2)[np.newaxis, ...]
x = np.arange(27).reshape((3, 3, 3))[np.newaxis, ...]

# %%

x = np.arange(9).reshape((3, 3, 1))[np.newaxis, ...]

# %%

x = np.stack((np.arange(0, 16).reshape(4, 4), np.arange(16, 32).reshape(4, 4), np.arange(32, 48).reshape(4, 4)), axis=2)
x = x[np.newaxis, ...]


# %%

dim = 3
channels = 3
sizes = [1, dim, dim, 1]
strides = [1, dim, dim, 1]
rates = [1, 1, 1, 1]
padding = 'SAME'

# %%

# Extract in shape (Image Number, Patches Lines Number, Patches Columns Number, Flattened Patch Length)
patches = extract_patches(x, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()

# Reshape to (Image Number, Patches Number, Flattened Patch Length)
img_number = patches.shape[0]
n_vertical_subpatches = patches.shape[1]
n_horizontal_subpatches = patches.shape[2]
patches = patches.reshape((img_number, n_vertical_subpatches*n_horizontal_subpatches, -1))

# Reshape to (Image Number, Patches Number, Patch Height, Patch Width, Patch Channels)
patches = patches.reshape((img_number, n_vertical_subpatches*n_horizontal_subpatches, dim, dim, channels))



# %%

n_vertical_subpatches = patches.shape[1]
n_horizontal_subpatches = patches.shape[2]
patches_res = patches.reshape((1, n_vertical_subpatches*n_horizontal_subpatches, 
                            dim, dim, 3))
patches_res_0_0 = patches_res[0, 0, :, :, :]