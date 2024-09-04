# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:09:20 2024

@author: Marcel
"""

# %% Load Libraries

import numpy as np
from package_doc.entropy.entropy_calc import EntropyCalculator

# %% Directories

x_dir = 'teste_x/'
y_dir = 'teste_y/'

# %% Load Arrays

x_train = np.load(x_dir + 'x_train.npy')
y_train = np.load(y_dir + 'y_train.npy')

x_valid = np.load(x_dir + 'x_valid.npy')
y_valid = np.load(y_dir + 'y_valid.npy')

x_test = np.load(x_dir + 'x_test.npy')
y_test = np.load(y_dir + 'y_test.npy')

# %%

class EntropyAddiction:
    def __init__(self, input_x_dir, input_y_dir, output_x_dir, output_y_dir):
        self.input_x_dir = input_x_dir
        self.input_y_dir = input_y_dir
        self.output_x_dir = output_x_dir
        self.output_y_dir = output_y_dir

