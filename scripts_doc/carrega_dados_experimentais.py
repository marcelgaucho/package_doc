# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:01:23 2024

@author: marce
"""
# %% Import Libraries

import numpy as np
import tensorflow as tf

# %% Directories   

x_dir = r'teste_x/'
y_dir = r'teste_y/'

# %% Load Numpy Arrays

x_train = np.load(x_dir + 'x_train.npy')
y_train = np.load(y_dir + 'y_train.npy')

x_valid = np.load(x_dir + 'x_valid.npy')
y_valid = np.load(y_dir + 'y_valid.npy')

# %%

train_dataset = tf.data.Dataset.load(x_dir + 'train_dataset/')
valid_dataset = tf.data.Dataset.load(x_dir + 'valid_dataset/')




