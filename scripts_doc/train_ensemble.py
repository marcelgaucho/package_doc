# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:25:36 2025

@author: Marcel
"""

# Train an ensemble of models

# %% Import Libraries

import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from pathlib import Path
import pdb

import argparse
import time

# %% Limit GPU Memory or, in case of no gpu available, limit number of threads used

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

num_threads = 1
if gpus:
    tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20480)])
else:
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
# %% Data for Script

batch_size = 16
model_type = 'resunet'
early_stopping_epochs = 2
n_models = 2

# %% Input and output directories

x_dir = r'experimentos_deforestation/x_dir/'
y_dir = r'experimentos_deforestation/y_dir/'
outputs_dir = fr'experimentos_deforestation/out_{model_type}/'

# %%

x_dir = Path(x_dir)
y_dir = Path(y_dir)
outputs_dir = Path(outputs_dir)

# %% Input shape and numper of classes

# Get array without opening array into memory
with open(x_dir / 'x_train.npy', 'rb') as f:
    major, minor = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    
input_shape = shape[1:] # (patch_size, patch_size, channels)
n_classes = 2

# %% Input shape and numper of classes

# Get array without opening array into memory
with open(x_dir / 'x_train.npy', 'rb') as f:
    major, minor = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    
input_shape = shape[1:] # (patch_size, patch_size, channels)
n_classes = 2

# %% Imports from package

from package_doc.treinamento.trainer import ModelTrainer
from package_doc.treinamento.metrics import MaskedPrecision, MaskedRecall, MaskedF1Score
from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict
from package_doc.treinamento.custom_loss import masked_weighted_cce

# %% Loss Function

weights = [0.4, 2.0]
weighted_cross = masked_weighted_cce(weights)

# %% Create output dir of models

outputs_dir.mkdir(exist_ok=True)

# %% Train loop

# Training loop of models
for i in range(n_models):
    # Output dir
    out_dir = outputs_dir / f'm_{i}' # simple name of dir
    out_dir.mkdir(exist_ok=True)
    
    # Build model and optimizer
    model = build_model(input_shape, n_classes, model_type=model_type, config_dict=config_dict)
    optimizer = Adam() 
    
    # Model trainer
    model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=out_dir, model=model,
                                 optimizer=optimizer)
    
    result = model_trainer.train_with_loop(epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                           metrics_train=[MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
                                           metrics_val=[MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
                                           learning_rate=0.0001, 
                                           loss_fn=weighted_cross,
                                           buffer_shuffle=None, batch_size=batch_size,
                                           data_augmentation=True, augment_batch_factor=2)
    
    # time.sleep(3600) # Comes inactive for 1 hour







