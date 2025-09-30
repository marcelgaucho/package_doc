# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:25:36 2025

@author: Marcel
"""

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
    
# %% Create parser

parser = argparse.ArgumentParser(description='Train an ensemble of models')

# %% Add arguments

# Input dirs
parser.add_argument('x_dir', type=str, help='Enter the X directory for training')
parser.add_argument('y_dir', type=str, help='Enter the Y directory for training')

# Output base dir
parser.add_argument('-od', '--out_dir', type=str, help='Enter the output directory for the trained models',
                    required=True)

# Batch size
parser.add_argument('-b', '--batch', metavar='batch_size', type=int, help='Enter the batch size used for training',
                    default=16)

# Model name
parser.add_argument('-m', '--model', metavar='model_name', type=str, help='Enter the model name to use in training',
                    choices=['unet', 'resunet', 'unetr', 'segformer_b5'])

# Early Stopping Epochs
parser.add_argument('-px', '--patience', metavar='early_stopping_epochs', type=int, 
                    help='Enter the number of epochs to early stopping',
                    default=25)

# Loss function
parser.add_argument('-l', '--loss', metavar='loss_function', type=str, 
                    help='Enter the loss function name',
                    choices=['cross', 'custom'])

# Number of models in ensemble
parser.add_argument('--n_models', type=int, help='Enter the number of models in the ensemble',
                    default=5)

# %% Parse args 

# args = parser.parse_args()
args = parser.parse_args(['experimentos1/x_dir',
                          'experimentos1/y_dir',
                          '-od', 'experimentos1/out/resunet',
                          '-b', '2',
                          '-m', 'resunet',
                          '-px', '3',
                          '-l', 'cross',
                          '--n_models', '2'])

# X and Y directories
x_dir = Path(args.x_dir)
y_dir = Path(args.y_dir)

# Output directory
outputs_dir = Path(args.out_dir)

# Batch size
batch_size = args.batch

# Model name
model_name = args.model

# Early Stopping Epochs
early_stopping_epochs = args.patience

# Loss function
loss = args.loss

# Number of models
n_models = args.n_models

# %% Input shape and numper of classes

# Get array without opening array into memory
with open(x_dir / 'x_train.npy', 'rb') as f:
    major, minor = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    
input_shape = shape[1:] # (patch_size, patch_size, channels)
n_classes = 2

# %% Imports from package

from package_doc.treinamento.trainer import ModelTrainer
from package_doc.treinamento.f1_metric import F1Score, RelaxedF1Score
from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict
from package_doc.treinamento.custom_loss import CustomEntropyLoss

# %% Loss Function

loss_fn = CategoricalCrossentropy() if loss == 'cross' else CustomEntropyLoss()

# %% Train loop

# Training loop of models
for i in range(n_models):
    # Output dir
    out_dir = outputs_dir / f'm_{i}' # simple name of dir
    out_dir.mkdir(exist_ok=True)
    
    # Build model and optimizer
    model = build_model(input_shape, n_classes, model_type=model_name, config_dict=config_dict)
    optimizer = Adam() 
    
    # Model trainer
    model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=out_dir, model=model,
                                 optimizer=optimizer)
    
    result = model_trainer.train_with_loop(epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                           metrics_train=[RelaxedF1Score(), Precision(class_id=1), Recall(class_id=1)],
                                           metrics_val=[RelaxedF1Score(), Precision(class_id=1), Recall(class_id=1)],
                                           learning_rate=0.001, 
                                           loss_fn=loss_fn,
                                           buffer_shuffle=None, batch_size=batch_size,
                                           data_augmentation=True, augment_batch_factor=2)
    del model_trainer
    # time.sleep(18000) # Comes inactive for 5 hours







