# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:28:47 2025

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from pathlib import Path
import pdb

import argparse

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

parser = argparse.ArgumentParser(description='Train model')

# %% Add arguments

# Input dirs
parser.add_argument('x_dir', type=str, help='Enter the X directory for training')
parser.add_argument('y_dir', type=str, help='Enter the Y directory for training')

# Output dir
parser.add_argument('-od', '--out_dir', type=str, help='Enter the output directory for the trained model',
                    required=True)

# Patch size and number of channels
parser.add_argument('-p', '--patch', metavar='patch_size', type=int, help='Enter the patch size of train data',
                    default=256)
parser.add_argument('-c', '--channels', metavar='number_of_channels', type=int, help='Enter the number of channels of train data')

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


# %% Parse args 

args = parser.parse_args()
# args = parser.parse_args(['experimentos/x_dir',
#                           'experimentos/y_dir',
#                           '-od', 'experimentos/saidas1/resunet_0',
#                           '-p', '256',
#                           '-c', '3',
#                           '-b', '2',
#                           '-m', 'resunet',
#                           '-px', '3',
#                           '-l', 'cross'])

# X and Y directories
x_dir = args.x_dir
y_dir = args.y_dir

# Output directory
output_dir = args.out_dir

# Patch size and number of channels
patch_size = args.patch
channels = args.channels

# Batch size
batch_size = args.batch

# Model name
model_name = args.model

# Early Stopping Epochs
early_stopping_epochs = args.patience

# Loss function
loss = args.loss

# %% Input shape and numper of classes

input_shape = (patch_size, patch_size, channels)
n_classes = 2

# %% Imports from package

from package_doc.treinamento.trainer import ModelTrainer
from package_doc.treinamento.f1_metric import F1Score, RelaxedF1Score
from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict
from package_doc.treinamento.custom_loss import CustomEntropyLoss

# %% Loss Function

loss_fn = CategoricalCrossentropy() if loss == 'cross' else CustomEntropyLoss()

# %% Build model

model = build_model(input_shape, n_classes, model_type=model_name, config_dict=config_dict)

# %% Build optimizer

#pdb.set_trace()
optimizer = Adam() 

# %% Train Model with Loop

if not Path(output_dir).exists():
    Path(output_dir).mkdir()
    
model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=output_dir, model=model,
                             optimizer=optimizer)
result = model_trainer.train_with_loop(epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                       metrics_train=[RelaxedF1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       metrics_val=[RelaxedF1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       learning_rate=0.001, 
                                       loss_fn=loss_fn,
                                       buffer_shuffle=None, batch_size=batch_size,
                                       data_augmentation=True, augment_batch_factor=2)
del model_trainer