#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:37:12 2026

@author: rotunno
"""

# Fine tune a model according to the model and X, Y data

# Various parameters must be passed

# The batch_size is the input batch size to the training,
# which will in practice be increased by data augmentation, 
# multiplying it by augment_batch_factor. 

# The model_type must be one of the types: "unet",
# "resunet", "unetr", "segformer_b5"

# The early_stopping_epochs is the number of epochs used
# to early stopping the training

# The input_shape is the shape of the patch used for
# training

# The n_classes is the number of classes used in 
# classification, e.g, for road and background use 2


# %% Imports

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from pathlib import Path
import pdb
import numpy as np

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
reduce_on_plateau = True

# %% Input and output directories

x_dir = r'experimentos_deforestation/x_dir/'
y_dir = r'experimentos_deforestation/y_dir/'
output_dir = r'experimentos_deforestation/out_resunet/fine_tuned/t_0/'
entropy_dir = 'experimentos_deforestation/out_resunet/uncertainty'  # Include uncertainty dir if it's second net
model_path = r'experimentos_deforestation/out_resunet/m_0/best_model.keras'


# %%

x_dir = Path(x_dir)
y_dir = Path(y_dir)
output_dir = Path(output_dir)

# %% Input shape and numper of classes

# Get array without opening array into memory
with open(x_dir / 'x_train.npy', 'rb') as f:
    major, minor = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    
input_shape = shape[1:] # (patch_size, patch_size, channels)
n_classes = 2

# %% Imports from package

from package_doc.treinamento.model_trainer import ModelTrainer
from package_doc.treinamento.metrics import CustomF1Score, RelaxedF1Score, MaskedPrecision, MaskedRecall, MaskedF1Score
from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict
from package_doc.treinamento.custom_loss import CustomEntropyLoss, masked_weighted_cce, custom_entropy_loss, masked_cce
from package_doc.treinamento.fine_tuning import LayerIndexStrategy


# %% Build categorical cross-entropy

weights = [0.4, 2.0]
weighted_cross = masked_weighted_cce(weights)

# %% Build model

model = build_model(input_shape, n_classes, model_type=model_type, config_dict=config_dict)

# %% Train Model with Loop

if not Path(output_dir).exists():
    Path(output_dir).mkdir()

# Build object    
model_trainer = ModelTrainer(x_dir=x_dir, output_dir=output_dir, model=model, optimizer=Adam())

# Stage 2: Pick a flexible Strategy pattern and execute
fine_tune_config = LayerIndexStrategy(fine_tune_at=31, learning_rate=1e-5)
result_finetune = model_trainer.fine_tune(model_path,
                                          strategy=fine_tune_config, epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                          metrics_train=[MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
                                          metrics_val=[MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
                                          loss_fn=custom_entropy_loss,
                                          buffer_shuffle=None, batch_size=batch_size,
                                          data_augmentation=True, augment_batch_factor=2,
                                          reduce_on_plateau=reduce_on_plateau,
                                          entropy_dir=entropy_dir)
del model_trainer

                  
                                      
                                      
                                      


