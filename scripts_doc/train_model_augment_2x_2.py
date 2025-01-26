# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

# Train a model according to the model and X, Y data

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

batch_size = 2
model_type = 'resunet'
early_stopping_epochs = 3

input_shape = (256, 256, 3)
n_classes = 2

# %% Input and output directories

# x_dir = r'entrada/'
# y_dir = r'y_directory/'
x_dir = r'teste_x2/'
y_dir = r'teste_y2/'
output_dir = fr'saida_{model_type}_loop_2x_{batch_size}b/'


# %% Imports from package

from package_doc.treinamento.trainer import ModelTrainer
from package_doc.treinamento.f1_metric import F1Score, RelaxedF1Score
from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict


# %% Build model


model = build_model(input_shape, n_classes, model_type=model_type, config_dict=config_dict)

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
                                       loss_fn=CategoricalCrossentropy(from_logits=False),
                                       buffer_shuffle=None, batch_size=batch_size,
                                       data_augmentation=True, augment_batch_factor=2)
del model_trainer

                  
                                      
                                      
                                      


