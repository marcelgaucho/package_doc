# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from pathlib import Path
import pdb


#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)

# Limita memória da GPU 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

# Limita Memória usada pela GPU 
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=20480)])


# %% Data for Script

batch_size = 16
model_type = 'unetr'
early_stopping_epochs = 25

# %% Diretórios de entrada e saída

x_dir = r'entrada/'
y_dir = r'y_directory/'
output_dirs = [fr'saida_{model_type}_loop_0x_{batch_size}b/', fr'saida_{model_type}_loop_2x_{batch_size}b/', fr'saida_{model_type}_loop_4x_{batch_size}b/']


# %%  Importa para treinamento

from package_doc.treinamento.functions_train import ModelTrainer, F1Score, transform_augment
from package_doc.treinamento.arquiteturas.modelos import build_model

# %% Constroi modelo ResUnet

input_shape = (256, 256, 3)
n_classes = 2
model = build_model(input_shape, n_classes, model_type=model_type)

# %% Constroi otimizador
#pdb.set_trace()
optimizer = Adam() 

# %% Treina Modelo com Loop (First Dir) - 0x - No Data Augmentation

output_dirs0 = output_dirs[0]
if not Path(output_dirs0).exists():
    Path(output_dirs0).mkdir(exist_ok=True)
    
    
model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=output_dirs0, model=model,
                             optimizer=optimizer)
result = model_trainer.train_with_loop(epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                       metrics_train=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       metrics_val=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       learning_rate=0.001, 
                                       loss_fn=CategoricalCrossentropy(from_logits=False),
                                       buffer_shuffle=None, batch_size=batch_size,
                                       data_augmentation=False, augment_batch_factor=2)
del model_trainer

# %% Treina Modelo com Loop (Second Dir) - 2x

optimizer = Adam() 

output_dirs1 = output_dirs[1]
if not Path(output_dirs1).exists():
    Path(output_dirs1).mkdir(exist_ok=True)
    
    
model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=output_dirs1, model=model,
                             optimizer=optimizer)
result = model_trainer.train_with_loop(epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                       metrics_train=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       metrics_val=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       learning_rate=0.001, 
                                       loss_fn=CategoricalCrossentropy(from_logits=False),
                                       buffer_shuffle=None, batch_size=batch_size,
                                       data_augmentation=True, augment_batch_factor=2)
del model_trainer

# %% Treina Modelo com Loop (Third Dir) - 4x

optimizer = Adam() 

output_dirs2 = output_dirs[2]
if not Path(output_dirs2).exists():
    Path(output_dirs2).mkdir(exist_ok=True)

model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=output_dirs2, model=model,
                             optimizer=optimizer)
result = model_trainer.train_with_loop(epochs=2000, early_stopping_epochs=early_stopping_epochs,
                                       metrics_train=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       metrics_val=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                                       learning_rate=0.001, 
                                       loss_fn=CategoricalCrossentropy(from_logits=False),
                                       buffer_shuffle=None, batch_size=batch_size,
                                       data_augmentation=True, augment_batch_factor=4)
del model_trainer
                                      
                                      
                                      
                                      


