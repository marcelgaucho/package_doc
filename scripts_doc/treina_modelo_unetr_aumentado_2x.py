# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

# %% Imports

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from pathlib import Path
import pdb

# %% Limita Memória usada pela GPU ou limita threads caso não haja GPU

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
model_type = 'unetr'
early_stopping_epochs = 25

# %% Diretórios de entrada e saída

# x_dir = r'entrada/'
# y_dir = r'y_directory/'
x_dir = r'teste_x/'
y_dir = r'teste_y/'
output_dir = fr'saida_{model_type}_loop_2x_{batch_size}b/'


# %%  Imports from package

from package_doc.treinamento.functions_train import ModelTrainer
from package_doc.treinamento.f1_metric import F1Score, RelaxedF1Score
from package_doc.treinamento.arquiteturas.modelos import build_model
from package_doc.treinamento.arquiteturas.unetr_2d import config_dict


# %% Constroi modelo ResUnet

input_shape = (256, 256, 3)
n_classes = 2
model = build_model(input_shape, n_classes, model_type=model_type, config_dict=config_dict)

# %% Constroi otimizador

#pdb.set_trace()
optimizer = Adam() 

# %% Treina Modelo com Loop 

if not Path(output_dir).exists():
    Path(output_dir).mkdir(exist_ok=True)
    
    
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

                  
                                      
                                      
                                      


