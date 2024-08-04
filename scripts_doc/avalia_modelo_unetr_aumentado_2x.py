# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:07:15 2023

@author: marcel.rotunno
"""


# %% Imports

import tensorflow as tf 
import os
import pdb

# %% Desabilita GPU

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% Limita cores usados (Limita CPU) 

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Imprime 0 gpus (usa cpu)

# %% Data for Script

batch_size = 16
model_type = 'unetr'
dist_buffers = [3]


# %% Diretórios de entrada e saída

x_dir = r'entrada/'
y_dir = r'y_directory/'
output_dir = fr'saida_{model_type}_loop_2x_{batch_size}b/'

# %%  Imports from package

from package_doc.avaliacao.avalia import avalia_modelo

# %% Roda avaliação

print('X Dir: ', x_dir)
print('Y Dir: ', y_dir)
print('Output Dir: ', output_dir)

#pdb.set_trace()
dicio_resultados = avalia_modelo(x_dir, y_dir, output_dir, metric_name='F1-Score',
                                 dist_buffers=dist_buffers, avalia_train=False, avalia_ate_teste=False)