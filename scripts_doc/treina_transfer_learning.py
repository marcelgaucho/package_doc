# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

import numpy as np
import tensorflow as tf


tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Limita memória da GPU 
'''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

# Limita Memória usada pela GPU 
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=12288)])
'''


# %% Diretórios de entrada e saída

input_dir = 'entrada_pequena/'
y_dir = 'y_directory_pequeno/'
output_dir = 'saida_pequena_teste/'
model_checkpoint = "nvidia/mit-b5"



# %%  Importa função

from package_doc import transfer_learning_segformer

 
# %% Teste da Função

breakpoint()
#tf.config.run_functions_eagerly(False)
historia = transfer_learning_segformer(input_dir, y_dir, output_dir, model_checkpoint, 'modelo_fine.weights.h5', 
                                       learning_rate=0.001, epochs=2000, early_stopping_epochs=50, batch_size=16)

