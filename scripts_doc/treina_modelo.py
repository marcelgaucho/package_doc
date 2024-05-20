# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

import numpy as np
import tensorflow as tf
import pdb


# Limita memória da GPU 
'''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

# Limita Memória usada pela GPU 
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=20480)])
'''


# %% Diretórios de entrada e saída

input_dir = 'entrada_pequena/'
y_dir = 'y_directory_pequeno/'
output_dir = 'saida_pequena_teste/'


# %%  Importa função
from package_doc.treinamento.functions_train import treina_modelo

 
# %% Teste da Função

#tf.config.run_functions_eagerly(False)
# pdb.set_trace()
treina_modelo(input_dir, y_dir, output_dir, epochs=2000, early_loss=False, 
              model_type='resunet', loss='cross', lr_decay=False, train_with_dataset=True, data_augmentation=True)
