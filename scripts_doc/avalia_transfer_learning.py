# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:07:15 2023

@author: marcel.rotunno
"""

from osgeo import gdal

import tensorflow as tf 
import os
import pdb

# Desabilita GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%

from package_doc.functions_bib import avalia_transfer_learning_segformer

# %% Teste da função de avaliação

input_dir = r'entrada/'
y_dir = r'y_directory/'
output_dir = r'saida_transfer/'
model_checkpoint = "nvidia/mit-b5"

print('Input Dir: ', input_dir)
print('Y Dir: ', y_dir)
print('Output Dir: ', output_dir)
print('Model Checkpoint: ', model_checkpoint)

#pdb.set_trace()
dicio_resultados = avalia_transfer_learning_segformer(input_dir, y_dir, output_dir, model_checkpoint, 'modelo_fine.weights.h5',
                                                      metric_name='F1-Score', dist_buffers=[0, 3], 
                                                      avalia_train=False, avalia_ate_teste=True, avalia_diff=False)