# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:17:46 2024

@author: marcel.rotunno
"""

# Evaluate a trained model in the output directory

# %% Import Libraries

from package_doc.avaliacao.evaluator import ModelEvaluator


# %% Directories

x_dir = 'teste_x1/'
y_dir = 'teste_y1/'
output_dir = 'saida_resunet_loop_2x_2b/'
label_tiles_dir = 'dataset_massachusetts_mnih_exp/test/maps/' # dir of test labels

# %% Parameters for evaluation

prefix = 'outmosaic'

buffers_px = [3]
include_avg_precision = True # Also evaluate average precision (in addition to precision, recall and F1)

evaluate_train=False
evaluate_valid=True
evaluate_test=True

# %% Evalute model groups (valid and test are default)

evaluator = ModelEvaluator(x_dir=x_dir, y_dir=y_dir, output_dir=output_dir,
                           label_tiles_dir=label_tiles_dir)
evaluator.evaluate_model(evaluate_train=evaluate_train, 
                         evaluate_valid=evaluate_valid, 
                         evaluate_test=evaluate_test,
                         buffers_px=buffers_px,
                         include_avg_precision=include_avg_precision)

# %% Build mosaics

evaluator.build_test_mosaics(prefix=prefix)

# %% Evaluate mosaics

evaluator.evaluate_mosaics(buffers_px=buffers_px, include_avg_precision=include_avg_precision) 

