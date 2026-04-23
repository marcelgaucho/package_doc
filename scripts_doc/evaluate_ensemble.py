# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:08:01 2025

@author: Marcel
"""

# %% Disable GPU and limit number of CPU threads (and thus the number of cpu cores) to use in evaluation

# from osgeo import gdal # First import gdal when it gives error
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use Keras 2
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Print 0 gpus (gpu disabled)

cpu_threads = 1

print(f"Number of CPU threads used in evaluation: {cpu_threads}") 

tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)

# %% Import Libraries

from package_doc.avaliacao.evaluator import ModelEvaluator

import re
from pathlib import Path

# %% Directories

x_dir = r'experimentos_deforestation/x_dir/'
y_dir = r'experimentos_deforestation/y_dir/'
outputs_dir = r'experimentos_deforestation/out_resunet/'
label_tiles_dir = 'tiles_t2_preprocessed/test/' # dir of test labels

# %%

x_dir = Path(x_dir)
y_dir = Path(y_dir)
outputs_dir = Path(outputs_dir)

# %% Parameters for evaluation

prefix = 'outmosaic'
export_pred_mosaics = False
export_prob_mosaics = False

buffers = [0]

include_avg_precision = False # Also evaluate average precision (in addition to precision, recall and F1)

evaluate_train=False
evaluate_valid=True
evaluate_test=True

# %% Model directories in base dir

model_dirs = [d for d in outputs_dir.iterdir() if re.match('m_\d', d.name) and d.is_dir()]

# %% Evaluate loop

for d in model_dirs:
    # Create evaluator instance
    evaluator = ModelEvaluator(x_dir=x_dir, y_dir=y_dir, output_dir=d,
                               label_tiles_dir=label_tiles_dir)
    
    # Evaluate model groups
    evaluator.evaluate_model(evaluate_train=evaluate_train, 
                             evaluate_valid=evaluate_valid, 
                             evaluate_test=evaluate_test,
                             buffers_px=buffers,
                             include_avg_precision=include_avg_precision)
    
    # Build mosaics
    evaluator.build_test_mosaics(prefix=prefix, export_pred_mosaics=export_pred_mosaics,
                                 export_prob_mosaics=export_prob_mosaics)
    
    # Evaluate mosaics    
    evaluator.evaluate_mosaics(buffers_px=buffers, include_avg_precision=include_avg_precision) 
    
    
