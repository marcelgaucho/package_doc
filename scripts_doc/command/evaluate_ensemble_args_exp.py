# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:08:01 2025

@author: Marcel
"""

# %% Disable GPU and limit number of CPU threads (and thus the number of cpu cores) to use in evaluation

# from osgeo import gdal # First import gdal when it gives error
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Print 0 gpus (gpu disabled)

cpu_threads = 1

print(f"Number of CPU threads used in evaluation: {cpu_threads}") 

tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)

# %% Import Libraries

from package_doc.avaliacao.evaluator import ModelEvaluator

import argparse
import re
from pathlib import Path

# %% Create parser

parser = argparse.ArgumentParser(description='Evaluate an ensemble of models')

# %% Add arugments

# Input dirs
parser.add_argument('x_dir', type=str, help='Enter the X directory for training')
parser.add_argument('y_dir', type=str, help='Enter the Y directory for training')

# Output base dir
parser.add_argument('-od', '--out_dir', type=str, help='Enter the output directory for the trained models',
                    required=True)

# Label tiles dir
parser.add_argument('-ld', '--label_dir', type=str, help='Label tile directory of test group')

# Pixel buffer values list
parser.add_argument('-bf', '--buffers', metavar='buffer_px_list', type=int, nargs='+',
                    default=[3], 
                    help='List of buffer values, in pixels, used in relaxed metrics')

# Group list to evaluate
parser.add_argument('-g', '--groups', metavar='group_list', type=str, 
                    nargs='+', default=['valid', 'test'], 
                    choices=['train', 'valid', 'test'],
                    help='Data group list to evaluate')

# Average precision
parser.add_argument('-ap', '--avg_precision', action='store_true',
                    help='Include average precision in evaluation')

# %% Parse args

args = parser.parse_args()
# args = parser.parse_args(['experimentos1/x_dir',
#                           'experimentos1/y_dir',
#                           '-od', 'experimentos1/out/resunet',
#                           '-ld', 'dataset_massachusetts_mnih_exp/test/maps',
#                           '-bf', '3',
#                           '-g', 'valid', 'test',
#                           '-ap'])

# X and Y directories
x_dir = args.x_dir
y_dir = args.y_dir

# Output directory
output_dir = Path(args.out_dir)

# Label tile directory of test group
label_dir = args.label_dir

# Buffers
buffers = args.buffers

# Groups
groups = args.groups

# Measure average precision
include_avg_precision = args.avg_precision

# %%  Groups to evaluate (True or False) and prefix to rasters exported

evaluate_train = True if 'train' in groups else False
evaluate_valid = True if 'valid' in groups else False
evaluate_test = True if 'test' in groups else False

prefix = 'outmosaic'

# %% Model directories in base dir

model_dirs = [d for d in output_dir.iterdir() if re.match('m_\d', d.name) and d.is_dir()]

# %% Evaluate loop

for d in model_dirs:
    # Create evaluator instance
    evaluator = ModelEvaluator(x_dir=x_dir, y_dir=y_dir, output_dir=d,
                               label_tiles_dir=label_dir)
    
    # Evaluate model groups
    evaluator.evaluate_model(evaluate_train=evaluate_train, 
                             evaluate_valid=evaluate_valid, 
                             evaluate_test=evaluate_test,
                             buffers_px=buffers,
                             include_avg_precision=include_avg_precision)
    
    # Build mosaics
    evaluator.build_test_mosaics(prefix=prefix)
    
    # Evaluate mosaics    
    evaluator.evaluate_mosaics(buffers_px=buffers, include_avg_precision=include_avg_precision) 
    
    
