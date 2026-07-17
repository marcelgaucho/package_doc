# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:53:10 2025

@author: marce
"""

# Calculate uncertainty for selected metric

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

# %% Imports

from package_doc.entropy.utils import DataGroups, UncertaintyMetric 
from package_doc.entropy.ensemb import Ensemble
from package_doc.avaliacao.mosaics import MosaicGenerator
import numpy as np
import json
from pathlib import Path

import argparse

# %% Create parser

parser = argparse.ArgumentParser(description='Create uncertainty mosaic')

# %% Add arguments

# Model directories included in uncertainty
parser.add_argument('--model_dirs', nargs='+', type=str, help='Model output '
                    'directories used to calculate uncertainty')

# Uncertainty metric
parser.add_argument('--metrics', nargs='+', metavar='uncertainty_metric', type=str, help='Uncertainty metric to compute',
                    choices=['entropy', 'surprise', 'weightedsurprise', 'probmean'])

# Uncertainty output directory
parser.add_argument('-od', '--out_dir', type=str, help='Output directory created '
                    'to store uncertainty arrays and mosaics')

# Min, max and percentage cut used in data normalization
parser.add_argument('--min_scale', type=float, help='Minimum value of target normalization scale', default=0)
parser.add_argument('--max_scale', type=float, help='Maximum value of target normalization scale', default=1)
parser.add_argument('--perc_cut', type=float, help='Percentage cut used before data normalization', default=2)

# Previous models (used to the directory name)
parser.add_argument('--previous_models', nargs='+', type=str, help='Previous models used in training',
                    choices=['unet', 'resunet', 'unetr', 'segformer_b5'])

# Group list to compute uncertainty
parser.add_argument('-g', '--groups', metavar='group_list', type=str, 
                    nargs='+', default=['valid', 'test'], 
                    choices=['train', 'valid', 'test'],
                    help='Data group list to compute uncertainty')

# Build mosaics
parser.add_argument('--build_mosaics', action='store_true')

# Label tiles dir
parser.add_argument('-ld', '--label_dir', type=str, help='Label tile directory of test group')

# Y dir (used to find metadata of extraction information)
parser.add_argument('-yd', '--y_dir', type=str, help='Y directory')

# %% Parse args 

args = parser.parse_args()
# args = parser.parse_args(['--model_dirs', 'experimentos1/saidas1/resunet_0', 'experimentos1/saidas1/resunet_1',
#                           '--metrics', 'entropy',
#                           '-od', 'experimentos1/saidas1/uncertainty1',
#                           '--min_scale', '0',
#                           '--max_scale', '1',
#                           '--perc_cut', '2',
#                           '-m', 'resunet',
#                           '-g', 'train', 'valid', 'test',
#                           '--build_mosaics',
#                           '-ld', 'dataset_massachusetts_mnih_exp/test/maps',
#                           '-yd', 'experimentos1/y_dir'])

# Model directories
model_dirs = args.model_dirs

# Uncertainty metric
metrics = args.metrics

# Uncertainty output directory
out_dir = Path(args.out_dir)

# Min, max and percentage cut
min_target_scale = args.min_scale
max_target_scale = args.max_scale
perc_cut = args.perc_cut

# Previous models
previous_models = args.previous_models

# Groups
data_groups = args.groups

# Build mosaics
build_mosaics = args.build_mosaics

# Label tile directory of test group
label_dir = Path(args.label_dir)

# Y directory for metadata
y_dir = Path(args.y_dir)

# %% Create directory

out_dir.mkdir(exist_ok=True)


# %% Previous models string

previous_models_str = '_'.join(previous_models)

# %% Calculate uncertainty metrics 

# Iterate over the selected data groups among train, valid, test
for data_group in data_groups:
    # Entropy
    if UncertaintyMetric.Entropy in metrics:    
        ensemble = Ensemble(model_dirs=model_dirs, data_group=data_group)
        entropy = ensemble.entropy(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                   perc_cut=perc_cut)
        np.save(out_dir / f'{previous_models_str}_{UncertaintyMetric.Entropy}_{data_group}.npy', entropy)

    # Surprise
    if UncertaintyMetric.Surprise in metrics:    
        ensemble = Ensemble(model_dirs=model_dirs, data_group=data_group)
        surprise = ensemble.surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                   perc_cut=perc_cut)
        np.save(out_dir / f'{previous_models_str}_{UncertaintyMetric.Surprise}_{data_group}.npy', surprise)

    # Weighted Surprise
    if UncertaintyMetric.WeightedSurprise in metrics:    
        ensemble = Ensemble(model_dirs=model_dirs, data_group=data_group)
        weighted_surprise = ensemble.weighted_surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                   perc_cut=perc_cut)
        np.save(out_dir / f'{previous_models_str}_{UncertaintyMetric.WeightedSurprise}_{data_group}.npy', weighted_surprise)          

    # Probability Mean
    if UncertaintyMetric.ProbMean in metrics:    
        ensemble = Ensemble(model_dirs=model_dirs, data_group=data_group)
        prob_mean = ensemble.prob_mean(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                   perc_cut=perc_cut)
        np.save(out_dir / f'{previous_models_str}_{UncertaintyMetric.ProbMean}_{data_group}.npy', prob_mean) 
        
        
# %% Calculate test mosaics if posible

# Raise error if build mosaics is activated and 
# test group isn't in the groups list
if build_mosaics and not DataGroups.Test in data_groups:
    raise Exception(f'Unable to build mosaics. {DataGroups.Test} must be in the metrics arguments.')
        
# Test group
data_group = DataGroups.Test

# Load metadata information
with open(y_dir / 'info_tiles_test.json') as fp:   
    info_tiles_test = json.load(fp)

# Iterate over the metrics
for metric in metrics: 
    # Prefix for mosaics names
    prefix = f'mosaic_{metric}_'
    
    # Load Unceratinty array data
    uncertainty_array = np.load(out_dir / f'{previous_models_str}_{metric}_{data_group}.npy')
    
    # Load test metadata information
    with open(y_dir / 'info_tiles_test.json') as fp:   
        info_tiles_test = json.load(fp)
        
    # Build and export mosaics 
    mosaics = MosaicGenerator(test_array=uncertainty_array, 
                              info_tiles=info_tiles_test, 
                              tiles_dir=label_dir,
                              output_dir=out_dir)
    mosaics.build_mosaics()
    mosaics.export_mosaics(prefix=prefix)
    
    

