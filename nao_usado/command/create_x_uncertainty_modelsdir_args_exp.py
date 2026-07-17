# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:31:02 2025

@author: Marcel
"""

# Create a X directory with original data concatenated with uncertainty for a folder with 
# multiple model directories

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

import argparse
from pathlib import Path
import re

from package_doc.entropy.ensemb import Ensemble
from package_doc.entropy.dir_uncertain import XDirUncertain 



# %% Create parser

parser = argparse.ArgumentParser(description='Create X directory with uncertainty for '
                                             'a directory with multiple model folders')

# %% Add arguments

# Directory with model folders
parser.add_argument('--models_dir', type=str, help='Directory with the '
                    'model folders used to calculate uncertainty')

# Uncertainty metric to concatenate
parser.add_argument('--metric', metavar='uncertainty_metric', type=str, help='Uncertainty metric concatenated',
                    choices=['entropy', 'surprise', 'weightedsurprise', 'probmean'])

# Min, max and percentage cut used in data normalization
parser.add_argument('--min_scale', type=float, help='Minimum value of target normalization scale', default=0)
parser.add_argument('--max_scale', type=float, help='Maximum value of target normalization scale', default=1)
parser.add_argument('--perc_cut', type=float, help='Percentage cut used before data normalization', default=2)

# X original directory
parser.add_argument('--x_dir', type=str, help='Original X directory')

# Y dir (used to build tensorflow datasets)
parser.add_argument('-yd', '--y_dir', type=str, help='Y directory')

# %% Parse args

# args = parser.parse_args()
args = parser.parse_args(['--models_dir', 'experimentos1/out/resunet',
                          '--metric', 'entropy',
                          '--min_scale', '0',
                          '--max_scale', '1',
                          '--perc_cut', '2',
                          '--x_dir', 'experimentos1/x_dir',
                          '-yd', 'experimentos1/y_dir'])

# Models directory
models_dir = Path(args.models_dir)

# Uncertainty metric
metric = args.metric

# Min, max and percentage cut
min_target_scale = args.min_scale
max_target_scale = args.max_scale
perc_cut = args.perc_cut

# X original directory
x_dir = Path(args.x_dir)

# Y dir
y_dir = Path(args.y_dir)

# %% Output models directories and output

model_dirs = [d for d in models_dir.iterdir() if re.match('m_\d', d.name) and d.is_dir()]

out_x_folder = models_dir / 'x_dir'

# %% Create directory

ensemble = Ensemble(model_dirs=model_dirs)

x_dir_uncer = XDirUncertain(in_x_folder=x_dir, y_folder=y_dir, 
                            out_x_folder=out_x_folder, 
                            model_dirs=model_dirs,
                            metric=metric, 
                            min_scale_uncertainty=min_target_scale, 
                            max_scale_uncertainty=max_target_scale,
                            perc_cut=perc_cut)
x_dir_uncer.create()

# %% Insert data

x_dir_uncer.insert_data() 

