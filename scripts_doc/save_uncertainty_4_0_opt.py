# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:25:09 2025

@author: Marcel
"""

# Calculate uncertainty and store it in ensemble dir

# %% Imports

from package_doc.entropy.ensemb import Ensemble, DataGroups
from package_doc.entropy.dir_uncertain import UncertaintyMetric 
from pathlib import Path
import numpy as np

# %% Disable GPU and limit number of CPU threads (and thus the number of cpu cores) to use in evaluation

# from osgeo import gdal # First import gdal when it gives error
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Print 0 gpus (gpu disabled)

cpu_threads = 3

print(f"Number of CPU threads used in evaluation: {cpu_threads}") 

tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)


# %% Configuration data

model_dirs = ['experimentos1/saidas1/resunet_0/',
              'experimentos1/saidas1/resunet_1/']

folder_uncertainty = Path(r'experimentos1/saidas1/uncertainty/')

min_target_scale = 0
max_target_scale = 1
perc_cut = 2

data_group = DataGroups.Test

model_name = 'resunet'

# %% Create directory

folder_uncertainty.mkdir(exist_ok=True)

# %% Calculate uncertainty metrics - Create ensemble

ensemble = Ensemble(model_dirs=model_dirs, data_group=data_group)

# %% Entropy

entropy = ensemble.entropy(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                           perc_cut=perc_cut)
np.save(folder_uncertainty / f'{model_name}_{UncertaintyMetric.Entropy}_{data_group}.npy', entropy)

# %% Surprise

surprise = ensemble.surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                             perc_cut=perc_cut)
np.save(folder_uncertainty / f'{model_name}_{UncertaintyMetric.Surprise}_{data_group}.npy', surprise)

# %% Weighted Surprise

weighted_surprise = ensemble.weighted_surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                               perc_cut=perc_cut)
np.save(folder_uncertainty / f'{model_name}_{UncertaintyMetric.WeightedSurprise}_{data_group}.npy', weighted_surprise)

# %% Probability mean
    
prob_mean = ensemble.prob_mean(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                               perc_cut=perc_cut)
np.save(folder_uncertainty / f'{model_name}_{UncertaintyMetric.ProbMean}_{data_group}.npy', prob_mean)

    	


