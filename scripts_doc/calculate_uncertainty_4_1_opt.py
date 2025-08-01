# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:25:09 2025

@author: Marcel
"""

# Calculate uncertainty and store it in ensemble dir

# %% Imports

from package_doc.entropy.ensemble import EnsembleDir, Ensemble, DataGroups

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

save_result = True

model_dirs = ['experimentos_1arede/saida_resunet_loop_2x_16b_0/',
              'experimentos_1arede/saida_resunet_loop_2x_16b_1/',
              'experimentos_1arede/saida_resunet_loop_2x_16b_2/',
              'experimentos_1arede/saida_resunet_loop_2x_16b_3/',
              'experimentos_1arede/saida_resunet_loop_2x_16b_4/']


ensemble_path = r'experimentos_1arede/ensemble_resunet/'

min_target_scale = 0
max_target_scale = 1
perc_cut = 2

data_group = DataGroups.Test

# %% Calculate uncertainty metrics

ensembledir = EnsembleDir(ensemble_folder=ensemble_path, model_dirs=model_dirs)

ensemble_entropy = {}
ensemble_surprise = {}
ensemble_weighted_surprise = {}
ensemble_prob_mean = {}

import pdb; pdb.set_trace()

ensemble = Ensemble(ensemble_dir=ensembledir, data_group=data_group)
    
entropy = ensemble.entropy(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                           perc_cut=perc_cut, save_result=save_result)
ensemble_entropy[data_group] = entropy

surprise = ensemble.surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                             perc_cut=perc_cut, save_result=save_result)
ensemble_surprise[data_group] = surprise

weighted_surprise = ensemble.weighted_surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                               perc_cut=perc_cut, save_result=save_result)
ensemble_weighted_surprise[data_group] = weighted_surprise
    
prob_mean = ensemble.prob_mean(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                               perc_cut=perc_cut, save_result=save_result)
ensemble_prob_mean[data_group] = prob_mean 
    	


