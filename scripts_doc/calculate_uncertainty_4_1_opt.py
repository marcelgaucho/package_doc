# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:25:09 2025

@author: Marcel
"""

# Calculate uncertainty and store it in ensemble dir

# %% Imports

from package_doc.entropy.ensemble import EnsembleDir, Ensemble, DataGroups

# %% Configuration data

save_result = True

model_dirs = ['experimentos/saida_resunet_loop_2x_2b_0/', 'experimentos/saida_resunet_loop_2x_2b_1/']

ensemble_path = r'experimentos/ensemble_dir_teste'

min_target_scale = 0
max_target_scale = 1
perc_cut = 2

# %% Calculate uncertainty metrics

ensembledir = EnsembleDir(ensemble_folder=ensemble_path, model_dirs=model_dirs)

ensemble_entropy = {}
ensemble_surprise = {}
ensemble_weighted_surprise = {}
ensemble_prob_mean = {}

for data_group in iter(DataGroups):
    ensemble = Ensemble(ensemble_dir=ensembledir, data_group=data_group)
    
    entropy = ensemble.entropy(min_target_scale=min_target_scale, max_target_scale=max_target_scale, perc_cut=perc_cut,
                               save_result=save_result)
    ensemble_entropy[data_group] = entropy
    
    surprise = ensemble.surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale, perc_cut=perc_cut,
                                 save_result=save_result)
    ensemble_surprise[data_group] = surprise
    
    weighted_surprise = ensemble.weighted_surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale, perc_cut=perc_cut,
                                                   save_result=save_result)
    ensemble_weighted_surprise[data_group] = weighted_surprise
    
    prob_mean = ensemble.prob_mean(min_target_scale=min_target_scale, max_target_scale=max_target_scale, perc_cut=perc_cut,
                                   save_result=save_result)
    ensemble_prob_mean[data_group] = prob_mean 


