# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:08:01 2025

@author: Marcel
"""

# Create new X directory that contains X data with uncertainty

# %% Imports

from package_doc.entropy.ensemble import EnsembleDir
from package_doc.entropy.dir_uncertain import XDirUncertain, UncertaintyMetric

# %% Information to calculate entropy

metric = UncertaintyMetric.Entropy
previous_model = 'resunet'
min_scale_uncertainty = 0
max_scale_uncertainty = 1

model_dirs = ['experimentos/saida_resunet_loop_2x_2b_0/', 'experimentos/saida_resunet_loop_2x_2b_1/']

ensemble_path = r'experimentos/ensemble_dir_teste/'

in_x_folder = 'experimentos/x_dir'
y_folder = 'experimentos/y_dir'
 
out_x_folder = f'experimentos/x_{metric}_{previous_model}_dir'

# %% Create directory

ensembledir = EnsembleDir(ensemble_folder=ensemble_path, model_dirs=model_dirs)

x_dir_uncer = XDirUncertain(in_x_folder=in_x_folder, y_folder=y_folder, 
                            out_x_folder=out_x_folder, 
                            ensemble_dir=ensembledir,
                            metric=metric, 
                            min_scale_uncertainty=min_scale_uncertainty, 
                            max_scale_uncertainty=max_scale_uncertainty)
x_dir_uncer.create()

# %% Insert data

x_dir_uncer.insert_data() 