# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:08:01 2025

@author: Marcel
"""

# Create new X directory that contains X data with uncertainty

# %% Imports

from package_doc.entropy.ensemb import Ensemble
from package_doc.entropy.dir_uncertain import XDirUncertain 
from package_doc.entropy.utils import UncertaintyMetric

# %% Information to calculate entropy

metric = UncertaintyMetric.Entropy
previous_models = ['resunet']
min_scale_uncertainty = 0
max_scale_uncertainty = 1
perc_cut = 2

model_dirs = ['experimentos1/saidas1/resunet_0/', 'experimentos1/saidas1/resunet_1/']

in_x_folder = 'experimentos1/x_dir'
y_folder = 'experimentos1/y_dir'

previous_models_str = '_'.join(previous_models)
out_x_folder = f'experimentos1/x_{metric}_{previous_models_str}_dir'

# %% Create directory

ensemble = Ensemble(model_dirs=model_dirs)

x_dir_uncer = XDirUncertain(in_x_folder=in_x_folder, y_folder=y_folder, 
                            out_x_folder=out_x_folder, 
                            model_dirs=model_dirs,
                            metric=metric, 
                            min_scale_uncertainty=min_scale_uncertainty, 
                            max_scale_uncertainty=max_scale_uncertainty,
                            perc_cut=perc_cut)
x_dir_uncer.create()

# %% Insert data

x_dir_uncer.insert_data() 