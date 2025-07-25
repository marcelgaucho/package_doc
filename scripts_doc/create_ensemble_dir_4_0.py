# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 21:19:28 2025

@author: Marcel
"""

# Create ensemble directory with probabilities arrays within the specified directories
# The ensemble directory can be used to store the uncertainty calculated, 
# but this is optional

# %% Imports

from package_doc.entropy.ensemble import EnsembleDir

# %% Directories to calculate entropy

model_dirs = ['experimentos/saida_resunet_loop_2x_2b_0/', 'experimentos/saida_resunet_loop_2x_2b_1/']

ensemble_path = r'experimentos/ensemble_dir_teste/'

# %% Create ensemble directory

ensembledir = EnsembleDir(ensemble_folder=ensemble_path, model_dirs=model_dirs)

ensembledir.create()

# %% Paste probability arrays

ensembledir.paste_data()

