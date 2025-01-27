# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:10:29 2025

@author: marce
"""

# Create a new X entropy directory with the original 
# X files and the aggregated entropy

# The model_output_dirs are the outputs directories
# resulting from the model training. The results stored
# in these directories will be used to build an ensemble 
# directory named ensemble_dir

# The ensemble_dir is used to store the ensemble files 
# and the entropy files

# The x_dir is the original X Directory and the y_dir
# is the Y Directory (the directory of the labels)

# The x_entropy_dir is the directory that will be created
# with files that concatenate the original X with the calculated
# entropy 


# %% Imports

from package_doc.entropy.entropy_calc import (EntropyCalculator,
                                              EntropyClasses)

# %% Directories to calculate entropy

model_output_dirs = ['saida_resunet_loop_2x_2b_0/',
                     'saida_resunet_loop_2x_2b_1/']

ensemble_dir = r'ensemble_dir/'

x_dir = 'teste_x1/'
y_dir = 'teste_y1/'

x_entropy_dir = 'x_entropy_resunet/'

entropy_classes = EntropyClasses.Road

# %% Create instance of EntropyCalculator and Ensemble Dir

entropy = EntropyCalculator(ensemble_dir=ensemble_dir, 
                            model_output_dirs=model_output_dirs)

# %% Calculate entropy and Create X entropy directory

entropy.create_x_entropy_dir(x_dir=x_dir, 
                             x_entropy_dir=x_entropy_dir,
                             entropy_classes=entropy_classes,
                             y_dir=y_dir)

