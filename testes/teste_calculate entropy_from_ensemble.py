# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:30:00 2024

@author: Marcel
"""

import numpy as np
import shutil
from pathlib import Path
from package_doc.entropy.entropy_calc import EntropyCalculator

# %% Directories

output_dirs = ['saida_resunet_loop_2x_16b_0/',
               'saida_resunet_loop_2x_16b_1/',
               'saida_resunet_loop_2x_16b_2/',
               'saida_resunet_loop_2x_16b_3/',
               'saida_resunet_loop_2x_16b_4/']
ensemble_dir = 'ensemble_saida_resunet_loop_2x_16b/'


# %% Use function to create dir

calc_entropy = EntropyCalculator(ensemble_dir=ensemble_dir)
calc_entropy.create_ensemble_dir(output_dirs=output_dirs)
calc_entropy.predictive_entropy(n_members=5, group='train', entropy_classes='road', save_result=True)
calc_entropy.predictive_entropy(n_members=5, group='valid', entropy_classes='road', save_result=True)
calc_entropy.predictive_entropy(n_members=5, group='test', entropy_classes='road', save_result=True)
calc_entropy.create_x_entropy_dir(input_x_dir='entrada/', x_entropy_dir='entrada_entropy/', entropy_classes='road and back', y_dir='y_directory/')