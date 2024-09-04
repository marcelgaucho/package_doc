# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:48:42 2024

@author: marcel.rotunno
"""
# %% Import Libraries

import numpy as np
from package_doc.entropy.entropy_calc import EntropyCalculator
from pathlib import Path
import shutil


# %% Directories   

output_dirs = ['saida_resunet_loop_2x_16b_0/',
               'saida_resunet_loop_2x_16b_1/',
               'saida_resunet_loop_2x_16b_2/',
               'saida_resunet_loop_2x_16b_3/',
               'saida_resunet_loop_2x_16b_4/']
# ensemble_dir = 'ensemble_saida_resunet_loop_2x_16b/'
ensemble_dir = r'ensemble_dir/'



    
# %% Calculate entropy

entropy_calc = EntropyCalculator(ensemble_dir=ensemble_dir)
test_entropy_all = entropy_calc.predictive_entropy(n_members=5, group='test', save_result=False)