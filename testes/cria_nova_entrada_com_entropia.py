# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:43:44 2024

@author: Marcel
"""

import numpy as np
from pathlib import Path

# %% Directories

x_dir = 'teste_x/'
ensemble_dir = 'ensemble_dir/'
x_entropy_dir = 'teste_x_entropy/'

# %%

def create_x_entropy_dir(input_x_dir, ensemble_dir, x_entropy_dir):
    # Load X Data
    x_train = np.load(input_x_dir + 'x_train.npy')
    x_valid = np.load(input_x_dir + 'x_valid.npy')
    x_test = np.load(input_x_dir + 'x_test.npy')
    
    # Load Entropy Data
    entropy_train = np.load(ensemble_dir + 'entropy_train.npy')
    entropy_valid = np.load(ensemble_dir + 'entropy_valid.npy')
    entropy_test = np.load(ensemble_dir + 'entropy_test.npy')
    
    # Concatenate x and entropy
    x_entropy_train = np.concatenate((x_train, entropy_train), axis=-1)
    x_entropy_valid = np.concatenate((x_valid, entropy_valid), axis=-1)
    x_entropy_test = np.concatenate((x_test, entropy_test), axis=-1)
    
    # Create dir and save numpy arrays of entropies to dir
    if not Path(x_entropy_dir).exist():
        Path(x_entropy_dir).mkdir(exist_ok=True)
        
    np.save(x_entropy_dir + 'x_train.npy', x_entropy_train)
    np.save(x_entropy_dir + 'x_valid.npy', x_entropy_valid)
    np.save(x_entropy_dir + 'x_test.npy', x_entropy_test)
    
