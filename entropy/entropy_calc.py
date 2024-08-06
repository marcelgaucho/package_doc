# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:24:31 2024

@author: Marcel
"""

import numpy as np

from package_doc.avaliacao.compute_functions import adiciona_prob_0


class EntropyCalculator:
    def __init__(self, ensemble_dir):
        self.ensemble_dir = ensemble_dir
        
    def _set_prob_array_test(self, n_members):
        prob_test_array = [adiciona_prob_0(np.load(self.ensemble_dir+f'prob_test_{i}.npy')) for i in range(n_members)]
        self.prob_test_array = np.array(prob_test_array)
        
    def _set_prob_array_valid(self, n_members):
        prob_valid_array = [adiciona_prob_0(np.load(self.ensemble_dir+f'prob_valid_{i}.npy')) for i in range(n_members)]
        self.prob_valid_array = np.array(prob_valid_array)
        
    def predictive_entropy_test(self, n_members, save_result=False):
        # Set probabilities arrays
        self._set_prob_array_test(n_members)
        
        # Mean probabilities for each class    
        prob_test_mean = np.mean(self.prob_test_array, axis=0)
        
        # Initialize result with zeros
        test_entropy = np.zeros(prob_test_mean.shape[0:-1] + (1,), dtype=np.float16) # (B, H, W, 1)
        
        K = prob_test_mean.shape[-1] # Number of classes
        epsilon = 1e-7 # Used to manage log 0
        
        # Calculate with log base and scale to [0, 1]
        for k in range(K):
            test_entropy = test_entropy + prob_test_mean[..., k:k+1] * np.log(prob_test_mean[..., k:k+1] + epsilon)
            
        test_entropy = - test_entropy / np.log(K) # scale to [0, 1], because entropy maximum is log(K)
                                                  
        test_entropy = np.clip(test_entropy, 0, 1) # Clip values to [0 e 1]
        
        if save_result: np.save(self.ensemble_dir + 'entropy_test.npy', test_entropy) # save array in disk
        
        return test_entropy
    
    def predictive_entropy_valid(self, n_members, save_result=False):
        # Set probabilities arrays
        self._set_prob_array_valid(n_members)
        
        # Mean probabilities for each class    
        prob_valid_mean = np.mean(self.prob_test_array, axis=0)
        
        # Initialize result with zeros
        valid_entropy = np.zeros(prob_valid_mean.shape[0:-1] + (1,), dtype=np.float16) # (B, H, W, 1)
        
        K = prob_valid_mean.shape[-1] # Number of classes
        epsilon = 1e-7 # Used to manage log 0
        
        # Calculate with log base and scale to [0, 1]
        for k in range(K):
            valid_entropy = valid_entropy + prob_valid_mean[..., k:k+1] * np.log(prob_valid_mean[..., k:k+1] + epsilon)
            
        valid_entropy = - valid_entropy / np.log(K) 
                                                  
        valid_entropy = np.clip(valid_entropy, 0, 1) 
        
        if save_result: np.save(self.ensemble_dir + 'entropy_valid.npy', valid_entropy)
        
        return valid_entropy