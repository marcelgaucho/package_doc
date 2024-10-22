# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:24:31 2024

@author: Marcel
"""

import tensorflow as tf

import numpy as np
from pathlib import Path
import shutil

from package_doc.avaliacao.compute_functions import adiciona_prob_0
from package_doc.functions_extract import onehot_numpy



class EntropyCalculator:
    def __init__(self, ensemble_dir):
        self.ensemble_dir = ensemble_dir
        
    def create_ensemble_dir(self, output_dirs):
        # Create ensemble dir
        if not Path(self.ensemble_dir).exists():
            Path(self.ensemble_dir).mkdir(exist_ok=True)
        
        # Copy and enumerate prob arrays in ascending order
        for i, folder in enumerate(sorted(output_dirs)):
            # Copy Prob Train if not exists
            if not (Path(self.ensemble_dir) / 'prob_train_{i}.npy').exists():
                shutil.copy(folder + 'prob_train.npy', self.ensemble_dir + f'prob_train_{i}.npy')
            
            # Copy Prob Valid if not exists
            if not (Path(self.ensemble_dir) / 'prob_valid_{i}.npy').exists():
                shutil.copy(folder + 'prob_valid.npy', self.ensemble_dir + f'prob_valid_{i}.npy')
            
            # Copy Prob Test if not exists
            if not (Path(self.ensemble_dir) / 'prob_test_{i}.npy').exists():
                shutil.copy(folder + 'prob_test.npy', self.ensemble_dir + f'prob_test_{i}.npy')
            
        return True
    
    def create_x_entropy_dir(self, input_x_dir, x_entropy_dir, entropy_classes='all', y_dir=None):
        # Load X Data
        x_train = np.load(input_x_dir + 'x_train.npy')
        x_valid = np.load(input_x_dir + 'x_valid.npy')
        x_test = np.load(input_x_dir + 'x_test.npy')
        
        # Load Entropy Data
        assert entropy_classes in ('all', 'road', 'back', 'road and back'), "Parameter entropy_class must be 'all', 'road'. 'back' or " \
                                                                            "'road and back'"
                                                        
        if entropy_classes in ('all', 'road', 'back'):        
            entropy_train = np.load(self.ensemble_dir + f'entropy_train_{entropy_classes}.npy')
            entropy_valid = np.load(self.ensemble_dir + f'entropy_valid_{entropy_classes}.npy')
            entropy_test = np.load(self.ensemble_dir + f'entropy_test_{entropy_classes}.npy')
        elif entropy_classes == 'road and back':
            entropy_train_road = np.load(self.ensemble_dir + 'entropy_train_road.npy')
            entropy_valid_road = np.load(self.ensemble_dir + 'entropy_valid_road.npy')
            entropy_test_road = np.load(self.ensemble_dir + 'entropy_test_road.npy')
            entropy_train_back = np.load(self.ensemble_dir + 'entropy_train_back.npy')
            entropy_valid_back = np.load(self.ensemble_dir + 'entropy_valid_back.npy')
            entropy_test_back = np.load(self.ensemble_dir + 'entropy_test_back.npy')
            entropy_train = np.concatenate((entropy_train_road, entropy_train_back), axis=-1) 
            entropy_valid = np.concatenate((entropy_valid_road, entropy_valid_back), axis=-1) 
            entropy_test =  np.concatenate((entropy_test_road, entropy_test_back), axis=-1)
            del entropy_train_road, entropy_valid_road, entropy_test_road
            del entropy_train_back, entropy_valid_back, entropy_test_back
            
        
        # Concatenate x and entropy
        x_entropy_train = np.concatenate((x_train, entropy_train), axis=-1)
        x_entropy_valid = np.concatenate((x_valid, entropy_valid), axis=-1)
        x_entropy_test = np.concatenate((x_test, entropy_test), axis=-1)
        
        # Create dir and save numpy arrays of entropies to dir
        if not Path(x_entropy_dir).exists():
            Path(x_entropy_dir).mkdir(exist_ok=True)
            
        if not (Path(x_entropy_dir) / 'x_train.npy').exists(): np.save(x_entropy_dir + 'x_train.npy', x_entropy_train)
        if not (Path(x_entropy_dir) / 'x_valid.npy').exists(): np.save(x_entropy_dir + 'x_valid.npy', x_entropy_valid)
        if not (Path(x_entropy_dir) / 'x_test.npy').exists(): np.save(x_entropy_dir + 'x_test.npy', x_entropy_test)
        
        # Create train and valid datasets if y_dir is given
        if y_dir:
            y_train = np.load(y_dir + 'y_train.npy')
            y_valid = np.load(y_dir + 'y_valid.npy')      
            
            y_train = onehot_numpy(y_train)
            y_valid = onehot_numpy(y_valid)
            
            train_dataset = tf.data.Dataset.from_tensor_slices((x_entropy_train, y_train))
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_entropy_valid, y_valid))
            
            train_dataset_path = Path(x_entropy_dir) / 'train_dataset'
            train_dataset_path.mkdir()

            valid_dataset_path = Path(x_entropy_dir) / 'valid_dataset'
            valid_dataset_path.mkdir()
            
            train_dataset.save(str(train_dataset_path))
            valid_dataset.save(str(valid_dataset_path))
            
        return True
        
    def _calc_prob_array(self, n_members, group):
        prob_array = [adiciona_prob_0(np.load(self.ensemble_dir+f'prob_{group}_{i}.npy')) for i in range(n_members)]
        prob_array = np.array(prob_array)
        
        return prob_array        
        
    def predictive_entropy(self, n_members, group='test', entropy_classes='all', save_result=False):
        # Check if group is correct
        assert group in ('train', 'valid', 'test'), "Parameter group must be 'train', 'valid' or test"

        # Set probabilities arrays
        prob_array = self._calc_prob_array(n_members, group=group)
        
        # Mean probabilities for each class    
        prob_mean = np.mean(prob_array, axis=0)
        
        # Initialize result with zeros
        entropy = np.zeros(prob_mean.shape[0:-1] + (1,), dtype=np.float16) # (B, H, W, 1)
        
        K = prob_mean.shape[-1] # Number of classes
        epsilon = 1e-7 # Used to manage log 0
        
        # Calculate entropy depending on the class or classes
        assert entropy_classes in ('all', 'road', 'back'), "Parameter entropy_class must be 'all', 'road' or 'back'"
        
        if entropy_classes == 'all':
            # Calculate with log base and scale to [0, 1]
            for k in range(K):
                entropy = entropy + prob_mean[..., k:k+1] * np.log(prob_mean[..., k:k+1] + epsilon)
                
            entropy = - entropy / np.log(K) # scale to [0, 1], because entropy maximum is log(K)
                                                      
            entropy = np.clip(entropy, 0, 1) # Clip values to [0 e 1]
        elif entropy_classes == 'road':
            # Road Class is 1
            entropy = entropy + prob_mean[..., 1:1+1] * np.log(prob_mean[..., 1:1+1] + epsilon)
            
            entropy = - entropy / np.log(K) # scale to [0, 1], because entropy maximum is log(K)
            
            entropy = np.clip(entropy, 0, 1) # Clip values to [0 e 1]
        elif entropy_classes == 'back':
            # Road Class is 0
            entropy = entropy + prob_mean[..., 0:0+1] * np.log(prob_mean[..., 0:0+1] + epsilon)
            
            entropy = - entropy / np.log(K) # scale to [0, 1], because entropy maximum is log(K)
            
            entropy = np.clip(entropy, 0, 1) # Clip values to [0 e 1]       
        
        if save_result: np.save(self.ensemble_dir + f'entropy_{group}_{entropy_classes}.npy', entropy) # save array in disk
            
            
            
        
        return entropy