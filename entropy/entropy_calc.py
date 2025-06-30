# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:24:31 2024

@author: Marcel
"""

# Class that builds an ensemble directory from a list
# of directories of model outputs
# The class can be used also to calculate entropy for 
# a specific class or other options and 
# can be used to build a new X directory with
# aggregated entropy to the original X files

# %% Imports

import tensorflow as tf

import numpy as np
from pathlib import Path
import shutil

from package_doc.avaliacao.utils import adiciona_prob_0
from package_doc.treinamento.utils import onehot_numpy

# %% Class Enumerators

from enum import Enum

class EntropyClasses(str, Enum):
    All = 'all'
    Road = 'road'
    Back = 'back'
    Road_and_Back = 'road_and_back'
    
class EntropyGroups(str, Enum):
    Train = 'train'
    Valid = 'valid'
    Test = 'test'   
    
# %% Entropy Class

class EntropyCalculator:
    def __init__(self, ensemble_dir: str, 
                 model_output_dirs: list[str] = None) -> None:
        self.ensemble_dir = Path(ensemble_dir)
        
        if model_output_dirs is None or len(model_output_dirs) == 0:
            raise Exception('Model Output dirs must contain at least one directory')
            
        self.model_output_dirs = model_output_dirs
            
        # Create Ensemble Dir with all data
        self._create_ensemble_dir()
        
    def _create_ensemble_dir(self) -> bool:
        # Create ensemble dir
        print("Creating ensemble directory ...")
        if not self.ensemble_dir.exists():
            self.ensemble_dir.mkdir(exist_ok=True)
        else:
            print('Ensemble directory already exists. '
                  'So it will not be modified.')
            return
        
        # Copy and enumerate prob arrays in ascending order
        for i, folder in enumerate(sorted(self.model_output_dirs)):
            # Copy Prob Train 
            print(f"Copying prob_train of folder {i} in list")
            shutil.copy2(Path(folder) / 'prob_train.npy', Path(self.ensemble_dir) / f'prob_train_{i}.npy')
            
            # Copy Prob Valid 
            print(f"Copying prob_valid of folder {i} in list")
            shutil.copy2(Path(folder) / 'prob_valid.npy', Path(self.ensemble_dir) / f'prob_valid_{i}.npy')
            
            # Copy Prob Test 
            print(f"Copying prob_test of folder {i} in list")
            shutil.copy2(Path(folder) / 'prob_test.npy', Path(self.ensemble_dir) / f'prob_test_{i}.npy')
            
        return True
    
    def entropy(self, group: str = EntropyGroups.Test, 
                entropy_classes: str = EntropyClasses.All) -> np.ndarray:
        # Calculate probability array
        prob_array = [adiciona_prob_0(np.load(self.ensemble_dir / f'prob_{group}_{i}.npy')) 
                      for i in range(len(self.model_output_dirs))]
        prob_array = np.array(prob_array)
        
        # Mean probabilities for each class    
        prob_mean = np.mean(prob_array, axis=0)
        
        # Initialize result with zeros
        entropy = np.zeros(prob_mean.shape[0:-1] + (1,), dtype=np.float16) # (B, H, W, 1)
        
        K = prob_mean.shape[-1] # Number of classes
        epsilon = 1e-7 # Used to manage log 0        
        
        # Calculate entropy depending on the class or classes
        if entropy_classes == EntropyClasses.All:
            # Calculate with log base and scale to [0, 1]
            for k in range(K):
                entropy = entropy + prob_mean[..., k:k+1] * np.log(prob_mean[..., k:k+1] + epsilon)
                
            entropy = - entropy / np.log(K) # scale to [0, 1], because entropy maximum is log(K)
            
            # Normalize to [0, 1]
            minimum = entropy.min()
            maximum = entropy.max()
            
            entropy = (entropy - minimum) / (maximum - minimum)
            
            entropy = np.clip(entropy, 0, 1) # Clip values to [0 e 1]
            return entropy
            
        elif entropy_classes == EntropyClasses.Road:
            # Road Class is 1
            entropy = entropy + prob_mean[..., 1:1+1] * np.log(prob_mean[..., 1:1+1] + epsilon)
            
            entropy = - entropy / np.log(K) # scale entropy
            
            # Normalize to [0, 1]
            minimum = entropy.min()
            maximum = entropy.max()
            
            entropy = (entropy - minimum) / (maximum - minimum)
            
            entropy = np.clip(entropy, 0, 1) # Clip values to [0 e 1]
            return entropy
            
        elif entropy_classes == EntropyClasses.Back:
            # Road Class is 0
            entropy = entropy + prob_mean[..., 0:0+1] * np.log(prob_mean[..., 0:0+1] + epsilon)
            
            entropy = - entropy / np.log(K) # scale entropy
            
            # Normalize to [0, 1]
            minimum = entropy.min()
            maximum = entropy.max()
            
            entropy = (entropy - minimum) / (maximum - minimum)
            
            entropy = np.clip(entropy, 0, 1) # Clip values to [0 e 1]     
            return entropy
        
        raise Exception('Specified entropy classes to calculate must be '
                       f'{EntropyClasses.All}, {EntropyClasses.Road} or '
                       f'{EntropyClasses.Back}')
    
    def _save_entropy(self, group: str = EntropyGroups.Test, 
                          entropy_classes: str = EntropyClasses.All) -> bool:
        ''' Calculate and save entropy in ensemble dir for specific group and classes '''
        # Calculate entropy
        entropy = self.entropy(group=group, entropy_classes=entropy_classes)
        
        # Save result        
        np.save(self.ensemble_dir / f'entropy_{group}_{entropy_classes}.npy', entropy) # save array in disk
        
        return True
    
    def create_x_entropy_dir(self, x_dir: str, x_entropy_dir: str, 
                             entropy_classes: str = EntropyClasses.All, 
                             y_dir: str = None) -> bool:
        ''' Create X dir for entropy. To create datasets the y_dir must be passed '''
        # Load X Data
        x_train = np.load(Path(x_dir) / 'x_train.npy')
        x_valid = np.load(Path(x_dir) / 'x_valid.npy')
        x_test = np.load(Path(x_dir) / 'x_test.npy')
        
        # Calculate entropy for entropy classes
        for group in iter(EntropyGroups):
            self._save_entropy(group=group, entropy_classes=entropy_classes)
        
        # Load Entropy Data
        if entropy_classes in (EntropyClasses.All, EntropyClasses.Road, EntropyClasses.Back):        
            entropy_train = np.load(self.ensemble_dir / f'entropy_train_{entropy_classes}.npy')
            entropy_valid = np.load(self.ensemble_dir / f'entropy_valid_{entropy_classes}.npy')
            entropy_test = np.load(self.ensemble_dir / f'entropy_test_{entropy_classes}.npy')
        elif entropy_classes == EntropyClasses.Road_and_Back:
            entropy_train_road = np.load(self.ensemble_dir / f'entropy_train_{EntropyClasses.Road}.npy')
            entropy_valid_road = np.load(self.ensemble_dir / f'entropy_valid_{EntropyClasses.Road}.npy')
            entropy_test_road = np.load(self.ensemble_dir / f'entropy_test_{EntropyClasses.Road}.npy')
            
            entropy_train_back = np.load(self.ensemble_dir / f'entropy_train_{EntropyClasses.Back}.npy')
            entropy_valid_back = np.load(self.ensemble_dir / f'entropy_valid_{EntropyClasses.Back}.npy')
            entropy_test_back = np.load(self.ensemble_dir / f'entropy_test_{EntropyClasses.Back}.npy')
            
            entropy_train = np.concatenate((entropy_train_road, entropy_train_back), axis=-1) 
            entropy_valid = np.concatenate((entropy_valid_road, entropy_valid_back), axis=-1) 
            entropy_test =  np.concatenate((entropy_test_road, entropy_test_back), axis=-1)
            del entropy_train_road, entropy_valid_road, entropy_test_road
            del entropy_train_back, entropy_valid_back, entropy_test_back            
        
        # Concatenate X and entropy
        x_entropy_train = np.concatenate((x_train, entropy_train), axis=-1)
        x_entropy_valid = np.concatenate((x_valid, entropy_valid), axis=-1)
        x_entropy_test = np.concatenate((x_test, entropy_test), axis=-1)
        
        # Create dir and save numpy arrays of entropies to dir
        x_entropy_dir = Path(x_entropy_dir) # transform str in Path
        if not x_entropy_dir.exists():
            x_entropy_dir.mkdir()
        elif x_entropy_dir.exists():
            print('X entropy dir already exists. Data will be overwritten')
            
        if not (x_entropy_dir / 'x_train.npy').exists(): np.save(x_entropy_dir / 'x_train.npy', x_entropy_train)
        if not (x_entropy_dir / 'x_valid.npy').exists(): np.save(x_entropy_dir / 'x_valid.npy', x_entropy_valid)
        if not (x_entropy_dir / 'x_test.npy').exists(): np.save(x_entropy_dir / 'x_test.npy', x_entropy_test)
        
        # Create train and valid datasets if y_dir is given
        if y_dir:
            y_train = np.load(y_dir + 'y_train.npy')
            y_valid = np.load(y_dir + 'y_valid.npy')      
            
            y_train = onehot_numpy(y_train)
            y_valid = onehot_numpy(y_valid)
            
            train_dataset = tf.data.Dataset.from_tensor_slices((x_entropy_train, y_train))
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_entropy_valid, y_valid))
            
            train_dataset_path = x_entropy_dir / 'train_dataset'
            if train_dataset_path.exists():
                shutil.rmtree(train_dataset_path)
            train_dataset_path.mkdir(exist_ok=True)

            valid_dataset_path = x_entropy_dir / 'valid_dataset'
            if valid_dataset_path.exists():
                shutil.rmtree(valid_dataset_path)
            valid_dataset_path.mkdir(exist_ok=True)
            
            train_dataset.save(str(train_dataset_path))
            valid_dataset.save(str(valid_dataset_path))
            
        return True