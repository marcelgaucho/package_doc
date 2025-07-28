# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:10:16 2025

@author: Marcel
"""

# Class of directory containing data with uncertainty 

# %% Imports 

import shutil
from pathlib import Path
from .ensemble import DataGroups, Ensemble, EnsembleDir
import numpy as np
from ..treinamento.utils import onehot_numpy
from tensorflow.data import Dataset

# %% Enumerated constant for uncertainty metrics

class UncertaintyMetric:
    Entropy = 'entropy'
    Surprise = 'surprise'
    WeightedSurprise = 'weightedsurprise'
    MeanProb = 'meanprob'

# %% Class for X with Uncertainty dir

class XDirUncertain:
    def __init__(self, in_x_folder: str, y_folder: str, out_x_folder: str, 
                 ensemble_dir: EnsembleDir, metric: UncertaintyMetric,
                 min_scale_uncertainty=0, max_scale_uncertainty=1,
                 perc_cut=None):
        self.in_x_folder = Path(in_x_folder)
        self.y_folder = Path(y_folder)
        self.out_x_folder = Path(out_x_folder)
        self.ensemble_dir = ensemble_dir
        self.metric = metric
        self.min_scale_uncertainty = min_scale_uncertainty
        self.max_scale_uncertainty = max_scale_uncertainty
        self.perc_cut = perc_cut
        
    def _calculate_uncertainty(self, data_group):
        ensemble = Ensemble(ensemble_dir=self.ensemble_dir, data_group=data_group)
        if self.metric == UncertaintyMetric.Entropy:
            return ensemble.entropy(min_target_scale=self.min_scale_uncertainty, 
                                    max_target_scale=self.max_scale_uncertainty,
                                    perc_cut=self.perc_cut)
        elif self.metric == UncertaintyMetric.Surprise:
            return ensemble.surprise(min_target_scale=self.min_scale_uncertainty, 
                                     max_target_scale=self.max_scale_uncertainty,
                                     perc_cut=self.perc_cut)
        elif self.metric == UncertaintyMetric.WeightedSurprise:
            return ensemble.weighted_surprise(min_target_scale=self.min_scale_uncertainty, 
                                              max_target_scale=self.max_scale_uncertainty,
                                              perc_cut=self.perc_cut)
        elif self.metric == UncertaintyMetric.MeanProb:
            return ensemble.mean_prob(min_target_scale=self.min_scale_uncertainty, 
                                      max_target_scale=self.max_scale_uncertainty,
                                      perc_cut=self.perc_cut)
        
    def create(self):
        # Create x dir
        if not self.out_x_folder.exists():
            print("Creating ensemble directory")
            self.out_x_folder.mkdir()
        elif self.out_x_folder.exists():
            raise Exception("X uncertainty directory already exists")
    
    def insert_data(self):
        # Load previous X Data
        x_data = {data_group: np.load(self.in_x_folder / f'x_{data_group}.npy') for 
                  data_group in iter(DataGroups)}
            
        # Calculate uncertainty
        uncertainty = {data_group: self._calculate_uncertainty(data_group) 
                       for data_group in iter(DataGroups)}
        
        # Concatenate X with uncertainty and save arrays
        for data_group in iter(DataGroups):
            x_data[data_group] = np.concatenate((x_data[data_group], uncertainty[data_group]), axis=-1)
            np.save(self.out_x_folder / f'x_{data_group}.npy', x_data[data_group])
            
        # Create datasets
        self._create_datasets()        
            
    def _create_datasets(self):
        # Load X Data (with uncertainty) and Y Data for dataset groups
        dataset_groups = [DataGroups.Train, DataGroups.Valid]
        x_data = {data_group: np.load(self.out_x_folder / f'x_{data_group}.npy') for 
                  data_group in dataset_groups}
        
        y_data = {data_group: onehot_numpy( np.load(self.y_folder / f'y_{data_group}.npy') )
                  for data_group in dataset_groups}        

        # Create datasets
        datasets = {data_group: Dataset.from_tensor_slices((x_data[data_group], y_data[data_group]))
                    for data_group in dataset_groups}
        
        # Save datasets in specific folder for each dataset
        for data_group in dataset_groups:
            dataset_path = self.out_x_folder / f'{data_group}_dataset'
            try:
                shutil.rmtree(dataset_path)
            except FileNotFoundError:
                dataset_path.mkdir()
            datasets[data_group].save(str(dataset_path))
        