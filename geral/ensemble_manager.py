# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:02:38 2026

@author: Marcel
"""

# %% Import Libraries

import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from enum import Enum

from ..entropy.utils import uncert_metric_method
from ..entropy.utils import plot_uncertainty_histogram

# Import existing core classes
from ..treinamento.model_trainer import ModelTrainer
from ..avaliacao.evaluator import ModelEvaluator
from ..entropy.uncertain import UncertaintyCalculator



# %%

class EnsembleManager:
    """Orchestrates training, fine-tuning, evaluation, and uncertainty calculation for an ensemble."""
    
    def __init__(self, x_dir: str, y_dir: str, base_output_dir: str, base_model: tf.keras.Model, 
                 n_models: int = 5):
        self.x_dir = Path(x_dir)
        self.y_dir = Path(y_dir)
        self.base_output_dir = Path(base_output_dir)
        self.base_model = base_model
        self.n_models = n_models
        
        # Ensure base directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dirs(self):
        """Helper to generate or retrieve individual model directories."""
        return [self.base_output_dir / f'm_{i}' for i in range(self.n_models)]

    def train_all(self, optimizer_class, train_kwargs):
        """Loops through and trains all models in the ensemble."""
        for i, out_dir in enumerate(self._get_model_dirs()):
            out_dir.mkdir(exist_ok=True)
            print(f"--- Training Model {i+1}/{self.n_models} ---")
            
            # Rebuild optimizer for each iteration to refresh internal variables
            optimizer = optimizer_class()
            
            trainer = ModelTrainer(x_dir=self.x_dir, output_dir=out_dir, model=self.base_model, 
                                   optimizer=optimizer)
            trainer.train_with_loop(**train_kwargs)

    def fine_tune_all(self, optimizer_class, strategy, fine_tune_kwargs):
        """Loops through and fine-tunes all models from existing weights."""
        for i, out_dir in enumerate(self._get_model_dirs()):
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = out_dir / 'best_model.keras' # Assuming tuning in place or specify a source_dir
            
            print(f"--- Fine-Tuning Model {i+1}/{self.n_models} ---")
            
            trainer = ModelTrainer(x_dir=self.x_dir, output_dir=out_dir, model=self.base_model, 
                                   optimizer=optimizer_class())
            trainer.fine_tune(model_path=model_path, strategy=strategy, **fine_tune_kwargs)

    def evaluate_all(self, label_tiles_dir, eval_kwargs, mosaic_kwargs, eval_mosaic_kwargs):
        """Evaluates all trained models."""
        for i, model_dir in enumerate(self._get_model_dirs()):
            print(f"--- Evaluating Model {i+1}/{self.n_models} ---")
            evaluator = ModelEvaluator(
                x_dir=self.x_dir, y_dir=self.y_dir, output_dir=model_dir, 
                label_tiles_dir=label_tiles_dir
            )
            evaluator.evaluate_model(**eval_kwargs)
            evaluator.build_test_mosaics(**mosaic_kwargs)
            evaluator.evaluate_mosaics(**eval_mosaic_kwargs)
            
    def calculate_uncertainty(self, data_groups: list, scale_result: bool, metric: Enum, 
                              metric_kwargs):
        """Calculates ensemble uncertainty and exports mosaics."""
        print("--- Calculating Ensemble Uncertainty ---")
        uncertainty_dir = self.base_output_dir / 'uncertainty'
        uncertainty_dir.mkdir(exist_ok=True)
        
        # Calculate uncertainty of data groups
        results = []
        for data_group in data_groups:
            # Pass the pre-computed model directories directly to your existing UncertaintyCalculator
            calc = UncertaintyCalculator(
                model_dirs=self._get_model_dirs(), 
                data_group=data_group, 
                scale_result=scale_result
            )
            
            # Dynamically call the requested metric (entropy, surprise, etc.) and append to results list
            metric_func = getattr(calc, uncert_metric_method[metric])
            result = metric_func(**metric_kwargs)
            results.append(result)
            
            # Save array
            np.save(uncertainty_dir / f'{metric.name.lower()}_{data_group.value}.npy', result)
            
        return results