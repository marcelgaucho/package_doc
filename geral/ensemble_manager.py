# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:02:38 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
import os
import json
import tensorflow as tf
from pathlib import Path
import numpy as np
from enum import Enum

from ..entropy.utils import plot_uncertainty_histogram
from ..entropy.utils import UncertaintyMetric
from ..avaliacao.mosaics import MosaicGenerator
from ..avaliacao.utils import stack_uneven, load_reference_mosaics

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
            
    def calculate_uncertainty(self, data_groups: list, scale_result: bool, metric: UncertaintyMetric, 
                              metric_kwargs: dict, label_tiles_dir: str = None, 
                              info_tiles_name: str = 'info_tiles_test.json',
                              ignore_index: int = 255, export_mosaics: bool = True, 
                              generate_plot: bool = True):
        """Calculates ensemble uncertainty and exports mosaics."""
        print("--- Calculating Ensemble Uncertainty ---")
        uncertainty_dir = self.base_output_dir / 'uncertainty'
        uncertainty_dir.mkdir(exist_ok=True)
        
        # Calculate uncertainty of data groups
        results = []
        for data_group in data_groups:
            # 1. Calculate Uncertainty
            calc = UncertaintyCalculator(
                model_dirs=self._get_model_dirs(), 
                data_group=data_group, 
                scale_result=scale_result
            )
            
            # Dynamically call the method directly from the Enum value
            metric_func = getattr(calc, metric.value)
            uncertainty_array = metric_func(**metric_kwargs)
            results.append(uncertainty_array)
            
            np.save(uncertainty_dir / f'{metric.value}_{data_group.value}.npy', uncertainty_array)
            
            # Load the true labels for this specific data group to mask ignored pixels
            y_array = np.load(self.y_dir / f'y_{data_group.value}.npy')
            
            # 2. Generate Plot (for Test set)
            if generate_plot and data_group.value == 'test':
                self._plot_uncertainty(
                    uncertainty_array=uncertainty_array, 
                    y_array=y_array, 
                    ignore_index=ignore_index, 
                    metric=metric, 
                    data_group=data_group, 
                    out_dir=uncertainty_dir
                )
            
            # 3. Build and Export Mosaics (only applies to Test set with tiles)
            if export_mosaics and label_tiles_dir and data_group.value == 'test':
                self._export_uncertainty_mosaics(
                    uncertainty_array=uncertainty_array,
                    y_array=y_array,
                    label_tiles_dir=Path(label_tiles_dir),
                    info_tiles_name=info_tiles_name,
                    ignore_index=ignore_index,
                    metric=metric,
                    out_dir=uncertainty_dir
                )
                
        return results
    
    def _plot_uncertainty(self, uncertainty_array, y_array, ignore_index, metric, data_group, out_dir):
        """Private helper to isolate plotting logic."""
        # Filter out ignored background pixels before plotting
        valid_pixels = uncertainty_array[y_array != ignore_index]
        
        title = f"{metric.display_name} Distribution ({data_group.value.capitalize()})"
        save_path = out_dir / f"{metric.value}_plot_{data_group.value}.png"
        
        plot_uncertainty_histogram(data=valid_pixels, title=title, log_scale=True, save_path=save_path)
        print(f"Saved plot: {save_path.name}")

    def _export_uncertainty_mosaics(self, uncertainty_array, y_array, label_tiles_dir, info_tiles_name, ignore_index, metric, out_dir):
        """Private helper to isolate mosaic generation and exporting logic."""
        # Load JSON configuration for tiles
        info_tiles_path = self.y_dir / info_tiles_name
        if not info_tiles_path.exists():
            print(f"Skipping mosaics: {info_tiles_path.name} not found.")
            return

        with open(info_tiles_path) as fp:
            info_tiles = json.load(fp)

        # Generate standard mosaics
        mosaics = MosaicGenerator(
            test_array=uncertainty_array, 
            info_tiles=info_tiles, 
            tiles_dir=label_tiles_dir, 
            output_dir=out_dir
        )
        mosaics.build_mosaics(dtype=np.float32)

        # Mask ignored regions directly on the generated mosaics based on reference shapes
        y_mosaics = load_reference_mosaics(label_tiles_dir)

        if ignore_index in y_mosaics:
            mosaics.mosaics[y_mosaics == ignore_index] = ignore_index

        prefix = f'mosaic_{metric.value}'
        mosaics.export_mosaics(prefix=prefix)
        print(f"Exported mosaics with prefix: {prefix}")