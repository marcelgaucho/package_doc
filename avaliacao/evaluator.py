# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:37:06 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import warnings

# %% Import from own modules

from ..treinamento.arquiteturas.unetr_2d import Patches
from ..treinamento.arquiteturas.segformer_tf_k2.models.modules import MixVisionTransformer
from ..treinamento.arquiteturas.segformer_tf_k2.models.Head import SegFormerHead
from ..treinamento.arquiteturas.segformer_tf_k2.models.utils import ResizeLayer
from ..treinamento.model_trainer import ModelTrainer

from .metric_calculator import RelaxedMetricCalculator
from .mosaics import MosaicGenerator
from .utils import stack_uneven, ignore_small_areas, load_reference_mosaics, decode_onehot

# %%


class ModelEvaluator:
    def __init__(self, x_dir: str, y_dir: str, output_dir: str, label_tiles_dir: str = None):
        self.x_dir = Path(x_dir)
        self.y_dir = Path(y_dir)
        self.output_dir = Path(output_dir)
        self.label_tiles_dir = Path(label_tiles_dir) if label_tiles_dir else None
        
        # Load Model
        self.model = load_model(
            self.output_dir / ModelTrainer.best_model_filename, 
            compile=False, 
            custom_objects={
                "Patches": Patches, 
                "MixVisionTransformer": MixVisionTransformer,
                "SegFormerHead": SegFormerHead,
                "ResizeLayer": ResizeLayer
            }
        )
        
        # Mosaics are loaded only if the directory is provided
        self.y_mosaics = load_reference_mosaics(self.label_tiles_dir) if self.label_tiles_dir else None

    def model_predict(self, x_array: np.ndarray, batch_step: int = 2) -> tuple:
        prob = self.model.predict(x_array, batch_size=batch_step, verbose=1)
        pred = np.argmax(prob, axis=-1)[..., np.newaxis]
        prob = prob[..., 1:2]  # Only road probabilities (Class 1)
        return prob, pred
    
    def _load_split_data(self, split_name: str, load_x: bool = True, load_y: bool = True,
                         normalize: bool = True) -> tuple:
        """
        Dynamically loads split data. Tries .npy files first, then falls back 
        to extracting arrays from a tf.data.Dataset folder.
        """
        x_npy_path = self.x_dir / f'x_{split_name}.npy'
        y_npy_path = self.y_dir / f'y_{split_name}.npy'
        dataset_path = self.x_dir / f'{split_name}_dataset'

        x_array, y_array = None, None

        # Path A: Load from Numpy Arrays (Fastest)
        if x_npy_path.exists() and y_npy_path.exists():
            if load_x: 
                x_array = np.load(x_npy_path)
            if load_y: 
                y_array = np.load(y_npy_path)
            return x_array, y_array

        # Path B: Extract from TensorFlow Dataset
        if dataset_path.exists():
            ds = tf.data.Dataset.load(str(dataset_path))
            x_list, y_list = [], []

            for x_batch, y_batch in ds:
                if load_x: 
                    x_list.append(x_batch.numpy())
                if load_y: 
                    y_list.append(y_batch.numpy())

            if load_x:
                x_array = np.stack(x_list, axis=0)
                if normalize:
                    x_array = x_array.astype(np.float32)/255 # Normalize for Massachusetts dataset (to save space, in these dataset train and valid data 
                                                             # aren't saved as .npy in order to save space)
            if load_y:
                y_array_onehot = np.stack(y_list, axis=0)
                y_array = decode_onehot(y_array_onehot, ignore_index=255)

            return x_array, y_array

        raise FileNotFoundError(f"Data for split '{split_name}' not found in {self.x_dir} or {self.y_dir}. Missing both .npy and _dataset folder.")

    def _evaluate_split(self, split_name: str, buffers_px: list, include_avg_precision: bool,
                        include_ece: bool):
        """Generic method to evaluate any data split, reducing code duplication."""
        prob_path = self.output_dir / f'prob_{split_name}.npy'
        pred_path = self.output_dir / f'pred_{split_name}.npy'

        # RAM OPTIMIZATION: Only load x_array if predictions are actually needed
        needs_prediction = not (prob_path.exists() and pred_path.exists())
        
        # Route through the new dynamic loader
        x_array, y_array = self._load_split_data(split_name, load_x=needs_prediction, load_y=True)

        if needs_prediction:
            prob, pred = self.model_predict(x_array=x_array)
            
            # Cast to occupy less space
            np.save(prob_path, prob.astype(np.float32))
            np.save(pred_path, pred.astype(np.uint8))
            
            # Free memory immediately after prediction
            del x_array 
        
        # Load predictions and probabilities
        pred_array = np.load(pred_path)
        prob_array = np.load(prob_path) if include_avg_precision or include_ece else None
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            calculator = RelaxedMetricCalculator(
                y_array=y_array, 
                pred_array=pred_array, 
                buffer_px=buffer_px, 
                prob_array=prob_array
            )
            calculator.calculate_metrics(
                include_avg_precision=include_avg_precision, 
                include_ece=include_ece,
                ece_strategy='adaptive', 
                ece_bins=10
            )
            calculator.export_results(output_dir=self.output_dir, group=split_name)

    def evaluate_model(self, splits: list = ['valid', 'test'], buffers_px: list = [3], 
                       include_avg_precision: bool = False, 
                       include_ece: bool = False):
        """Evaluates the model on specified dataset splits."""
        for split in splits:
            if split in ['train', 'valid', 'test']:
                self._evaluate_split(split, buffers_px, include_avg_precision, include_ece)
            else:
                warnings.warn(f"Unknown split '{split}' ignored.")

    def build_test_mosaics(self, prefix: str = 'outmosaic', export_pred_mosaics: bool = False, 
                           export_prob_mosaics: bool = False, ignore_index: int = 255,
                           min_area_px=None):
        if not self.label_tiles_dir:
            raise ValueError("Cannot build mosaics: label_tiles_dir was not provided.")
            
        print('Building mosaics using pred_test from output directory...')
        
        prob_test = np.load(self.output_dir / 'prob_test.npy')
        pred_test = np.load(self.output_dir / 'pred_test.npy')
        
        with open(self.y_dir / 'info_tiles_test.json') as fp:   
            info_tiles_test = json.load(fp)
        
        ignore_in_y = ignore_index in self.y_mosaics 
        
        # Helper function to keep DRY
        def _process_mosaic(test_array, dtype, file_prefix, export_flag, out_prefix):
            mosaics = MosaicGenerator(test_array=test_array, info_tiles=info_tiles_test, 
                                      tiles_dir=self.label_tiles_dir, output_dir=self.output_dir)
            mosaics.build_mosaics(dtype=dtype)
            if ignore_in_y:
                mosaics.mosaics[self.y_mosaics == ignore_index] = ignore_index                
            if min_area_px and file_prefix == 'pred':
                print(f'Area elements smaller than {min_area_px} px will be removed')
                mosaics.mosaics = ignore_small_areas(mosaics.mosaics, 
                                                     min_area_px=min_area_px,
                                                     ignore_index=ignore_index)                
            mosaics.save_mosaics(prefix=file_prefix)
            if export_flag:
                mosaics.export_mosaics(prefix=out_prefix)

        _process_mosaic(pred_test, np.uint8, 'pred', export_pred_mosaics, f"{prefix}_pred_")
        _process_mosaic(prob_test, np.float32, 'prob', export_prob_mosaics, f"{prefix}_prob_")

    def evaluate_mosaics(self, buffers_px: list = [3], include_avg_precision: bool = False, min_area_px: int = None, 
                         include_ece: bool = True):
        if self.y_mosaics is None:
            raise ValueError("Reference mosaics not loaded. Provide label_tiles_dir on initialization.")
            
        prob_mosaics = np.load(self.output_dir / 'prob_mosaics.npy') if include_avg_precision or include_ece else None
        pred_mosaics = np.load(self.output_dir / 'pred_mosaics.npy')
        
        for buffer_px in buffers_px:
            calculator = RelaxedMetricCalculator(
                y_array=self.y_mosaics, 
                pred_array=pred_mosaics, 
                buffer_px=buffer_px, 
                prob_array=prob_mosaics, 
                min_area_px=min_area_px
            )
            calculator.calculate_metrics(include_avg_precision=include_avg_precision,
                                         include_ece=include_ece,
                                         ece_strategy='adaptive', # Defaulting to the equal-mass strategy
                                         ece_bins=10)
            calculator.export_results(output_dir=self.output_dir, group='mosaics')