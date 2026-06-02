# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:37:06 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
from pathlib import Path
import numpy as np
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
from .utils import stack_uneven, ignore_small_areas

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
        self.y_mosaics = self._load_reference_mosaics() if self.label_tiles_dir else None

    def _load_reference_mosaics(self) -> np.ndarray:
        labels_paths = sorted([
            str(p) for p in self.label_tiles_dir.iterdir() 
            if p.suffix in ('.tiff', '.tif')
        ])
        y_mosaics = [gdal.Open(path).ReadAsArray() for path in labels_paths]
        return stack_uneven(y_mosaics)[..., np.newaxis]

    def model_predict(self, x_array: np.ndarray, batch_step: int = 2) -> tuple:
        prob = self.model.predict(x_array, batch_size=batch_step, verbose=1)
        pred = np.argmax(prob, axis=-1)[..., np.newaxis]
        prob = prob[..., 1:2]  # Only road probabilities (Class 1)
        return prob, pred

    def _evaluate_split(self, split_name: str, buffers_px: list, include_avg_precision: bool):
        """Generic method to evaluate any data split, reducing code duplication."""
        prob_path = self.output_dir / f'prob_{split_name}.npy'
        pred_path = self.output_dir / f'pred_{split_name}.npy'

        # Predict and Save if files do not exist
        if not (prob_path.exists() and pred_path.exists()):
            # Lazy load data only when needed to save RAM
            x_array = np.load(self.x_dir / f'x_{split_name}.npy')
            prob, pred = self.model_predict(x_array=x_array)
            
            # Cast to occupy less space
            np.save(prob_path, prob.astype(np.float16))
            np.save(pred_path, pred.astype(np.uint8))
            
            # Free memory immediately after prediction
            del x_array 
        
        # Load targets and predictions
        y_array = np.load(self.y_dir / f'y_{split_name}.npy')
        pred_array = np.load(pred_path)
        prob_array = np.load(prob_path) if include_avg_precision else None
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            calculator = RelaxedMetricCalculator(
                y_array=y_array, 
                pred_array=pred_array, 
                buffer_px=buffer_px, 
                prob_array=prob_array
            )
            calculator.calculate_metrics(include_avg_precision=include_avg_precision)
            calculator.export_results(output_dir=self.output_dir, group=split_name)

    def evaluate_model(self, splits: list = ['valid', 'test'], buffers_px: list = [3], include_avg_precision: bool = False):
        """Evaluates the model on specified dataset splits."""
        for split in splits:
            if split in ['train', 'valid', 'test']:
                self._evaluate_split(split, buffers_px, include_avg_precision)
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
        


    def evaluate_mosaics(self, buffers_px: list = [3], include_avg_precision: bool = False, min_area_px: int = None):
        if self.y_mosaics is None:
            raise ValueError("Reference mosaics not loaded. Provide label_tiles_dir on initialization.")
            
        prob_mosaics = np.load(self.output_dir / 'prob_mosaics.npy') if include_avg_precision else None
        pred_mosaics = np.load(self.output_dir / 'pred_mosaics.npy')
        
        for buffer_px in buffers_px:
            calculator = RelaxedMetricCalculator(
                y_array=self.y_mosaics, 
                pred_array=pred_mosaics, 
                buffer_px=buffer_px, 
                prob_array=prob_mosaics, 
                min_area_px=min_area_px
            )
            calculator.calculate_metrics(include_avg_precision=include_avg_precision)
            calculator.export_results(output_dir=self.output_dir, group='mosaics')