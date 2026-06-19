# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:38:38 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np
from sklearn.utils.extmath import stable_cumsum

import json, pickle, warnings

from .buffer_function import buffer_patches_array, buffer_valid_array
from .utils import ignore_small_areas
from .metrics_calc_prec_rec_f1 import calculate_relaxed_prec_recall_f1
from .metric_calc_avg_prec import calculate_relaxed_avg_precision

from skimage import morphology

from pathlib import Path

# %%

class RelaxedMetricCalculator:
    def __init__(self, y_array: np.ndarray, pred_array: np.ndarray, prob_array: np.ndarray = None, 
                 buffer_px: int = 3, ignore_index: int = 255, min_area_px: int = None):
        self.ignore_index = ignore_index
        self.buffer_px = buffer_px
        self.min_area_px = min_area_px
        
        self.y_array = y_array
        self.prob_array = prob_array
        self.mask = (self.y_array != ignore_index) # y true mask
        
        self.pred_array = pred_array
        
        self.metrics = None
        self.ap_lists = {}

    def calculate_metrics(self, value_zero_division: float = None, include_avg_precision: bool = False, 
                          print_interval: int = 200, interpolated_avg_precision: bool = True,
                          include_ece: bool = False) -> dict:
        
        if include_avg_precision and self.prob_array is None:
            raise ValueError("prob_array must be set during initialization to calculate average precision.")
        
        # Ignore also the small areas in pred array if min_area_px is passed to object
        if self.min_area_px:
            mask = self.mask & (self.pred_array != self.ignore_index)
        else:
            mask = self.mask
        
        # Buffer and mask arrays
        buffer_y_array = buffer_valid_array(self.y_array, mask, self.buffer_px)[mask]
        buffer_pred_array = buffer_valid_array(self.pred_array, mask, self.buffer_px)[mask]
        
        pred_array = self.pred_array[mask]
        y_array = self.y_array[mask]
        
        self.metrics = calculate_relaxed_prec_recall_f1(y=y_array,
                                                        buffer_y=buffer_y_array, 
                                                        pred=pred_array, 
                                                        buffer_pred=buffer_pred_array, 
                                                        value_zero_division=value_zero_division)
        # Average precision calculus
        if include_avg_precision:
            self.metrics['relaxed_avg_precision'], \
            self.ap_lists = calculate_relaxed_avg_precision(y=self.y_array, 
                                                            prob=self.prob_array,
                                                            buffer_px=self.buffer_px,
                                                            interpolated=interpolated_avg_precision,
                                                            ignore_index=self.ignore_index,
                                                            min_area_px=self.min_area_px)
        
        if include_ece:
            self.metrics['ece'] = ...
            
       
        return self.metrics        
        
    def export_results(self, output_dir: Path, group: str = 'test', export_ap_lists: bool = True):
        if group not in ('train', 'valid', 'test', 'mosaics'):
            raise ValueError("Group must be 'train', 'valid', 'test', or 'mosaics'")
        
        if self.metrics is None:
            raise RuntimeError("Metrics not calculated. Call calculate_metrics() first.")
        
        with open(output_dir / f'relaxed_metrics_{group}_{self.buffer_px}px.json', 'w') as f:
            json.dump(self.metrics, f)
            
        if export_ap_lists and self.ap_lists:
            with open(output_dir / f'avg_precision_lists_{group}_{self.buffer_px}px.pickle', "wb") as fp: 
                pickle.dump(self.ap_lists, fp)