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

    def _calculate_avg_precision(self, print_interval: int, interpolated: bool) -> float:
        prob_flat = self.prob_array[self.mask] 
        
        desc_score_indices = np.argsort(prob_flat, kind="mergesort")[::-1]
        prob_flat = prob_flat[desc_score_indices]
        
        diff_scores = np.diff(prob_flat)
        change_idxs = np.where(diff_scores)[0]        
        threshold_idxs = np.r_[change_idxs, prob_flat.size - 1]
        thresholds = prob_flat[threshold_idxs]
        
        # Initialize metric lists
        cumtp_thres_recall = []     
        cumtp_thres_prec = []
        cumpos_thres = []
        
        # Standard fast-path if min_area is NOT used
        if self.min_area_px is None:
            buffer_y_flat = self.buffer_y_array[desc_score_indices]
            cumtp_prec = stable_cumsum(buffer_y_flat)
            cumtp_thres_prec = cumtp_prec[threshold_idxs]
            cumpos_thres = threshold_idxs + 1
        else:
            # PERFORMANCE SAFETY: Downsample thresholds if applying spatial filters dynamically
            if len(thresholds) > 100:
                warnings.warn(f"Downsampling {len(thresholds)} thresholds to 100 to prevent massive execution times from spatial filtering.")
                thresholds = np.linspace(1.0, 0.0, 100)
    
        actual_pos = self.y_array.sum()   
    
        for i, threshold in enumerate(thresholds):
            if print_interval and i % print_interval == 0:
                print(f'Calculating threshold {i:>6d}/{len(thresholds):>6d}')
            
            # 1. Base prediction for threshold
            pred = (self.prob_array >= threshold).astype(int)
            
            # 2. Dynamically remove small areas
            if self.min_area_px:
                pred = ignore_small_areas(pred, self.min_area_px, self.ignore_index)
            
            # 3. Recall Calculation (Requires buffering the filtered prediction)
            # TODO: small ignored areas must "sum" with other ignored areas in the mask
            # mask = self.mask_ytrue & (pred != self.ignore_index)
            pred_buffer = self._buffer_array(pred)[self.mask]
            true_pos_rec = (self.y_array * pred_buffer).sum()
            cumtp_thres_recall.append(true_pos_rec)        
            
            # 4. Precision Calculation (Requires masking the filtered prediction)
            if self.min_area_px:
                pred_masked = pred[self.mask]
                pred_pos = pred_masked.sum()
                true_pos_prec = (self.buffer_y_array * pred_masked).sum()
                
                cumpos_thres.append(pred_pos)
                cumtp_thres_prec.append(true_pos_prec)
    
        cumtp_thres_recall = np.array(cumtp_thres_recall) 
        
        if self.min_area_px:
            cumtp_thres_prec = np.array(cumtp_thres_prec)
            cumpos_thres = np.array(cumpos_thres)
        
        # Calculate Precision
        precision = np.zeros_like(cumtp_thres_prec, dtype=float)
        np.divide(cumtp_thres_prec, cumpos_thres, out=precision, where=(cumpos_thres != 0))
        
        precision = np.hstack((precision[::-1], 1))
        if interpolated:
            precision = np.maximum.accumulate(precision)            
            
        self.ap_lists['precision'] = precision 
        
        # Calculate Recall
        if actual_pos == 0:
            print("No positive class found in y_true, recall is set to 1 for all thresholds.")
            recall = np.ones_like(precision)
        else:
            recall = cumtp_thres_recall / actual_pos
            
        recall = np.hstack((recall[::-1], 0))
        self.ap_lists['recall'] = recall
        self.ap_lists['thresholds'] = thresholds[::-1] 
    
        return np.sum(-np.diff(recall) * precision[:-1]) 
    
    def calculate_metrics(self, value_zero_division: float = None, include_avg_precision: bool = False, 
                          print_interval: int = 200, interpolated_avg_precision: bool = True) -> dict:
        
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
        # TODO: Complete average precision
        if include_avg_precision:
            pass
            # self.metrics['relaxed_avg_precision'],
            # self.ap_lists = calculate_relaxed_avg_precision(y=y_array, 
            #                                                 prob=self.prob_array,
            #                                                 mask,
            #                                                 self.buffer_px,
                                                            
            #                                                 print_interval=print_interval,
            #     interpolated=interpolated_avg_precision
            # )
        
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