# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:37:58 2026

@author: Marcel
"""

# %% Import Libraries

import warnings
import numpy as np
from sklearn.utils.extmath import stable_cumsum
from .utils import ignore_small_areas
from .buffer_function import buffer_valid_array


# %%


def calculate_relaxed_avg_precision(y, prob, valid_mask, buffer_px,
                                    print_interval=200,
                                    interpolated=False,
                                    ignore_index=255, min_area_px=None) -> float:
    ap_lists = {} # Average precision lists to fill later
    
    # Buffer the y array
    buffer_y = buffer_valid_array(y, valid_mask, buffer_px)
    
    prob_flat = prob[valid_mask] # Filter by valid mask, returns a flat array 
    
    # Sort probabilities in descending order
    desc_score_indices = np.argsort(prob_flat, kind="mergesort")[::-1]
    prob_flat = prob_flat[desc_score_indices]
    
    # Calculate Thresholds using changing indexes
    diff_scores = np.diff(prob_flat)
    change_idxs = np.where(diff_scores)[0]        
    threshold_idxs = np.r_[change_idxs, prob_flat.size - 1]
    thresholds = prob_flat[threshold_idxs]
    
    # Initialize metric lists
    cumtp_thres_recall = []     
    cumtp_thres_prec = []
    cumpos_thres = []
    
    # Standard fast-path if min_area is NOT used
    if min_area_px is None:
        # Calculate Cumulative TP and Cumulative Predicted Positives for Precision
        buffer_y_flat = buffer_y[desc_score_indices] # sort y buffer in descending order
        cumtp_prec = stable_cumsum(buffer_y_flat)
        cumtp_thres_prec = cumtp_prec[threshold_idxs]
        cumpos_thres = threshold_idxs + 1
    else:
        # PERFORMANCE SAFETY: Downsample thresholds if applying spatial filters dynamically
        if len(thresholds) > 100:
            warnings.warn(f"Downsampling {len(thresholds)} thresholds "
                          "to 100 to prevent massive execution times from spatial filtering.")
            thresholds = np.linspace(1.0, 0.0, 100)

    actual_pos = y.sum()   

    for i, threshold in enumerate(thresholds):
        if print_interval and i % print_interval == 0:
            print(f'Calculating threshold {i:>6d}/{len(thresholds):>6d}')
        
        # 1. Base prediction for threshold
        pred = (prob >= threshold).astype(int)
        
        # 2. Dynamically ignore small areas
        if min_area_px:
            pred = ignore_small_areas(pred, min_area_px, ignore_index)
        
        # 3. Recall Calculation (Requires buffering the filtered prediction)
        # TODO: small ignored areas must "sum" with other ignored areas in the mask
        mask = valid_mask & (pred != ignore_index)
        pred_buffer = buffer_valid_array(pred, mask, buffer_px)[mask]
        true_pos_rec = (y * pred_buffer).sum()
        cumtp_thres_recall.append(true_pos_rec)        
        
        # 4. Precision Calculation (Requires masking the filtered prediction)
        if min_area_px:
            pred_masked = pred[mask]
            pred_pos = pred_masked.sum()
            true_pos_prec = (buffer_y * pred_masked).sum()
            
            cumpos_thres.append(pred_pos)
            cumtp_thres_prec.append(true_pos_prec)

    cumtp_thres_recall = np.array(cumtp_thres_recall) 
    
    if min_area_px:
        cumtp_thres_prec = np.array(cumtp_thres_prec)
        cumpos_thres = np.array(cumpos_thres)
    
    # Calculate Precision
    precision = np.zeros_like(cumtp_thres_prec, dtype=float)
    np.divide(cumtp_thres_prec, cumpos_thres, out=precision, where=(cumpos_thres != 0))
    
    precision = np.hstack((precision[::-1], 1))
    if interpolated:
        precision = np.maximum.accumulate(precision)            
        
    ap_lists['precision'] = precision 
    
    # Calculate Recall
    if actual_pos == 0:
        print("No positive class found in y_true, recall is set to 1 for all thresholds.")
        recall = np.ones_like(precision)
    else:
        recall = cumtp_thres_recall / actual_pos
        
    recall = np.hstack((recall[::-1], 0))
    ap_lists['recall'] = recall
    ap_lists['thresholds'] = thresholds[::-1] 

    return np.sum(-np.diff(recall) * precision[:-1])