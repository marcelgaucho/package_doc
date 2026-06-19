# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:37:58 2026

@author: Marcel
"""

# %% Import Libraries

import warnings
import numpy as np
from .utils import ignore_small_areas
from .buffer_function import buffer_valid_array


# %%

def calculate_relaxed_avg_precision(y: np.ndarray, prob: np.ndarray, buffer_px: int,
                                    print_interval: int = 200,
                                    interpolated: bool = False,
                                    ignore_index: int = 255, 
                                    min_area_px: int = None) -> tuple:
    """
    Calculates the relaxed average precision, allowing for spatial tolerance 
    and optionally ignoring small predicted areas. Fast-tracked for buffer_px = 0.
    """
    # 1. Base Mask: Universally valid pixels (ignoring background/border index)
    base_valid_mask = (y != ignore_index)
    ap_lists = {}
    
    # 2. Filter probabilities by the base mask and calculate thresholds
    prob_flat = prob[base_valid_mask]
    y_flat = y[base_valid_mask]
    
    desc_score_indices = np.argsort(prob_flat, kind="mergesort")[::-1]
    prob_flat = prob_flat[desc_score_indices]
    y_flat = y_flat[desc_score_indices] # Also sort y to match probabilities
    
    diff_scores = np.diff(prob_flat)
    change_idxs = np.where(diff_scores)[0]        
    threshold_idxs = np.r_[change_idxs, prob_flat.size - 1]
    thresholds = prob_flat[threshold_idxs]
    
    actual_pos = y_flat.sum()   

    # =========================================================================
    # OPTIMIZED FAST-PATH: No spatial relaxation (buffer = 0) and no area filtering
    # =========================================================================
    if min_area_px is None and buffer_px == 0:
        if actual_pos == 0:
            print("No positive class found in y_true, recall is set to 1 for all thresholds.")
            precision = np.ones(len(thresholds) + 1, dtype=float)
            recall = np.ones(len(thresholds) + 1, dtype=float)
        else:
            # Vectorized standard AP computation via cumulative sums
            cumtp = np.cumsum(y_flat)
            
            # Extract cumulative true positives at the threshold boundaries
            cumtp_thres = cumtp[threshold_idxs]
            cumpos_thres = threshold_idxs + 1
            
            # Precision = TP / (TP + FP)
            precision = cumtp_thres / cumpos_thres
            precision = np.hstack((precision[::-1], 1.0))
            
            # Recall = TP / Actual Positives
            recall = cumtp_thres / actual_pos
            recall = np.hstack((recall[::-1], 0.0))

        if interpolated:
            precision = np.maximum.accumulate(precision)   
            
        ap_lists['precision'] = precision 
        ap_lists['recall'] = recall
        ap_lists['thresholds'] = thresholds[::-1] 

        avg_precision = np.sum(-np.diff(recall) * precision[:-1])
        return avg_precision, ap_lists

    # =========================================================================
    # STANDARD/RELAXED PATHS (Executed when buffer > 0 or min_area_px is set)
    # =========================================================================
    cumtp_thres_recall = []     
    cumtp_thres_prec = []
    cumpos_thres = []
    
    if min_area_px is None:
        # We know buffer_px > 0 here because buffer_px == 0 was caught above
        buffer_y = buffer_valid_array(y, base_valid_mask, buffer_px)[base_valid_mask]
        buffer_y_flat = buffer_y[desc_score_indices]
        cumtp_prec = np.cumsum(buffer_y_flat) # stable_cumsum replaced with np.cumsum
        cumtp_thres_prec = cumtp_prec[threshold_idxs]
        cumpos_thres = threshold_idxs + 1
    else:
        if len(thresholds) > 100:
            warnings.warn(f"Downsampling {len(thresholds)} thresholds to 100 to prevent massive execution times from spatial filtering.")
            thresholds = np.linspace(1.0, 0.0, 100)

    for i, threshold in enumerate(thresholds):
        if print_interval and i % print_interval == 0:
            print(f'Calculating threshold {i:>6d}/{len(thresholds):>6d}')
        
        pred = (prob >= threshold).astype(int)
        
        if min_area_px:
            pred = ignore_small_areas(pred, min_area_px, ignore_index)
            current_mask = base_valid_mask & (pred != ignore_index)
            
            current_y = y[current_mask]
            current_pred = pred[current_mask]
            
            pred_buffer = buffer_valid_array(pred, current_mask, buffer_px)[current_mask]
            current_buffer_y = buffer_valid_array(y, current_mask, buffer_px)[current_mask]
            
            true_pos_rec = (current_y * pred_buffer).sum()
            pred_pos = current_pred.sum()
            true_pos_prec = (current_buffer_y * current_pred).sum()
            
            cumpos_thres.append(pred_pos)
            cumtp_thres_prec.append(true_pos_prec)
        else:
            pred_buffer = buffer_valid_array(pred, base_valid_mask, buffer_px)[base_valid_mask]
            true_pos_rec = (y[base_valid_mask] * pred_buffer).sum()
            
        cumtp_thres_recall.append(true_pos_rec)        
        
    cumtp_thres_recall = np.array(cumtp_thres_recall) 
    
    if min_area_px:
        cumtp_thres_prec = np.array(cumtp_thres_prec)
        cumpos_thres = np.array(cumpos_thres)
    
    precision = np.zeros_like(cumtp_thres_prec, dtype=float)
    np.divide(cumtp_thres_prec, cumpos_thres, out=precision, where=(cumpos_thres != 0))
    
    precision = np.hstack((precision[::-1], 1))
    if interpolated:
        precision = np.maximum.accumulate(precision)            
        
    ap_lists['precision'] = precision 
    
    if actual_pos == 0:
        print("No positive class found in y_true, recall is set to 1 for all thresholds.")
        recall = np.ones_like(precision)
    else:
        recall = cumtp_thres_recall / actual_pos
        
    recall = np.hstack((recall[::-1], 0))
    ap_lists['recall'] = recall
    ap_lists['thresholds'] = thresholds[::-1] 

    avg_precision = np.sum(-np.diff(recall) * precision[:-1])
    return avg_precision, ap_lists