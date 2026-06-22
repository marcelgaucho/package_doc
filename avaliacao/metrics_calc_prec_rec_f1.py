# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:40:00 2026

@author: Marcel
"""

# %% Import Libraries

import warnings

# %%

def calculate_relaxed_prec_recall_f1(y, buffer_y, pred, buffer_pred, 
                                     value_zero_division: float = None) -> dict:
    epsilon = 1e-7 
    
    # In precision TP is obtained with a buffer in the reference
    true_positive_precision = (buffer_y * pred).sum()
    predicted_positive = pred.sum()
    
    # In recall TP is obtained with a buffer in the predicion
    true_positive_recall = (y * buffer_pred).sum() 
    actual_positive = y.sum()

    relaxed_precision = true_positive_precision / (predicted_positive + epsilon)
    relaxed_recall = true_positive_recall / (actual_positive + epsilon)
    
    if predicted_positive == 0:
        warnings.warn("Predicted positives are equal to 0.")
        if value_zero_division is not None:
            relaxed_precision = value_zero_division
        
    if actual_positive == 0:
        warnings.warn("Actual positives are equal to 0.")
        if value_zero_division is not None:
            relaxed_recall = value_zero_division
    
    relaxed_f1 = (2 * relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall + epsilon)
    
    return {
        'relaxed_precision': relaxed_precision, 
        'relaxed_recall': relaxed_recall, 
        'relaxed_f1': relaxed_f1
    }