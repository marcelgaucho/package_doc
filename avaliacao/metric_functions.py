#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:19:29 2026

@author: rotunno
"""

# %% Import libraries

import numpy as np

# %% Function to calculate Expected Calibration Error (ECE) - Binary

def binary_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the Binary Expected Calibration Error (Binary ECE) for classification.
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Ground truth integer labels (0 or 1).
    y_prob : array-like
        Predicted probabilities of the positive class (class 1).
        Can be a 1D array of shape (n_samples,) or a 2D array of 
        shape (n_samples, 2) where the second column is used.
    n_bins : int, default=10
        Number of uniformly spaced bins between 0 and 1.
        
    Returns:
    --------
    ece : float
        The Marginal Expected Calibration Error.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # 1. Ensure we are looking at the probability of the positive class
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            # Take P(y=1) from the second column
            y_prob = y_prob[:, 1] 
        else:
            raise ValueError("This function is a Binary ECE implementation. "         
                             "If the array has two dimensions, the "
                             "last dimension must be at most 2.") 
    else:
        y_prob = y_prob.flatten()

    # 2. Define bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    # 3. Iterate over bins and compute the weighted absolute error
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Assign samples to the bin. Include upper boundary in the last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        prop_in_bin = np.mean(in_bin) # Fraction of total samples in this bin
        
        # Only calculate for non-empty bins
        if prop_in_bin > 0:
            fraction_positives = np.mean(y_true[in_bin]) # Actual fraction of positives in this bin
            mean_predicted_prob = np.mean(y_prob[in_bin]) # mean probability inside the bin

            # Weights absolute difference between probability mean and positive fraction by 
            # fraction of total samples in the bin
            ece += prop_in_bin * np.abs(fraction_positives - mean_predicted_prob)
            
    return ece

# %%

# 5 samples, values represent the probability of class 1
true_labels_bin = [1, 0, 1, 1, 0]
pred_probs_bin = [0.9, 0.1, 0.8, 0.4, 0.2]

ece_bin = binary_expected_calibration_error(true_labels_bin, pred_probs_bin, n_bins=5)
print(f"Binary ECE: {ece_bin:.4f}")




