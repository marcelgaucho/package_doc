#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:19:29 2026

@author: rotunno
"""

# %% Import libraries

import numpy as np

# %% Function to calculate Expected Calibration Error (ECE) - Binary

def calculate_binary_ece(y_true, y_prob, n_bins=10, strategy='adaptive'):
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
        Number of bins to use.
    strategy : str, default='quantile'
        Strategy used to define the widths of the bins.
        'uniform' : Bins have equal widths.
        'adaptive' : Bins have equal number of samples.
        
    Returns:
    --------
    ece : float
        The Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Ensure we are looking at the probability of the positive class
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  # Take P(y=1)
        elif y_prob.shape[1] == 1:
            y_prob = y_prob.ravel()  # Flatten column vector
        else:
            raise ValueError(f"Unexpected 2D shape {y_prob.shape}. "
                             "Second dimension must be 1 or 2.")
    else:
        y_prob = y_prob.ravel()
        
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
        
    ece = 0.0

    # Handle distinct strategies
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

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
    
    elif strategy == 'adaptive':
        # Equal-mass binning: split sorted indices into N equal chunks
        sorted_indices = np.argsort(y_prob)
        chunks = np.array_split(sorted_indices, n_bins)
        
        for chunk_indices in chunks:
            # Safely skip empty chunks in extreme edge cases
            if len(chunk_indices) == 0:
                continue
                
            prop_in_bin = len(chunk_indices) / len(y_prob)
            fraction_positives = np.mean(y_true[chunk_indices])
            mean_predicted_prob = np.mean(y_prob[chunk_indices])
            
            ece += prop_in_bin * np.abs(fraction_positives - mean_predicted_prob)
    else:
        raise ValueError("Strategy must be either 'uniform' or 'adaptive'.")
            
    return ece

# %%

if __name__ == '__main__':
    # 5 samples, values represent the probability of class 1
    true_labels_bin = [1, 0, 1, 1, 0]
    pred_probs_bin = [0.9, 0.1, 0.8, 0.4, 0.2]
    
    # %%
    ece_bin = calculate_binary_ece(true_labels_bin, pred_probs_bin, n_bins=4, strategy='adaptive')
    print(f"Binary ECE: {ece_bin:.4f}")




