#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:19:29 2026

@author: rotunno
"""

# %% Import libraries

import numpy as np

# %% Function to calculate Expected Calibration Error (ECE)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE) for classification.
    
    Parameters:
    y_true (array-like): Ground truth (correct) target labels (0 to n_classes-1).
    y_prob (array-like): Predicted probabilities of shape (n_samples, n_classes) 
                         or (n_samples,) for binary classification.
    n_bins (int): Number of bins to partition the confidence space.
    
    Returns:
    float: The calculated Expected Calibration Error.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Handle multiclass vs binary probability arrays
    # With top-label ECE
    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        # Top-label calibration: get the highest probability and its class index
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        # Binary: probability of the *predicted* class
        # If P(y=1) is 0.2, the confidence in the prediction (0) is 0.8
        y_prob = y_prob.flatten()
        confidences = np.maximum(y_prob, 1 - y_prob)
        predictions = (y_prob >= 0.5).astype(int)
        
    # Accuracy per sample (1 if correct, 0 otherwise)
    accuracies = (predictions == y_true)
    
    # Create equally spaced bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    # Calculate error contribution from each bin
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find which samples fall into the current bin range [bin_lower, bin_upper)
        # Include upper boundary in the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prop_in_bin = np.mean(in_bin)  # Fraction of total samples in this bin
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin]) # accuracy fraction inside the bin
            confidence_in_bin = np.mean(confidences[in_bin]) # mean confidence inside the bin
            
            # Weights absolute difference between confidence and accuracy by fraction of total samples in bin
            ece += prop_in_bin * np.abs(confidence_in_bin - accuracy_in_bin)
            
    return ece