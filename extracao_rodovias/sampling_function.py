# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:10:16 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np
from typing import Optional

# %% Function to get stratified subsample indices

def get_stratified_subsample_indices(y_data: np.ndarray, max_patches: int, target_class: int = 1, num_bins: int = 10, seed: Optional[int] = 42):
    """
    Returns indices for a stratified subsample based on the target-class pixel percentage of each patch.

    Stratification uses quantile-based bins, so each bin contains approximately the same number of patches when possible. Duplicate
    quantile boundaries are removed, which is important for datasets with many patches having identical target-class ratios (e.g., 0%).

    Samples are allocated proportionally to the original bin sizes, preserving the empirical distribution of target-class pixel ratios.

    Args:
        y_data: Array of shape (N, H, W, C) for one-hot labels or shape (N, H, W) for class-index labels.
        max_patches: Maximum number of patches to sample.
        target_class: Class used to calculate the target-pixel ratio.
        num_bins: Maximum number of quantile bins.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray:
            Indices of the selected patches.
    """

    n_patches = len(y_data)

    # Data validation and seed for reproducibility
    if max_patches >= n_patches:
        return np.arange(n_patches)

    if max_patches <= 0:
        raise ValueError("max_patches must be greater than 0.")

    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")

    rng = np.random.default_rng(seed)

    # 1. Calculate target-class pixel ratio for each patch
    pixels_per_patch = y_data.shape[1] * y_data.shape[2]

    if y_data.ndim == 4:
        # One-hot labels: (N, H, W, C) -> slice the target channel
        target_counts = np.sum(y_data[..., target_class], axis=(1, 2))
    elif y_data.ndim == 3:
        # Integer class-indices: (N, H, W) -> check equality
        target_counts = np.sum(y_data == target_class, axis=(1, 2))
    else:
        raise ValueError(f"Unsupported y_data dimension: {y_data.ndim}. Expected 3D or 4D array.")

    target_ratios = target_counts / pixels_per_patch

    # 2. Create quantile-based strata
    quantile_levels = np.linspace(0, 100, num_bins + 1)

    quantile_edges = np.percentile(target_ratios, quantile_levels)

    # Remove duplicate edges. Important when many patches have the same target ratio (e.g., 0%-road patches)
    bins = np.unique(quantile_edges)

    if len(bins) == 1:
        # All patches have exactly the same target ratio.
        bin_indices = np.zeros(n_patches, dtype=int)
    else:
        # Internal edges only. Using bins[1:-1] avoids problems with values exactly equal to the minimum or maximum.
        bin_indices = np.digitize(target_ratios, bins[1:-1], right=False)

    # 3. Determine original distribution of the strata
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)

    # 4. Proportional allocation using the largest-remainder method
    exact_samples = (bin_counts / n_patches) * max_patches

    # Initial allocation
    samples_per_bin = np.floor(exact_samples).astype(int)

    # Number of samples still to distribute
    remaining = (max_patches - np.sum(samples_per_bin))

    if remaining > 0:
        # Fractional remainder of each bin
        fractional_parts = (exact_samples - samples_per_bin)

        # Bins with the largest fractional remainder receive the remaining samples.
        order = np.argsort(fractional_parts)[::-1]

        for i in range(remaining):
            samples_per_bin[order[i]] += 1

    # 5. Randomly sample within each stratum
    selected_indices = []

    for i, bin_idx in enumerate(unique_bins):
        indices_in_bin = np.flatnonzero(bin_indices == bin_idx)

        n_samples = samples_per_bin[i]

        if n_samples > 0:
            sampled = rng.choice(indices_in_bin, size=n_samples, replace=False)

            selected_indices.extend(sampled)

    # 6. Shuffle all the final selected indices
    selected_indices = np.asarray(selected_indices, dtype=int)

    rng.shuffle(selected_indices) 

    return selected_indices

# %% Function Test with Sample Data


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Tiny Debug Dataset: 5 patches, shape (5, 2, 2) -> (B, H, W)
    # -------------------------------------------------------------------------
    # Integer class indices: 0 = background, 1 = target class (road)
    # Pixels per patch = H * W = 2 * 2 = 4 pixels
    #
    # Patch 0: 0/4 target pixels (0.0% ratio) -> All background
    # Patch 1: 0/4 target pixels (0.0% ratio) -> All background (creates zero-inflation)
    # Patch 2: 1/4 target pixels (25.0% ratio) -> Sparse target
    # Patch 3: 2/4 target pixels (50.0% ratio) -> Medium target
    # Patch 4: 4/4 target pixels (100.0% ratio) -> Full target
    
    y_data = np.array([
        # Patch 0 (0.0%)
        [[0, 0],
         [0, 0]],
        
        # Patch 1 (0.0%)
        [[0, 0],
         [0, 0]],
        
        # Patch 2 (25.0%)
        [[1, 0],
         [0, 0]],
        
        # Patch 3 (50.0%)
        [[1, 1],
         [0, 0]],
        
        # Patch 4 (100.0%)
        [[1, 1],
         [1, 1]]
    ], dtype=np.int64)  # Shape: (5, 2, 2)
    
    print(f"y_data shape: {y_data.shape} (Batch, Height, Width)")
    
    # -------------------------------------------------------------------------
    # Run with a small max_patches and num_bins to watch the mechanics
    # -------------------------------------------------------------------------
    selected_indices = get_stratified_subsample_indices(
        y_data=y_data,
        max_patches=3,     # Subsample 3 out of 5 patches
        target_class=1,    # Target class to measure
        num_bins=3,        # Divide the data in a number of bins
        seed=42
    )
    
    print("Original dataset size:", len(y_data))
    print("Selected indices:", selected_indices)


