# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:15:30 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np
import warnings

# %%

class MidpointPatchMerger:
    def __init__(self, target_shape, patch_size, overlap_percent):
        self.h, self.w, self.c = target_shape
        self.patch_size = patch_size
        self.overlap_px = int(patch_size * overlap_percent)
        
        # Pre-initialize accumulators
        self.img_acc = np.zeros(target_shape, dtype=np.float32)
        self.weight_acc = np.zeros(target_shape, dtype=np.float32)
        
        # Pre-calculate a base mask to save memory allocations in the loop
        self._base_mask = np.ones((patch_size, patch_size, 1), dtype=np.float32)

    def __call__(self, patches, coords, dtype=np.uint8):
        # Global coordinate bounds
        min_y, min_x = np.min(coords[:, 0]), np.min(coords[:, 1])
        max_y, max_x = np.max(coords[:, 2]), np.max(coords[:, 3])
        
        half_ov = self.overlap_px // 2
        rem_ov = self.overlap_px - half_ov

        for patch, (ymin, xmin, ymax, xmax) in zip(patches, coords):
            # 1. Generate mask on the fly and handle overlaps
            mask = self._base_mask.copy()
            if ymin > min_y: mask[:half_ov, :] = 0 # Top overlap
            if xmin > min_x: mask[:, :half_ov] = 0 # Left overlap
            if ymax < max_y: mask[-rem_ov:, :] = 0 # Bottom overlap 
            if xmax < max_x: mask[:, -rem_ov:] = 0 # Right overlap

            # 2. Map coordinates to target image bounds
            img_ymin, img_xmin = max(0, ymin), max(0, xmin) # Image starts in (0, 0)
            img_ymax, img_xmax = min(self.h, ymax), min(self.w, xmax) # Image ends in (h, w)
            
            # 3. Slice the patch/mask (for image clipping)
            p_ymin, p_xmin = img_ymin - ymin, img_xmin - xmin
            p_ymax, p_xmax = p_ymin + (img_ymax - img_ymin), p_xmin + (img_xmax - img_xmin)
            
            target_mask = mask[p_ymin:p_ymax, p_xmin:p_xmax]
            
            # 4. Accumulate
            self.img_acc[img_ymin:img_ymax, img_xmin:img_xmax] += patch[p_ymin:p_ymax, p_xmin:p_xmax] * target_mask
            self.weight_acc[img_ymin:img_ymax, img_xmin:img_xmax] += target_mask

        # 5. Finalize & Validate
        if np.any(self.weight_acc == 0):
            raise ValueError("Reconstruction failed: Gap detected in patch coverage.")
        
        if np.any(self.weight_acc > 1):
            warnings.warn("Overlap detected: some pixels merged from multiple patches.", UserWarning)

        # Average and cast (for safety)
        return np.divide(self.img_acc, self.weight_acc, 
                         out=np.zeros_like(self.img_acc), 
                         where=self.weight_acc != 0).astype(dtype)