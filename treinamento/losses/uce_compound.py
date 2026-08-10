#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:31:27 2026

@author: rotunno
"""

# %% Import Libraries

import tensorflow as tf
from .uce_distribution import get_u_categorical_crossentropy
from .uce_region import get_u_dice_loss

# %% Combo-UCE Loss: U-CE + U-Dice

def get_u_combo_loss(
    loss_weights=(0.5, 0.5),
    uce_class_weights=None,
    udice_class_weights=(0., 1.),
    use_mask=True,
    alpha=1.0,
    downweight_uncertainty=False,
    return_components=False
):
    """
    Factory function for an Uncertainty-Aware Combo Loss combining 
    Categorical Cross-Entropy (U-CE) and Masked Dice Loss (U-Dice).

    Args:
        loss_weights (list or tuple): A two-element array containing the multiplier weights 
                                      for the losses U-CE and U-Dice, respectively 
                                      (e.g., [0.5, 0.5] for the average between the two loss components).
        uce_class_weights (list or tuple): U-CE class weights (e.g., None or [1, 1] for equal weights).
        udice_class_weights (list or tuple): U-Dice class weights (e.g., [0.0, 1.0] for only foreground).
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
        alpha (float): Hyperparameter controlling the intensity of the U-CE penalty.
        downweight_uncertainty (bool): Passed to Dice loss (inverse weighting option).
        return_components (bool): If True, returns (total_loss, ce_loss, dice_loss) 
                                   for detailed metric tracking/logging.

    Returns:
        function: Loss function accepting (y_true, y_pred, sigma=None).
    """
    # Unpack loss weights into their components
    weight_ce, weight_dice = loss_weights
    
    # Instantiate the two component loss functions
    u_ce_loss_fn = get_u_categorical_crossentropy(
        class_weights=uce_class_weights,
        use_mask=use_mask,
        alpha=alpha
    )
    
    u_dice_loss_fn = get_u_dice_loss(
        class_weights=udice_class_weights,
        use_mask=use_mask,
        alpha=alpha,
        downweight_uncertainty=downweight_uncertainty
    )

    def loss(y_true, y_pred, sigma=None):
        # 1. Compute individual uncertainty-aware loss components
        ce_loss = u_ce_loss_fn(y_true, y_pred, sigma=sigma)
        dice_loss = u_dice_loss_fn(y_true, y_pred, sigma=sigma)
        
        # 2. Combine into a single weighted scalar loss
        total_loss = (weight_ce * ce_loss) + (weight_dice * dice_loss)
        
        # Option to return individual components for training loop metric tracking
        if return_components:
            return total_loss, ce_loss, dice_loss
            
        return total_loss

    return loss
