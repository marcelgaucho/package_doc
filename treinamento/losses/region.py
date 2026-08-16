# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:39:45 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf

# %% Generalized Dice Loss

def get_generalized_dice_loss(use_mask=True):
    """
    Factory function that returns a Generalized Dice Loss (Sudre et al., 2017).

    This loss dynamically weights classes by the inverse square of their volume,
    to handle class imbalance.

    Args:
        use_mask (bool): If True, ignores spatial locations where y_true is all zeros.
    """

    def loss(y_true, y_pred):
        # Smooth constant added to avoid zero division
        smooth = 1e-5

        # 1. Ensure float tensors
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 2. Apply mask
        if use_mask:
            mask = tf.cast(
                tf.reduce_sum(y_true, axis=-1, keepdims=True) > 0,
                tf.float32
            )

            y_true *= mask
            y_pred *= mask

        # 3. Get global class totals (Num_classes,), reducing over other dimensions
        rank = tf.rank(y_true)
        reduce_axes = tf.range(rank - 1)

        intersection = tf.reduce_sum(y_true * y_pred, axis=reduce_axes)
        true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
        pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)

        # 4. Calculate Generalized Dice weights for the classes (w_c = 1 / |G_c|²)
        # Classes absent from the batch receive zero weight
        weights = tf.where(
            true_sum > 0.0,
            tf.math.reciprocal(tf.square(true_sum)),
            tf.zeros_like(true_sum),
        )
        
        # 5. Apply weights to the numerator and denominator across all classes
        numerator = tf.reduce_sum(weights * intersection)
        denominator = tf.reduce_sum(weights * (true_sum + pred_sum))

        # 6. Calculate final Generalized Dice score
        dice = (2.0 * numerator + smooth) / (denominator + smooth)

        return 1.0 - dice

    return loss

