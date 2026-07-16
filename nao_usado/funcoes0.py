# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:53:52 2026

@author: Marcel
"""

# %% Import Libraries

import copy
import numpy as np
from skimage import morphology
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy



# %% Other approach

def ignore_small_areas1(array: np.ndarray, min_area_px: int, ignore_index: int = 255) -> np.ndarray:
    if len(array.shape) != 4 or array.shape[-1] != 1:
        raise ValueError('Array shape must be in [B, H, W, 1] format.')
    
    # Start with a clean copy of the original array to preserve 0s, 1s, and 255s perfectly
    cleaned_array = array.copy()
    
    # Pad size should ideally be a small buffer or based on sqrt(min_area) to save memory/time,
    # but keeping your pad_size logic here safely:
    pad_size = int(np.ceil(np.sqrt(min_area_px))) # Optimization: padding by sqrt is enough for edge effects
    pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    
    for b in range(array.shape[0]):
        img_slice = array[b, :, :, 0]
        
        # FIX 1: Isolate ONLY the objects (1s) into a binary mask. 
        # This completely ignores 0 and 255 background interference.
        object_mask = (img_slice == 1)
        
        # Pad the binary mask
        padded_slice = np.pad(object_mask, pad_width=pad_width, mode='symmetric')
        
        # Run area opening on the binary mask (removes small component islands of 1s)
        cleaned_slice = morphology.remove_small_objects(padded_slice, min_area_px, connectivity=1)
        cleaned_slice = cleaned_slice[pad_size:-pad_size, pad_size:-pad_size] # crop back
        
        # FIX 2: Now we can accurately find which pixels were flipped from 1 to 0
        removed_pixels = (object_mask == 1) & (cleaned_slice == 0)
        
        # FIX 3: Update the output array directly using the mask
        cleaned_array[b, :, :, 0][removed_pixels] = ignore_index
        
    return cleaned_array

# %% Function to remove small areas from an array

def remove_small_areas_generic(
    array: np.ndarray, 
    min_area_px: int, 
    ignore_index: int = 255, 
    background_index: int = 0
) -> np.ndarray:
    if len(array.shape) != 4 or array.shape[-1] != 1:
        raise ValueError('Array shape must be in [B, H, W, 1] format.')
    
    # Start with a clean copy of the original array.
    # This ensures ignore_index and all valid classes are preserved by default.
    cleaned_array = array.copy()
    
    # Find all unique classes that are NOT the background or the ignore index
    unique_classes = np.unique(array)
    unique_classes = unique_classes[(unique_classes != ignore_index) & (unique_classes != background_index)]
    
    for b in range(array.shape[0]):
        img_slice = array[b, :, :, 0]
        
        for cls in unique_classes:
            # 1. Isolate ONLY the pixels belonging to the current class
            class_mask = (img_slice == cls)
            
            # 2. Find connected components of this class and remove small ones
            cleaned_mask = morphology.area_opening(class_mask, min_area_px, connectivity=1)
            
            # 3. Identify exactly which pixels were removed
            removed_pixels = class_mask & ~cleaned_mask
            
            # 4. Safely turn ONLY those removed pixels into the designated background
            cleaned_array[b, :, :, 0][removed_pixels] = background_index
            
    return cleaned_array

# %% Function to merge dicts with preference to the second dict

def merge_dicts_with_preference(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dicts with preference to the second (dict2) in case of duplicate keys.
    If the duplicated key has inner dict as value, they will be merged maintaining the preference.
    """
    # Create a copy to not modify original data
    result = copy.deepcopy(dict1)
    
    # Loops through second dict to apply preference
    for key, value2 in dict2.items():
        # If the key exists in both dictionaries and the value of both is a dictionary, 
        # merge the contents inside
        if key in result and isinstance(result[key], dict) and isinstance(value2, dict):
            result[key] = result[key] | value2
        else:
            # Else, dict2 value overwrites dict1 value (or appends to it)
            result[key] = copy.deepcopy(value2)
            
    return result

# %% Masked Weighted Categorical Cross Entropy (Implementação IA Google)

def masked_weighted_cce(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # print('y_true', type(y_true), y_true)
        # print('y_pred', type(y_pred), y_pred)
        
        # Compute the standard cross loss (without reduction)
        cce_tf = CategoricalCrossentropy(reduction='none')
        cce_loss = cce_tf(y_true, y_pred)
        
        # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
        mask = tf.reduce_sum(y_true, axis=-1)
        mask = tf.cast(mask > 0, dtype=tf.float32)
        
        # Create the Weight Tensor
        weight_tensor = tf.reduce_sum(weights * y_true, axis=-1)
        
        # Weight the loss and mask it
        weighted_masked_loss = cce_loss * weight_tensor * mask
        
        # Reduce loss dividing by the sum of the (weight_tensor * mask) to keep the gradient consistent
        denominator = tf.reduce_sum(weight_tensor * mask) + tf.keras.backend.epsilon()
        
        # Return the division
        return tf.reduce_sum(weighted_masked_loss) / denominator

    return loss

# %% Masked Categorical Cross Entropy 

def masked_cce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # print('y_true', type(y_true), y_true)
    # print('y_pred', type(y_pred), y_pred)
    
    # Compute the standard cross loss (without reduction)
    cce_tf = CategoricalCrossentropy(reduction='none')
    cce_loss = cce_tf(y_true, y_pred)
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Mask the loss
    masked_loss = cce_loss * mask
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator



# %% Custom offset entropy loss

def custom_offset_entropy_loss(y_true, y_pred, entropy_weight):
    ''' Masked loss with L = CE * (1 + Entropy) ''' 
    # Squeeze and Cast entropy as float32
    entropy_weight = tf.cast(tf.squeeze(entropy_weight, axis=-1), tf.float32)
    
    # Compute the standard cross loss (without reduction)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Multiply by (1+entropy) and mask the loss
    masked_loss = cce_loss * (1 + entropy_weight)
    masked_loss = masked_loss * mask
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator


# %% Custom addictive entropy loss

def custom_add_entropy_loss(y_true, y_pred, entropy_weight):
    ''' Masked loss with L = CE + Entropy ''' 
    # Squeeze and Cast entropy as float32
    entropy_weight = tf.cast(tf.squeeze(entropy_weight, axis=-1), tf.float32)
    
    # Compute the standard cross loss (without reduction)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Compute the mask (0 is set to pixels with all 0s in all one-hot classes)
    mask = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(mask > 0, dtype=tf.float32)
    
    # Sum with entropy and mask the loss
    masked_loss = cce_loss + entropy_weight
    masked_loss = masked_loss * mask
    
    # Reduce loss dividing by the total number of non-ignored pixels
    denominator = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    
    # Return the division
    return tf.reduce_sum(masked_loss) / denominator