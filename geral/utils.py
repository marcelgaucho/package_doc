# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:28:19 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
import os
import copy

# %% Setup hardware (GPU RAM limit or CPU threads)

def setup_hardware(use_gpu: bool = True, cpu_threads: int = 1, gpu_memory_limit: int = None):
    """Centralizes GPU/CPU configuration."""
    gpus = tf.config.list_physical_devices('GPU')
    if use_gpu and gpus: # Setup GPU memory to use
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit)]
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Set CPU threads
    tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    
# %% Merge dicts

def merge_dicts_with_preference(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dicts with preference to the second (dict2) in case of duplicate keys.
    If the duplicated key has an inner dict as value, they will be deeply merged.
    """
    result = copy.deepcopy(dict1)
    
    for key, value2 in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value2, dict):
            # Recursively merge deeper levels instead of using the shallow | operator
            result[key] = merge_dicts_with_preference(result[key], value2)
        else:
            result[key] = copy.deepcopy(value2)
            
    return result