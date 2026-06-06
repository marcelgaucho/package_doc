# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:28:19 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
import os

# %% Setup hardware (GPU RAM limit or CPU threads)

def setup_hardware(cpu_threads: int = 1, gpu_memory_limit: int = None):
    """Centralizes GPU/CPU configuration."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and gpu_memory_limit:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit)]
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)