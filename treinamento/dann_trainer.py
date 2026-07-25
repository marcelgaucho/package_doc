# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:50:27 2026

@author: Marcel
"""

# %% Import Libraries

from pathlib import Path
import tensorflow as tf
from .model_trainer import ModelTrainer
from .utils import parse_and_normalize, transform_augment_batch


# %%

class DomainAdaptationTrainer(ModelTrainer):
    def __init__(self, x_dir: str, target_dir: str, output_dir: str, optimizer, model=None):
        # Initialize the base class properties
        super().__init__(x_dir, output_dir, optimizer, model)
        self.target_dir = Path(target_dir) # Store the new unlabeled target domain path

    def _prepare_datasets(self, uncertainty_dir, batch_size, buffer_shuffle, uncertainty_metric, data_augmentation):
        # 1. Load the labeled Source Data using the parent's pristine logic
        source_train_ds, valid_ds = super()._prepare_datasets(
            uncertainty_dir, batch_size, buffer_shuffle, uncertainty_metric, data_augmentation
        )

        # 2. Load the unlabeled Target Data
        target_train_ds = tf.data.Dataset.load(str(self.target_dir / 'train_dataset/'))
        
        # 3. Apply the same batching/augmentation to the Target Data
        shuffle_target = buffer_shuffle or len(target_train_ds)
        target_train_ds = target_train_ds.shuffle(shuffle_target).map(
            parse_and_normalize, num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if data_augmentation:
            target_train_ds = target_train_ds.batch(batch_size).map(
                transform_augment_batch, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            target_train_ds = target_train_ds.batch(batch_size)
            
        target_train_ds = target_train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # 4. Zip them together for the DANN forward pass 
        # This will yield batches in the format: ((source_x, source_y), target_x)
        combined_train_ds = tf.data.Dataset.zip((source_train_ds, target_train_ds))

        return combined_train_ds, valid_ds
        
    def train_with_loop(self, **kwargs):
        # Optional: You can override this slightly to automatically inject 
        # the DANN step function (shown below) into your train_model_loop.
        kwargs['step_fn'] = dann_train_step
        return super().train_with_loop(**kwargs)