# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:19:25 2026

@author: Marcel
"""

# %% Import Libraries

import numpy as np
import time
import pickle
import types
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from .metrics import CustomF1Score
from .lr_decay import StepDecay, ConstantLR, LRStrategy
from .training_loop import train_model_loop
# from .training_loop_uce import train_model_loop
from .utils import show_training_plot, parse_and_normalize
from .fine_tuning import FineTuneStrategy
from ..entropy.utils import UncertaintyMetric #, uncert_metric_method

# %%

class ModelTrainer:
    best_model_filename = 'best_model.keras'
    early_stopping_delta = 0.001

    def __init__(self, x_dir: str, output_dir: str, optimizer, model=None):
        # Configuration Paths 
        self.x_dir = Path(x_dir)
        self.output_dir = Path(output_dir)
        self.model_path = self.output_dir / self.best_model_filename
        
        # Core Components
        self.model = model
        self.optimizer = optimizer
        
        # Track absolute execution performance
        self.best_loss = None
        self.best_metric = None

    def train_with_loop(self, epochs=2000, early_stopping_epochs=50, 
                        metrics_train=None, metrics_val=None,
                        learning_rate=0.001, loss_fn=None,
                        buffer_shuffle=None, batch_size=16, data_augmentation=True,
                        mode='max', augment_batch_factor=2, lr_strategy: LRStrategy=None, 
                        uncertainty_dir=None, uncertainty_metric=UncertaintyMetric.Entropy):
        
        # Handle mutable defaults safely
        metrics_train = metrics_train or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        metrics_val = metrics_val or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        loss_fn = loss_fn or tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        lr_strategy = lr_strategy or ConstantLR() # Constant LR if no strategy is provided
        
        # Snapshot execution configuration parameters for logging
        dict_parameters = locals().copy()
        del dict_parameters['self']

        # Clone the base model template and track time
        model = tf.keras.models.clone_model(self.model)
        start_time = time.time()
        
        # Scale batch size down for data augmentation factoring
        if data_augmentation:
            if batch_size % augment_batch_factor != 0: 
                raise ValueError(f"Batch size ({batch_size}) must be perfectly divisible by augment_batch_factor ({augment_batch_factor}).")
            batch_size //= augment_batch_factor

        # 1. Pipeline Stage: Build and structure Datasets
        train_dataset, valid_dataset = self._prepare_datasets(uncertainty_dir, batch_size, buffer_shuffle,
                                                              uncertainty_metric)

        # 2. Pipeline Stage: Configure Schedulers and Optimizer
        optimizer = self._configure_optimizer(model, learning_rate, lr_strategy, train_dataset, batch_size)

        # 3. Pipeline Stage: Fire Custom Loop Execution Engine
        results = train_model_loop(
            model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
            train_dataset=train_dataset, valid_dataset=valid_dataset,
            optimizer=optimizer, loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val,
            model_path=self.model_path, early_stopping_delta=self.early_stopping_delta,
            data_augmentation=data_augmentation, lr_strategy=lr_strategy,  
            mode=mode, augment_batch_factor=augment_batch_factor,
        )
        
        # Support both the legacy raw list output and the modern structured dictionary output
        if isinstance(results, dict):
            result_history = [results["history_train"], results["history_valid"]]
            self.best_loss = results.get("best_loss")
            self.best_metric = results.get("best_metric")
        else:
            result_history = results

        # 4. Pipeline Stage: Persist Artifacts and Visualization Plots
        duration = time.time() - start_time
        self._save_training_artifacts(model, result_history, duration, dict_parameters, metrics_val)
        
        return result_history
    
    def fine_tune(self, base_model_path, strategy: FineTuneStrategy, epochs=1000, early_stopping_epochs=30,
                  metrics_train=None, metrics_val=None, loss_fn=None,
                  buffer_shuffle=None, batch_size=16, data_augmentation=False,
                  mode='max', augment_batch_factor=2, lr_strategy: LRStrategy=None,
                  uncertainty_dir=None, uncertainty_metric=UncertaintyMetric.Entropy):
        """Phase 2 execution using an interchangeable FineTuneStrategy configuration."""
        
        metrics_train = metrics_train or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        metrics_val = metrics_val or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        loss_fn = loss_fn or tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        lr_strategy = lr_strategy or ConstantLR()

        dict_parameters = locals().copy()
        del dict_parameters['self']
        dict_parameters['strategy'] = strategy.__class__.__name__

        start_time = time.time()

        # 1. Load the actual trained model weights from Phase 1 instead of cloning an empty skeleton
        print(f"Loading best baseline model weights from {base_model_path}...")
        model = tf.keras.models.load_model(base_model_path)

        # 2. Apply Strategy to alter layer states (freezing/unfreezing logic)
        model = strategy.apply(model)

        if data_augmentation:
            if batch_size % augment_batch_factor != 0: 
                raise ValueError(f"Batch size ({batch_size}) must be perfectly divisible by augment_batch_factor ({augment_batch_factor}).")
            batch_size //= augment_batch_factor

        # 3. Reuse your clean internal pipeline helpers
        train_dataset, valid_dataset = self._prepare_datasets(uncertainty_dir, batch_size, buffer_shuffle,
                                                              uncertainty_metric)
        
        # 4. Configure optimizer using strategy's custom micro-learning rate
        optimizer = self._configure_optimizer(
            model, strategy.learning_rate, lr_strategy, train_dataset, batch_size
        )

        # 5. Fire training engine loop for fine-tuning
        print(f"Beginning fine-tuning loop with learning rate: {strategy.learning_rate}")
        results = train_model_loop(
            model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
            train_dataset=train_dataset, valid_dataset=valid_dataset,
            optimizer=optimizer, loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val,
            model_path=self.model_path, early_stopping_delta=self.early_stopping_delta,
            data_augmentation=data_augmentation, lr_strategy=lr_strategy,  
            mode=mode, augment_batch_factor=augment_batch_factor            
        )
        
        if isinstance(results, dict):
            result_history = [results["history_train"], results["history_valid"]]
            self.best_loss = results.get("best_loss")
            self.best_metric = results.get("best_metric")
        else:
            result_history = results

        duration = time.time() - start_time
        self._save_training_artifacts(model, result_history, duration, dict_parameters, metrics_val)
        
        return result_history

    def _prepare_datasets(self, uncertainty_dir, batch_size, buffer_shuffle, uncertainty_metric):
        """Internal helper to load, combine weights, shuffle, and batch tf.data pipelines."""
        train_ds = tf.data.Dataset.load(str(self.x_dir / 'train_dataset/'))
        valid_ds = tf.data.Dataset.load(str(self.x_dir / 'valid_dataset/'))

        if uncertainty_dir:
            uncertainty_train = np.load(Path(uncertainty_dir) / f'{uncertainty_metric.value}_train.npy')
            uncertainty_valid = np.load(Path(uncertainty_dir) / f'{uncertainty_metric.value}_valid.npy')
            
            weights_train = tf.data.Dataset.from_tensor_slices(uncertainty_train)
            weights_valid = tf.data.Dataset.from_tensor_slices(uncertainty_valid)
            
            train_ds = tf.data.Dataset.zip((train_ds, weights_train)).map(lambda xy, w: (xy[0], xy[1], w))
            valid_ds = tf.data.Dataset.zip((valid_ds, weights_valid)).map(lambda xy, w: (xy[0], xy[1], w))

        # Dynamic fallback configurations for dataset shuffle buffers
        shuffle_train = buffer_shuffle or len(train_ds)
        
        return (
        train_ds.shuffle(shuffle_train)
                .map(parse_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE),
                
        valid_ds.map(parse_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def _configure_optimizer(self, model, learning_rate, lr_strategy, train_dataset, batch_size):
        """Internal helper to build optimizer variables and assign learning rate routines."""
        optimizer = self.optimizer
        optimizer.build(model.trainable_variables)
        steps_per_epoch = train_dataset.cardinality().numpy() # cardinality == total number of batches (dataset is already batched)

        # Strategy pattern execution
        lr_strategy.setup(optimizer, steps_per_epoch, learning_rate)
            
        return optimizer

    def _get_metric_name(self, metric):
        """Safely parses type declarations to isolate explicit execution names."""
        if isinstance(metric, str):
            return metric
        if hasattr(metric, '__class__') and not isinstance(metric, (types.FunctionType, types.BuiltinFunctionType)):
            return metric.__class__.__name__.lower()
        if isinstance(metric, (types.FunctionType, types.BuiltinFunctionType)):
            return metric.__name__
        return "metric"

    def _save_training_artifacts(self, model, result_history, duration, parameters, metrics_val):
        """Handles file serialization, directory logging, and plot exports cleanly."""
        # Save structural history outputs
        with open(self.output_dir / 'history_best_model.txt', 'w') as f:
            f.write('Resultado = \n')
            f.write(str(result_history))
            f.write(f'\nTempo total gasto no treinamento foi de {duration:.2f} segundos, {duration/3600:.1f} horas.')
            f.write(f'\n[Melhor Época] Melhor Loss: {self.best_loss:.4f}')
            f.write(f'\n[Melhor Época] Melhor Métrica: {self.best_metric:.4f}')
            
        with open(self.output_dir / 'history_pickle_best_model.pickle', "wb") as fp: 
            pickle.dump(result_history, fp)
            
        # Parse metric names and render evaluation charts
        metric_name = self._get_metric_name(metrics_val[0])
        show_training_plot(result_history, metric_name=metric_name, save=True, save_path=self.output_dir)
        
        # Save underlying network structural data configurations
        with open(self.output_dir / 'model_configuration_used.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(str(parameters) + '\n')
            


