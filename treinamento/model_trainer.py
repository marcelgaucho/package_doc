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
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from .metrics import CustomF1Score
from .lr_decay import StepDecay
from .training_loop import train_model_loop
from .utils import show_training_plot
from .fine_tuning import FineTuneStrategy

# %%

class ModelTrainer:
    best_model_filename = 'best_model.keras'
    early_stopping_delta = 0.001

    def __init__(self, x_dir: str, output_dir: str, model, optimizer):
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
                        buffer_shuffle=None, batch_size=16, data_augmentation=False,
                        mode='max', augment_batch_factor=2, step_decay=False, reduce_on_plateau=True,
                        entropy_dir=None):
        
        # Handle mutable defaults safely
        metrics_train = metrics_train or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        metrics_val = metrics_val or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        loss_fn = loss_fn or tf.keras.losses.CategoricalCrossentropy(from_logits=False)

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
        train_dataset, valid_dataset = self._prepare_datasets(entropy_dir, batch_size, buffer_shuffle)

        # 2. Pipeline Stage: Configure Schedulers and Optimizer
        optimizer = self._configure_optimizer(model, learning_rate, step_decay, reduce_on_plateau, train_dataset, batch_size)

        # 3. Pipeline Stage: Fire Custom Loop Execution Engine
        results = train_model_loop(
            model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
            train_dataset=train_dataset, valid_dataset=valid_dataset,
            optimizer=optimizer, loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val,
            model_path=self.model_path, early_stopping_delta=self.early_stopping_delta,
            data_augmentation=data_augmentation, reduce_on_plateau=reduce_on_plateau,
            mode=mode, augment_batch_factor=augment_batch_factor
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
    
    def fine_tune(self, model_path, strategy: FineTuneStrategy, epochs=1000, early_stopping_epochs=30,
                  metrics_train=None, metrics_val=None, loss_fn=None,
                  buffer_shuffle=None, batch_size=16, data_augmentation=False,
                  mode='max', augment_batch_factor=2, step_decay=False, reduce_on_plateau=True,
                  entropy_dir=None):
        """Phase 2 execution using an interchangeable FineTuneStrategy configuration."""
        
        metrics_train = metrics_train or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        metrics_val = metrics_val or [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
        loss_fn = loss_fn or tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        dict_parameters = locals().copy()
        del dict_parameters['self']
        dict_parameters['strategy'] = strategy.__class__.__name__

        start_time = time.time()

        # 1. Load the actual trained model weights from Phase 1 instead of cloning an empty skeleton
        print(f"Loading best baseline model weights from {model_path}...")
        model = tf.keras.models.load_model(model_path)

        # 2. Apply Strategy to alter layer states (freezing/unfreezing logic)
        model = strategy.apply(model)

        if data_augmentation:
            if batch_size % augment_batch_factor != 0: 
                raise ValueError(f"Batch size ({batch_size}) must be perfectly divisible by augment_batch_factor ({augment_batch_factor}).")
            batch_size //= augment_batch_factor

        # 3. Reuse your clean internal pipeline helpers
        train_dataset, valid_dataset = self._prepare_datasets(entropy_dir, batch_size, buffer_shuffle)
        
        # 4. Configure optimizer using strategy's custom micro-learning rate
        optimizer = self._configure_optimizer(
            model, strategy.learning_rate, step_decay, reduce_on_plateau, train_dataset, batch_size
        )

        # 5. Fire training engine loop for fine-tuning
        print(f"Beginning fine-tuning loop with learning rate: {strategy.learning_rate}")
        results = train_model_loop(
            model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
            train_dataset=train_dataset, valid_dataset=valid_dataset,
            optimizer=optimizer, loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val,
            model_path=self.model_path, early_stopping_delta=self.early_stopping_delta,
            data_augmentation=data_augmentation, reduce_on_plateau=reduce_on_plateau,
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

    def _prepare_datasets(self, entropy_dir, batch_size, buffer_shuffle):
        """Internal helper to load, combine weights, shuffle, and batch tf.data pipelines."""
        train_ds = tf.data.Dataset.load(str(self.x_dir / 'train_dataset/'))
        valid_ds = tf.data.Dataset.load(str(self.x_dir / 'valid_dataset/'))

        if entropy_dir:
            entropy_train = np.load(Path(entropy_dir) / 'entropy_train.npy')
            entropy_valid = np.load(Path(entropy_dir) / 'entropy_valid.npy')
            
            weights_train = tf.data.Dataset.from_tensor_slices(entropy_train)
            weights_valid = tf.data.Dataset.from_tensor_slices(entropy_valid)
            
            train_ds = tf.data.Dataset.zip((train_ds, weights_train)).map(lambda xy, w: (xy[0], xy[1], w))
            valid_ds = tf.data.Dataset.zip((valid_ds, weights_valid)).map(lambda xy, w: (xy[0], xy[1], w))

        # Dynamic fallback configurations for dataset shuffle buffers
        shuffle_train = buffer_shuffle or len(train_ds)
        shuffle_valid = buffer_shuffle or len(valid_ds)

        return (
            train_ds.shuffle(shuffle_train).batch(batch_size),
            valid_ds.shuffle(shuffle_valid).batch(batch_size)
        )

    def _configure_optimizer(self, model, learning_rate, step_decay, reduce_on_plateau, train_dataset, batch_size):
        """Internal helper to build optimizer variables and assign learning rate routines."""
        optimizer = self.optimizer
        optimizer.build(model.trainable_variables)

        if step_decay and reduce_on_plateau:
            raise ValueError('Only one decay parameter can be True, except if all decay parameters are False')
            
        if step_decay:
            steps_per_epoch = train_dataset.cardinality().numpy() # dataset is already batched, so cardinality == total number of batches
            optimizer.learning_rate = StepDecay(
                initial_lr=learning_rate, steps_per_epoch=steps_per_epoch,
                drop_rate=0.1, epochs_per_drop=10
            )
        else:
            optimizer.learning_rate.assign(learning_rate)
            
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
            


