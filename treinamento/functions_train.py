# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:25:39 2024

@author: Marcel
"""

''' Imports '''
import numpy as np, os, shutil, math, pickle, time, types, gc

# Import tensorflow related
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy

# Import from other modules inside package
from .f1_metric import F1Score, RelaxedF1Score
from .plot_training import show_training_plot
from .arquiteturas.unetr_2d import config_dict
print(config_dict)
from .train_loop_functions import train_model_loop
from .augment_functions import transform_augment_or_maintain        


class ModelTrainer:
    best_model_filename = 'best_model.keras'
    early_stopping_delta = 0.01 # Delta in relation to best result for training to continue 
    def __init__(self, x_dir: str, y_dir: str, output_dir: str, model, optimizer):
        # Directories
        self.x_dir = x_dir # Dir with X data
        self.y_dir = y_dir # Dir with Y data
        self.output_dir = output_dir # Dir to save Output data
        
        self.model = model # Model object
        
        self.optimizer = optimizer # Optimizer, has to be created outside class in order to be a singleton
        
        self.model_path = output_dir + self.best_model_filename # Path to save model
        
    def _set_datasets(self):
        # Load Datasets
        self.train_dataset = tf.data.Dataset.load(self.x_dir + 'train_dataset/')
        self.valid_dataset = tf.data.Dataset.load(self.x_dir + 'valid_dataset/')
        
    def _set_numpy_arrays(self, convert_to_tensor=False):
        # Load Y Train, do One-Hot encoding if necessary. If specified, convert to tensor and clean memory 
        self.y_train = np.load(self.y_dir + 'y_train.npy')        
        if self.y_train.shape[-1] == 1:
            self.y_train = to_categorical(self.y_train, num_classes=2)
        if convert_to_tensor:    
            with tf.device('/CPU:0'):
                self.y_train = tf.convert_to_tensor(self.y_train)
            gc.collect()
        
        # Load Y Valid
        self.y_valid = np.load(self.y_dir + 'y_valid.npy')
        if self.y_valid.shape[-1] == 1:
            self.y_valid = to_categorical(self.y_valid, num_classes=2)
        if convert_to_tensor:    
            with tf.device('/CPU:0'):
                self.y_valid = tf.convert_to_tensor(self.y_valid)
            gc.collect()
        
        # Load X Train
        self.x_train = np.load(self.x_dir + 'x_train.npy')
        if convert_to_tensor:  
            with tf.device('/CPU:0'):
                self.x_train = tf.convert_to_tensor(self.x_train)
            gc.collect() 
        
        # Load X Valid
        self.x_valid = np.load(self.x_dir + 'x_valid.npy')
        if convert_to_tensor:  
            with tf.device('/CPU:0'):
                self.x_valid = tf.convert_to_tensor(self.x_valid)
            gc.collect()  

    def train_with_loop(self, epochs=2000, early_stopping_epochs=50, 
                        metrics_train=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                        metrics_val=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                        learning_rate=0.001, loss_fn=CategoricalCrossentropy(from_logits=False),
                        buffer_shuffle=None, batch_size=16, data_augmentation=False,
                        early_stopping_on_metric=True,
                        augment_batch_factor=2):
        # Lists of metrics must not be empty
        assert len(metrics_train) > 0, "List of metrics on train must have at least one element"
        assert len(metrics_val) > 0, "List of metrics on validation must have at least one element"
        
        # Dictionary of parameters used when method is invoked
        dict_parameters = locals().copy()
        del dict_parameters['self']
        
        # Clone model
        model = tf.keras.models.clone_model(self.model)
        
        # Compute total time to train. Begin to count
        start = time.time()
        
        # augment_batch_factor * maintained_batch_size = batch_size
        # Here we consider the batch size as the final batch size, after data augmentation on the batch
        if data_augmentation:
            assert batch_size % augment_batch_factor == 0, "Batch size must be divisible by the augment " \
                                                           "factor of the batch when doing data augmentation"
            batch_size = batch_size // augment_batch_factor
        
        # Set datasets
        self._set_datasets()
        
        # Optimizer and learning rate
        optimizer = self.optimizer
        optimizer.build(model.trainable_variables)
        optimizer.learning_rate.assign(learning_rate) # Assign learning rate to optimizer
        
        # By default, shuffle dataset by its length
        if not buffer_shuffle:
            train_dataset = self.train_dataset.shuffle(len(self.train_dataset)).batch(batch_size)
            valid_dataset = self.valid_dataset.shuffle(len(self.valid_dataset)).batch(batch_size)
        else:
            train_dataset = self.train_dataset.shuffle(buffer_shuffle).batch(batch_size)
            valid_dataset = self.valid_dataset.shuffle(buffer_shuffle).batch(batch_size)
            
        # Run loop of train
        result_history = train_model_loop(model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
                                  train_dataset=train_dataset, valid_dataset=valid_dataset,
                                  optimizer=optimizer, loss_fn=loss_fn, metrics_train=metrics_train, metrics_val=metrics_val,
                                  model_path=self.model_path, early_stopping_delta=self.early_stopping_delta,
                                  data_augmentation=data_augmentation, early_stopping_on_metric=True,                                  
                                  augment_batch_factor=augment_batch_factor)
        
        # Delete datasets to clean object
        del self.train_dataset, self.valid_dataset
        
        # End the counting of time
        end = time.time()
        
        # Save history and total time in text and in pickle file
        with open(os.path.join(self.output_dir, 'history_best_model.txt'), 'w') as f:
            f.write('Resultado = \n')
            f.write(str(result_history))
            f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos, {(end-start)/3600:.1f} horas.')
            
        with open(os.path.join(self.output_dir, 'history_pickle_best_model.pickle'), "wb") as fp: 
            pickle.dump(result_history, fp)
            
        # Use first metric of list of validation metrics to plot history
        metric0 = metrics_val[0]
        if isinstance(metric0, str): # Metric is string
            metric_name = metric0
        elif isinstance(metric0, object): # Metric is instance
            metric_name = metric0.__class__.__name__.lower()
        elif isinstance(metric0, (types.FunctionType, types.BuiltinFunctionType)): # Metric is function
            metric_name = metric0.__name__ 
            
        # Save output in plot
        show_training_plot(result_history, metric_name=metric_name, save=True, save_path=self.output_dir)
        
        # Write model summary of model used in text file and list of arguments to the method
        with open(os.path.join(self.output_dir, 'model_configuration_used.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(str(dict_parameters) + '\n')
        
        return result_history
    
    def train_with_fit(self, epochs=2000, early_stopping_epochs=50, 
                       metrics=[F1Score(), Precision(class_id=1), Recall(class_id=1)],
                       learning_rate=0.001, loss_fn=CategoricalCrossentropy(from_logits=False),
                       buffer_shuffle=None, batch_size=16, use_dataset=True, convert_to_tensor_if_numpy=False,
                       data_augmentation=False, early_stopping_on_metric=True,  
                       n_repeat=None, 
                       tensorboard_log=False, lr_decay_dict=None, **kwargs):
        # Lists of metrics must not be empty
        assert len(metrics) > 0, "List of metrics during train must have at least one element"
        
        # Dictionary of parameters used when method is invoked
        dict_parameters = locals().copy()
        del dict_parameters['self']
        
        # Clone model
        model = tf.keras.models.clone_model(self.model)
        
        # Compute total time to train. Begin to count
        start = time.time()
        
        # Set datasets or numpy arrays
        if use_dataset:
            # Set datasets
            self._set_datasets()
            if not buffer_shuffle:
                train_dataset = self.train_dataset.shuffle(len(self.train_dataset)).batch(batch_size)
                valid_dataset = self.valid_dataset.shuffle(len(self.valid_dataset)).batch(batch_size)
            else:
                train_dataset = self.train_dataset.shuffle(buffer_shuffle).batch(batch_size)
                valid_dataset = self.valid_dataset.shuffle(buffer_shuffle).batch(batch_size)
                
            # Data augmentation to dataset if specified, shuffle and batch dataset
            len_dataset = len(train_dataset)
            if not n_repeat:
                n_repeat = 1 # Set n_repeat to 1 if it is None (keep same dataset size)
            if not buffer_shuffle:
                buffer_shuffle = n_repeat*len_dataset # Set shuffle buffer to length of dataset (augmented or not) if it is None 
            if data_augmentation:
                train_dataset = self.train_dataset.repeat(n_repeat).map(transform_augment_or_maintain, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                train_dataset = self.train_dataset.repeat(n_repeat)
            train_dataset = train_dataset.shuffle(buffer_size=buffer_shuffle).batch(batch_size=batch_size).prefetch(buffer_size=1)
            valid_dataset = self.valid_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)                
        else:
            self._set_numpy_arrays(convert_to_tensor=convert_to_tensor_if_numpy)
            
        # Tensorboard callback
        if tensorboard_log:
            log_dir = os.path.join(self.output_dir, 'logs', 'fit') #, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            
        # Apply Learning Rate Decay if specified
        # Example: lr_decay_dict = {'initial_lrate':0.001, 'drop':0.1, 'epochs_drop':30}
        if lr_decay_dict:
            print(f'Initial Learning Rate={lr_decay_dict["initial_lrate"]}, Drop={lr_decay_dict["drop"]}, Epochs Drop={lr_decay_dict["epochs_drop"]}')        
            def step_decay(epoch):
                lrate = lr_decay_dict["initial_lrate"] * math.pow(lr_decay_dict["drop"], math.floor( (1+epoch) / lr_decay_dict["epochs_drop"] )  )
                return lrate
            lrate_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)
            
        # Early Stopping and Model Checkpoint Callbacks
        # If Early Stopping is defined, use first metric of list of metrics to Early Stopping
        metric0 = metrics[0]
        if isinstance(metric0, str): # Metric is string
            metric_name = metric0
        elif isinstance(metric0, object): # Metric is instance
            metric_name = metric0.__class__.__name__.lower()
        elif isinstance(metric0, (types.FunctionType, types.BuiltinFunctionType)): # Metric is function
            metric_name = metric0.__name__ 
        if early_stopping_epochs:
            # Early Stopping on Metric
            if early_stopping_on_metric:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_'+metric_name, patience=early_stopping_epochs, mode='max', restore_best_weights=True,
                                                              min_delta=self.early_stopping_delta, verbose=1)                
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                                  monitor='val_'+metric_name,
                                                                  mode='max',
                                                                  save_weights_only=False,
                                                                  verbose=1,
                                                                  save_freq='epoch',
                                                                  save_best_only=True)
            # Early Stopping on Loss
            else:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                              min_delta=self.early_stopping_delta, verbose=1)                
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                                  monitor='val_loss',
                                                                  mode='min',
                                                                  save_weights_only=False,
                                                                  verbose=1,
                                                                  save_freq='epoch',
                                                                  save_best_only=True)
        # Without Early Stopping, salve the best model, which is the model with minimum loss in validation data
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, 
                                                             monitor='val_loss',
                                                             mode='min',
                                                             save_weights_only=False,
                                                             verbose=1,
                                                             save_freq='epoch',
                                                             save_best_only=True)
            
        # Build list of callbacks
        if tensorboard_log:
            if lr_decay_dict:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback, tensorboard_callback, lrate_scheduler]
                else:
                    callbacks = [cp_callback, tensorboard_callback, lrate_scheduler]
            
            else:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback, tensorboard_callback]
                else:
                    callbacks = [cp_callback, tensorboard_callback]
        else:
            if lr_decay_dict:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback, lrate_scheduler]
                else:
                    callbacks = [cp_callback, lrate_scheduler]
            
            else:
                if early_stopping_epochs:
                    callbacks = [early_stop, cp_callback]
                else:
                    callbacks = [cp_callback]

        # Optimizer and learning rate
        optimizer = self.optimizer
        optimizer.build(model.trainable_variables)
        optimizer.learning_rate.assign(learning_rate) # Assign learning rate to optimizer
            
        # Compile model
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
                    
        # Train Model
        if use_dataset:
            historia = model.fit(x=train_dataset, epochs=epochs, verbose='auto',      # for fast debug run in only 5 batches:
                                 callbacks=callbacks, validation_data=valid_dataset)  #, steps_per_epoch=5, validation_steps=5) 
                                                                               
        else:
            historia = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose='auto',
                                 callbacks=callbacks, validation_data=(self.x_valid, self.y_valid))
            
        # Transform result in list of 2 lists (train and validation)
        history = historia.history
        
        list_loss = history['loss']
        list_metric = history[metric_name]
        list_val_loss = history['val_loss']
        list_val_metric = history['val_' + metric_name]
        
        history_train = [np.array([dupla]) for dupla in zip(list_loss, list_metric)]
        history_valid = [np.array([dupla]) for dupla in zip(list_val_loss, list_val_metric)]
        result_history = [ history_train, history_valid ]
            
        # Delete datasets or numpy arrays to clean object
        del self.train_dataset, self.valid_dataset
        
        # End the counting of time
        end = time.time()
        
        # Save history and total time in text and in pickle file
        with open(os.path.join(self.output_dir, 'history_best_model.txt'), 'w') as f:
            f.write('Resultado = \n')
            f.write(str(result_history))
            f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos, {(end-start)/3600:.1f} horas.')
            
        with open(os.path.join(self.output_dir, 'history_pickle_best_model.pickle'), "wb") as fp: 
            pickle.dump(result_history, fp)
            
        # Save output in plot
        show_training_plot(result_history, metric_name=metric_name, save=True, save_path=self.output_dir)
        
        # Write model summary of model used in text file and list of arguments to the method
        with open(os.path.join(self.output_dir, 'model_configuration_used.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(str(dict_parameters) + '\n')
        
        return result_history 
    
    
        





        



        



