# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:24:57 2024

@author: marce
"""

# F1-Score and Relaxed F1-Score metrics implementation to use in early stopping

# %% Imports

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from skimage.morphology import disk

# %% F1-Score class

class CustomF1Score(Metric):
    def __init__(self, name='f1score', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
        # initializing an object of the super class
        super().__init__(name=name, **kwargs) # super(F1Score, self)
          
        # initializing state variables
        self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives 
        self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
        self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives
          
        # initializing other atrributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2 
        self.threshold = threshold
        self.epsilon = epsilon
    
    def update_state(self, ytrue, ypred, sample_weight=None):
        # Pega só referente a classe 1
        ypred = ypred[..., 1:2] 
        ytrue = ytrue[..., 1:2] 
          
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)
        
        #print(f'Shape Shape de Y True é {ytrue.shape}')
        #print(f'Shape Shape de Y Pred é {ytrue.shape}')
          
        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
            
        self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
        self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
        self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute
    
    def result(self):
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
          
        # calculating fbeta
        self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
        
        return self.fb
    
    def reset_state(self):
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero
        

# %% Relaxed F1-Score class

class RelaxedF1Score(Metric):
    def __init__(self, name='f1score', beta=1, threshold=0.5, epsilon=1e-7, 
                 radius_px=3, **kwargs):
        # initializing an object of the super class
        super().__init__(name=name, **kwargs) # super(F1Score, self)
          
        # initializing state variables
        
        # For relaxed precision
        self.tp_relax_prec = self.add_weight(name='tp_relax_prec', initializer='zeros') # initializing true positives 
        self.actual_positive_relax_prec = self.add_weight(name='ap_relax_prec', initializer='zeros') # initializing actual positives
        self.predicted_positive_relax_prec = self.add_weight(name='pp_relax_prec', initializer='zeros') # initializing predicted positives
        
        # For relaxed recall
        self.tp_relax_recall = self.add_weight(name='tp_relax_recall', initializer='zeros') # initializing true positives 
        self.actual_positive_relax_recall = self.add_weight(name='ap_relax_recall', initializer='zeros') # initializing actual positives
        self.predicted_positive_relax_recall = self.add_weight(name='pp_relax_recall', initializer='zeros') # initializing predicted positives
          
        # initializing other atrributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2 
        self.threshold = threshold
        self.epsilon = epsilon
        
        # Making Structure Element
        self.struct_elem = disk(radius_px)[:, :, np.newaxis]
        
    def _buffer(self, tensor):
        # Parameters to pass to function
        strides = (1, 1, 1, 1)
        padding = 'SAME'
        dilations = (1, 1, 1, 1)
        
        # Buffer as dilation by disk
        buffered = tf.nn.dilation2d(input=tensor, filters=self.struct_elem, 
                                       strides=strides, padding=padding,
                                       data_format="NHWC", dilations=dilations)
        buffered = buffered - tf.ones_like(buffered) # Necessary for binary dilation
        
        return buffered
    
    def update_state(self, ytrue, ypred, sample_weight=None):
        # Pega só referente a classe 1
        ypred = ypred[..., 1:2] 
        ytrue = ytrue[..., 1:2]
        
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)
        
        #print(f'Shape Shape de Y True é {ytrue.shape}')
        #print(f'Shape Shape de Y Pred é {ytrue.shape}')
          
        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
        
        # Buffer ytrue and ypred
        ytrue_buffer = self._buffer(ytrue)
        ypred_buffer = self._buffer(ypred)
        
        # Update variables for relaxed precision
        self.tp_relax_prec.assign_add(tf.reduce_sum(ytrue_buffer*ypred)) # updating true positives atrribute
        self.predicted_positive_relax_prec.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
        # self.actual_positive_relax_prec.assign_add(tf.reduce_sum(ytrue_buffer)) # updating actual positive atrribute
        
        # Update variables for relaxed recall
        self.tp_relax_recall.assign_add(tf.reduce_sum(ytrue*ypred_buffer)) 
        # self.predicted_positive_relax_prec.assign_add(tf.reduce_sum(ypred_buffer)) 
        self.actual_positive_relax_recall.assign_add(tf.reduce_sum(ytrue)) 
    
    def result(self):
        self.relaxed_precision = self.tp_relax_prec/(self.predicted_positive_relax_prec+self.epsilon) # calculates precision
        self.relaxed_recall = self.tp_relax_recall/(self.actual_positive_relax_recall+self.epsilon) # calculates recall
          
        # calculating fbeta
        self.fb = (1+self.beta_squared)*self.relaxed_precision*self.relaxed_recall / (self.beta_squared*self.relaxed_precision + self.relaxed_recall + self.epsilon)
        
        return self.fb
    
    def reset_state(self):
        # For relaxed precision
        self.tp_relax_prec.assign(0) # resets true positives to zero
        self.predicted_positive_relax_prec.assign(0) # resets predicted positives to zero
        
        # For relaxed recall
        self.tp_relax_recall.assign(0)
        self.actual_positive_relax_recall.assign(0) 
        
        
# %% Masked F1-Score (mask all-zeros pixels)
        
class MaskedF1Score(Metric):
    def __init__(self, name='masked_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(class_id=1)
        self.recall = tf.keras.metrics.Recall(class_id=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. Create mask from y_true (1 if labeled, 0 if all-zeros)
        mask = tf.reduce_sum(y_true, axis=-1)
        mask = tf.cast(mask > 0, tf.float32)
        
        # 2. Update internal precision/recall with the mask
        self.precision.update_state(y_true, y_pred, sample_weight=mask)
        self.recall.update_state(y_true, y_pred, sample_weight=mask)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-7))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
        
# %% Masked Precision

class MaskedPrecision(Metric):
    def __init__(self, name='masked_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(class_id=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. Create mask from y_true (1 if labeled, 0 if all-zeros)
        mask = tf.reduce_sum(y_true, axis=-1)
        mask = tf.cast(mask > 0, tf.float32)
        
        # 2. Update internal precision/recall with the mask
        self.precision.update_state(y_true, y_pred, sample_weight=mask)

    def result(self):
        p = self.precision.result()
        return p

    def reset_state(self):
        self.precision.reset_state()
        
# %% Masked Recall
        
class MaskedRecall(Metric):
    def __init__(self, name='masked_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.recall = tf.keras.metrics.Recall(class_id=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. Create mask from y_true (1 if labeled, 0 if all-zeros)
        mask = tf.reduce_sum(y_true, axis=-1)
        mask = tf.cast(mask > 0, tf.float32)
        
        # 2. Update internal precision/recall with the mask
        self.recall.update_state(y_true, y_pred, sample_weight=mask)

    def result(self):
        r = self.recall.result()
        return r

    def reset_state(self):
        self.recall.reset_state()

