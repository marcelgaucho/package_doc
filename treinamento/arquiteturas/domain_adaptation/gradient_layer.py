# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:47:22 2026

@author: Marcel
"""

# %% Import Libraries

import tensorflow as tf
from tensorflow.keras.layers import Layer

# %% Gradient Reversal Layer

@tf.custom_gradient
def gradient_reverse(x, lambda_weight):
    def grad(dy):
        # Multiply gradients by negative lambda_weight during backprop
        return dy * -lambda_weight, None
    return x, grad

class GradientReversalLayer(Layer):
    def __init__(self, lambda_weight=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_weight = tf.Variable(lambda_weight, dtype=tf.float32, trainable=False)

    def call(self, x):
        return gradient_reverse(x, self.lambda_weight)
    
    
