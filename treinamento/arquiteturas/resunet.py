# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:39:21 2025

@author: marce
"""

# ResUnet model
# The function comprises the input shape in format (patch_height, patch_width, patch_channels)
# and the number of classes in classification
# The build_model_resunet function requires other funcions that build parts of the model

# %% Imports

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, SpatialDropout2D
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# %% ResUnet accessory functions

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1, use_dropout=None, dropout_rate=None):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    
    # Inject Spatial Dropout here if rate > 0
    if dropout_rate is not None and dropout_rate > 0:
        # training=True forces dropout to stay active; training=None turns to automatic behavior
        training_flag = True if use_dropout else None
        x = SpatialDropout2D(dropout_rate)(x, training=training_flag) 
    
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
    
    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    
    """ Addition """
    x = x + s
    return x

def decoder_block(inputs, skip_features, num_filters, use_dropout=None, dropout_rate=None):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1, use_dropout=use_dropout, dropout_rate=dropout_rate) # Pass the dropout rate down to the residual block
    return x

# %% ResUnet Build Function

def build_model_resunet(input_shape, n_classes, use_dropout=None, dropout_rate=None):
    """ RESUNET Architecture (with MC Dropout) """
    inputs = Input(input_shape)
    
    """ Encoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s
    
    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2, use_dropout=use_dropout, dropout_rate=None)
    s3 = residual_block(s2, 256, strides=2, use_dropout=use_dropout, dropout_rate=None)
    
    """ Bridge """
    b = residual_block(s3, 512, strides=2, use_dropout=use_dropout, dropout_rate=dropout_rate) # Apply Dropout

    """ Decoder 1, 2, 3 """
    x = decoder_block(b, s3, 256, use_dropout=use_dropout, dropout_rate=dropout_rate) # Deep decoder
    x = decoder_block(x, s2, 128, use_dropout=use_dropout, dropout_rate=dropout_rate) # Mid decoder
    x = decoder_block(x, s1, 64, use_dropout=use_dropout, dropout_rate=None)          # Final layer (no dropout)
    
    """ Classifier """
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(x)
    
    """ Model """
    model = Model(inputs, outputs, name="Res-UNet")
    return model


