# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:58:42 2026

@author: Marcel
"""

# %% Import Libraries

from tensorflow.keras.layers import Conv2D, Input, Add, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from ..resunet import batchnorm_relu, residual_block, decoder_block
from .gradient_layer import GradientReversalLayer

# %%

def build_encoder(input_shape, dropout_rate=0):
    """ Feature Extractor (Encoder) """
    inputs = Input(shape=input_shape, name="encoder_input")
    
    # Encoder 1
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = Add()([x, s])
    
    # Encoder 2, 3
    s2 = residual_block(s1, 128, strides=2, dropout_rate=0)
    s3 = residual_block(s2, 256, strides=2, dropout_rate=0)
    
    # Bridge
    b = residual_block(s3, 512, strides=2, dropout_rate=dropout_rate)
    
    # We must return the bridge AND the skip connections for the decoder
    return Model(inputs, [b, s3, s2, s1], name="FeatureExtractor")

# %%

def build_decoder(bridge_shape, s3_shape, s2_shape, s1_shape, n_classes, dropout_rate=0):
    """ Label Predictor (Decoder) """
    b_input = Input(shape=bridge_shape, name="bridge_input")
    s3_input = Input(shape=s3_shape, name="s3_input")
    s2_input = Input(shape=s2_shape, name="s2_input")
    s1_input = Input(shape=s1_shape, name="s1_input")
    
    # Decoder blocks
    x = decoder_block(b_input, s3_input, 256, dropout_rate=dropout_rate)
    x = decoder_block(x, s2_input, 128, dropout_rate=dropout_rate)
    x = decoder_block(x, s1_input, 64, dropout_rate=0)
    
    # Classifier (Segmentation Mask)
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax", name="seg_output")(x)
    
    return Model([b_input, s3_input, s2_input, s1_input], outputs, name="LabelPredictor")

# %%

def build_domain_classifier(bridge_shape):
    """ Domain Classifier (Source vs Target) """
    inputs = Input(shape=bridge_shape, name="domain_input")
    
    # Gradient Reversal Layer
    x = GradientReversalLayer()(inputs)
    
    # Reduce spatial dimensions to make a single domain prediction per image
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="sigmoid", name="domain_output")(x)
    
    return Model(inputs, x, name="DomainClassifier")