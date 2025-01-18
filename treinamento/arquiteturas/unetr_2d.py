# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 00:24:23 2024

@author: Marcel
"""

# UNETR model
# The function comprises the input shape in format (patch_height, patch_width, patch_channels),
# the number of classes in classification and a hyperparameters dictionary
# The build_unetr_2d function requires other funcions and a class that build parts of the model

# %% Imports

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Embedding, LayerNormalization,
                                     MultiHeadAttention, Add, Dropout, Reshape,
                                     Conv2DTranspose, Conv2D, 
                                     BatchNormalization, ReLU, Concatenate)

from tensorflow.keras.models import Model

from tensorflow.keras import layers

import numpy as np

from math import log2

# %% Accessory functions

def mlp(x, config_dict):
    x = Dense(config_dict["MLP_DIM"], activation="gelu")(x)
    x = Dense(config_dict["HIDDEN_DIM"], activation="gelu")(x)
    return x

def transformer_encoder(x, config_dict):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=config_dict["NUM_HEADS"], key_dim=config_dict["HIDDEN_DIM"]
    )(x, x)
    x = Dropout(config_dict["DROPOUT"])(x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, config_dict)
    x = Add()([x, skip_2])
    x = Dropout(config_dict["DROPOUT"])(x)
    
    return x

def conv_block(x, num_filters, kernel_size=3):
    x = Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def deconv_block(x, num_filters, strides=2):
    x = Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=strides)(x)
    return x

# %% Class that extract patches from the input image (equivalent to a CNN patch)

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        
        patch_dims = tf.shape(patches)[-1]
        num_patches_h = tf.shape(patches)[1]
        num_patches_w = tf.shape(patches)[2]

        patches = tf.reshape(patches, [batch_size, num_patches_h*num_patches_w,
                                       patch_dims])
        return patches

# %% UNETR Build Function

def build_unetr_2d(input_shape, n_classes, config_dict):
    """ Inputs """
    inputs = Input(input_shape) 
    
    """ Patch Extraction """
    patches = Patches(config_dict["PATCH_SIZE"])(inputs)
    
    """ Patch + Position Embeddings """
    patch_embed = Dense(config_dict["HIDDEN_DIM"])(patches)  
    
    positions = tf.range(start=0, limit=config_dict["NUM_PATCHES"], delta=1) 
    
    pos_embed = Embedding(input_dim=config_dict["NUM_PATCHES"], output_dim=config_dict["HIDDEN_DIM"])(positions)
    
    x = patch_embed + pos_embed 
    
    """ Transformer Encoder """
    skip_connection_index = [3, 6, 9, 12]
    skip_connections = []
    
    for i in range(1, config_dict['NUM_LAYERS']+1, 1):
        x = transformer_encoder(x, config_dict)

        if i in skip_connection_index:
            skip_connections.append(x)
            
    """ CNN Decoder """
    z3, z6, z9, z12 = skip_connections 
    
    # Reshape para shape original da imagem ou 
    z0 = Reshape((config_dict["IMG_SIZE"], config_dict["IMG_SIZE"], 
                  config_dict["NUM_CHANNELS"]))(inputs) 
    
    # para shape semelhante, no caso das camadas usadas nos Transformers
    shape = (
        config_dict["IMG_SIZE"]//config_dict["PATCH_SIZE"],
        config_dict["IMG_SIZE"]//config_dict["PATCH_SIZE"],
        config_dict["HIDDEN_DIM"]
    ) 
    z3 = Reshape(shape)(z3) 
    z6 = Reshape(shape)(z6) 
    z9 = Reshape(shape)(z9) 
    z12 = Reshape(shape)(z12) 
    
    '''
    ## Additional layers for managing different patch sizes
    # Coloca layers adicionais para que imagem fique no formato que se espera
    total_upscale_factor = int(log2(config_dict["PATCH_SIZE"]))
    upscale = total_upscale_factor - 4
    ic(total_upscale_factor, upscale)
    
    if upscale >= 2: ## Patch size 16 or greater
        z3 = deconv_block(z3, z3.shape[-1], strides=2**upscale)
        z6 = deconv_block(z6, z6.shape[-1], strides=2**upscale)
        z9 = deconv_block(z9, z9.shape[-1], strides=2**upscale)
        z12 = deconv_block(z12, z12.shape[-1], strides=2**upscale)
        # print(z3.shape, z6.shape, z9.shape, z12.shape)

    if upscale < 0: ## Patch size less than 16
        p = 2**abs(upscale)
        z3 = MaxPool2D((p, p))(z3)
        z6 = MaxPool2D((p, p))(z6)
        z9 = MaxPool2D((p, p))(z9)
        z12 = MaxPool2D((p, p))(z12)
                
    ic(z3, z6, z9, z12)
    '''
    
    ## Decoder 1
    x = deconv_block(z12, config_dict["MAX_FILTERS"]) 

    s = deconv_block(z9, config_dict["MAX_FILTERS"])
    s = conv_block(s, config_dict["MAX_FILTERS"]) 

    x = Concatenate()([x, s]) 

    x = conv_block(x, config_dict["MAX_FILTERS"]) 
    x = conv_block(x, config_dict["MAX_FILTERS"])

    ## Decoder 2
    x = deconv_block(x, config_dict["MAX_FILTERS"]//2) 

    s = deconv_block(z6, config_dict["MAX_FILTERS"]//2)
    s = conv_block(s, config_dict["MAX_FILTERS"]//2)
    s = deconv_block(s, config_dict["MAX_FILTERS"]//2)
    s = conv_block(s, config_dict["MAX_FILTERS"]//2) 

    x = Concatenate()([x, s]) 
    x = conv_block(x, config_dict["MAX_FILTERS"]//2)
    x = conv_block(x, config_dict["MAX_FILTERS"]//2) 

    ## Decoder 3
    x = deconv_block(x, config_dict["MAX_FILTERS"]//4) 

    s = deconv_block(z3, config_dict["MAX_FILTERS"]//4)
    s = conv_block(s, config_dict["MAX_FILTERS"]//4)
    s = deconv_block(s, config_dict["MAX_FILTERS"]//4)
    s = conv_block(s, config_dict["MAX_FILTERS"]//4)
    s = deconv_block(s, config_dict["MAX_FILTERS"]//4)
    s = conv_block(s, config_dict["MAX_FILTERS"]//4) 

    x = Concatenate()([x, s]) 
    x = conv_block(x, config_dict["MAX_FILTERS"]//4)
    x = conv_block(x, config_dict["MAX_FILTERS"]//4) 

    ## Decoder 4
    x = deconv_block(x, config_dict["MAX_FILTERS"]//8) 

    s = conv_block(z0, config_dict["MAX_FILTERS"]//8)
    s = conv_block(s, config_dict["MAX_FILTERS"]//8) 

    x = Concatenate()([x, s])
    x = conv_block(x, config_dict["MAX_FILTERS"]//8)
    x = conv_block(x, config_dict["MAX_FILTERS"]//8)

    """ Output """
    outputs = Conv2D(n_classes, kernel_size=1, padding="same", activation="softmax")(x)
    
    return Model(inputs, outputs, name="UNETR_2D")

