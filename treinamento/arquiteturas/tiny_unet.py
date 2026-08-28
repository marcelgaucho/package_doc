# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:03:01 2026

@author: Marcel
"""

# %% Import Libraries

from tensorflow.keras import layers, Model

# %% Conv block

def conv_block(x, filters, kernel_size=3):
    """A two-convolution block."""
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x)   
    
    return x

# %% Tiny-UNet build function

def build_tiny_unet(input_shape, num_classes=2):
    """
    Tiny-UNet for semantic segmentation.                                    
    
    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of segmentation classes.

    Returns:
        tf.keras.Model: Tiny U-Net model.   
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER ---
    c1 = conv_block(inputs, 16)                            # H
    p1 = layers.MaxPooling2D(pool_size=2)(c1)              # H/2
    
    c2 = conv_block(p1, 32)                                # H/2
    p2 = layers.MaxPooling2D(pool_size=2)(c2)              # H/4
    
    c3 = conv_block(p2, 64)                                # H/4
    p3 = layers.MaxPooling2D(pool_size=2)(c3)              # H/8
    
    # ---- BOTTLENECK ----
    bn = conv_block(p3, 128)                               # H/8
    
    # ---- DECODER ----
    u3 = layers.UpSampling2D(size=2, interpolation="bilinear")(bn) # H/4
    u3 = layers.Concatenate()([u3, c3])                    # H/4
    d3 = conv_block(u3, 64)                                # H/4 
    
    u2 = layers.UpSampling2D(size=2, interpolation="bilinear")(d3) # H/2
    u2 = layers.Concatenate()([u2, c2])                    # H/2
    d2 = conv_block(u2, 32)                                # H/2
    
    u1 = layers.UpSampling2D(size=2, interpolation="bilinear")(d2) # H
    u1 = layers.Concatenate()([u1, c1])                    # H
    d1 = conv_block(u1, 16)                                # H
    
    # --- OUTPUT ---
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same", activation="softmax")(d1)
    
    model = Model(inputs, outputs, name="tiny_unet")
    return model