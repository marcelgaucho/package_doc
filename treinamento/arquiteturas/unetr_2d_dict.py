# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:20:45 2025

@author: marce
"""

# UNETR dictionary of hyperparameters

# %% Hyperparameters dictionary 

IMG_SIZE = 256 # This is the equivalent to the patch size for CNN
NUM_CHANNELS = 3
PATCH_SIZE = 16 # This is the size of a non-overlapping patch in the image
NUM_PATCHES = (IMG_SIZE//PATCH_SIZE) ** 2

HIDDEN_DIM = 128 # Embedding dimension  
NUM_LAYERS = 12 # Number of Encoders (Layers) in the Transformer
NUM_HEADS = 8 # Number of Heads in Multihead Attention
MLP_DIM = 256 # MLP dimension
DROPOUT = 0.1 # Dropout rate
MAX_FILTERS = 512 # Maximum number of filters in the Decoder


# There are 4 deconvolutions and we start from the patch size value
# In each deconvolution, the patch duplicate its size
# In the end the size after deconvolution has to be equal to image size
# So the image size must be patch_size*2*2*2*2 = patch_size*2^4 = patch_size*16

config_dict = {'IMG_SIZE': IMG_SIZE,
               'NUM_CHANNELS': NUM_CHANNELS,
               'PATCH_SIZE': PATCH_SIZE,
               'NUM_PATCHES': NUM_PATCHES,
               'HIDDEN_DIM': HIDDEN_DIM,
               'NUM_LAYERS': NUM_LAYERS,
               'NUM_HEADS': NUM_HEADS, 
               'MLP_DIM': MLP_DIM,
               'DROPOUT': DROPOUT,
               'MAX_FILTERS': MAX_FILTERS}