# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:14:37 2025

@author: marce
"""

# U-Net model
# The function comprises the input shape in format (patch_height, patch_width, patch_channels)
# and the number of classes in classification

# %% Imports

from tensorflow.keras.layers import Input, concatenate, Conv2D
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose
from tensorflow.keras.models import Model

# %% Function to build U-Net

def build_model_unet(input_shape, n_classes):
    # U-Net architecture
    input_img = Input(input_shape)

    # Contract stage
    f1 = 64
    b1conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b1conv1')(input_img)
    b1conv2 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b1conv2')(b1conv1)

    pool1 = MaxPool2D((2 , 2), name = 'pooling1')(b1conv2)

    b2conv1 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b2conv1')(pool1)
    b2conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b2conv2')(b2conv1)

    pool2 = MaxPool2D((2 , 2), name = 'pooling2')(b2conv2)

    b3conv1 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b3conv1')(pool2)
    b3conv2 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b3conv2')(b3conv1)

    pool3 = MaxPool2D((2 , 2), name = 'pooling3')(b3conv2)

    b4conv1 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b4conv1')(pool3)
    b4conv2 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b4conv2')(b4conv1)

    pool4 = MaxPool2D((2 , 2), name = 'pooling4')(b4conv2)

    b5conv1 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name = 'b5conv1')(pool4)
    b5conv2 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name = 'b5conv2')(b5conv1)

    # Expansion stage
    upsample1 = Conv2DTranspose(f1*8, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling1')(b5conv2)
    concat1 = concatenate( [upsample1,b4conv2] )
    b6conv1 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b6conv1')(concat1)
    b6conv2 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b6conv2')(b6conv1)
    
    upsample2 = Conv2DTranspose(f1*4, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling2')(b6conv2)
    concat2 = concatenate( [upsample2,b3conv2] )
    b7conv1 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b7conv1')(concat2)
    b7conv2 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b7conv2')(b7conv1)

    upsample3 = Conv2DTranspose(f1*2, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling3')(b7conv2)
    concat3 = concatenate( [upsample3,b2conv2] )
    b8conv1 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b8conv1')(concat3)
    b8conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b8conv2')(b8conv1)

    upsample4 = Conv2DTranspose(f1, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling4')(b8conv2)
    concat4 = concatenate( [upsample4,b1conv2] )
    b9conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv1')(concat4)
    b9conv2 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv2')(b9conv1)

    # Output segmentation
    b9conv3 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv3')(b9conv2)
    b9conv4 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv4')(b9conv3)

    output = Conv2D(n_classes,(1,1), activation = 'softmax')(b9conv4)
    
    return Model(inputs = input_img, outputs = output, name='U-Net')


