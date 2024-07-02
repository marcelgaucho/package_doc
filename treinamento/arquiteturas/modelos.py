# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:44:18 2024

@author: Marcel
"""

from tensorflow.keras.layers import Input, concatenate, Conv2D, BatchNormalization, Activation 
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dropout, Add
from tensorflow.keras.models import Model

from .unetr_2d import config_dict, build_unetr_2d

from .segformer_tf_k3.models import SegFormer_B0
from .segformer_tf_k3.models import SegFormer_B1
from .segformer_tf_k3.models import SegFormer_B2
from .segformer_tf_k3.models import SegFormer_B3
from .segformer_tf_k3.models import SegFormer_B4
from .segformer_tf_k3.models import SegFormer_B5
# Res-UNet

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def batchnorm_elu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("elu")(x)
    
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    """ Addition """
    x = x + s
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x

def build_model_resunet(input_shape, n_classes):
    """ RESUNET Architecture """
    inputs = Input(input_shape)
    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s
    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)
    """ Bridge """
    b = residual_block(s3, 512, strides=2)

    """ Decoder 1, 2, 3 """
    x = decoder_block(b, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)
    """ Classifier """
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(x)
    """ Model """
    model = Model(inputs, outputs, name="Res-UNet")
    return model


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



def resnet_block_chamorro(x, n_filter, ind, dropout=True, dropout_rate=0.2):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    if dropout:
        x = Dropout(dropout_rate, name = 'drop_net'+str(ind))(x)
        print(f'ATENÇÃO DROPOUT RATE = {dropout_rate}')
    else:
        pass
    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    
    ## Add
    x = Add()([x, s])
    return x





# Residual U-Net model
def build_resunet_chamorro(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_chamorro(pool2, nb_filters[2], 3) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_chamorro(pool3, nb_filters[2], 4)
    
    res_block5 = resnet_block_chamorro(res_block4, nb_filters[2], 5)
    
    res_block6 = resnet_block_chamorro(res_block5, nb_filters[2], 6)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)


# Residual U-Net model para Pós-Processamento
def build_resunet_chamorro_semdropout_curta(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    print('Rede curta')
    
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1, dropout=False)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2, dropout=False) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_chamorro(pool2, nb_filters[2], 3, dropout=False)
    
    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(res_block3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)


# Residual U-Net model para Pós-Processamento
def build_resunet_chamorro_semdropout(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1, dropout=False)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2, dropout=False) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_chamorro(pool2, nb_filters[2], 3, dropout=False) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_chamorro(pool3, nb_filters[2], 4, dropout=False)
    
    res_block5 = resnet_block_chamorro(res_block4, nb_filters[2], 5, dropout=False)
    
    res_block6 = resnet_block_chamorro(res_block5, nb_filters[2], 6, dropout=False)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)



# Model CorrED
def conv_block_corred(inputs, num_filters, part='encoder'):
    """ Convolutional Layers """
    if part=='encoder':
        x = Conv2D(num_filters , (3 , 3) , padding='same', strides=2)(inputs)
    elif part=='decoder':
        x = Conv2D(num_filters , (3 , 3) , padding='same', strides=1)(inputs)
    else:
        raise Exception('Valor de parte deve "encoder" ou "decoder"')
        
    x = batchnorm_elu(x)
        
    x = Conv2D(num_filters , (3 , 3) , padding='same', strides=1)(x)
    
    x = batchnorm_elu(x)
    
    return x

def build_model_corred(input_shape, n_classes):
    input_img = Input(input_shape)
    
    '''Encoder'''
    # Bloco Encoder 1 - 2 Convoluções seguindas de um Batch Normalization com Relu
    b1 = conv_block_corred(input_img, 16)
    
    # Bloco Encoder 2 - " " "
    b2 = conv_block_corred(b1, 32, part='encoder')
    
    # Bloco Encoder 3
    b3 = conv_block_corred(b2, 32, part='encoder')
    
    # Bloco Encoder 4
    b4 = conv_block_corred(b3, 32, part='encoder')
    
    
    '''Decoder'''
    # Bloco Decoder 1 - Upsample seguido de 2 convoluções
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(b4)
    up1 = conv_block_corred(up1, 16, part='decoder')
    
    # Bloco Decoder 2 - " " "
    up2 = UpSampling2D((2, 2), interpolation='bilinear')(up1)
    up2 = conv_block_corred(up2, 16, part='decoder')
    
    # Bloco Decoder 3
    up3 = UpSampling2D((2, 2), interpolation='bilinear')(up2)
    up3 = conv_block_corred(up3, 16, part='decoder')
    
    # Bloco Decoder 4
    up4 = UpSampling2D((2, 2), interpolation='bilinear')(up3)
    up4 = conv_block_corred(up4, 16, part='decoder')
    
    # Final Convolution
    output = Conv2D(n_classes, (1,1), activation = 'softmax')(up4)
    
    
    return Model(inputs = input_img, outputs = output, name='CorrED')

        
# Build UNet or Res-UNet or CorrED
def build_model(input_shape, n_classes, model_type='unet'):
    if model_type == 'unet':
        return build_model_unet(input_shape, n_classes)
    
    elif model_type == 'resunet':
        return build_model_resunet(input_shape, n_classes)
    
    elif model_type == 'resunet chamorro':
        return build_resunet_chamorro(input_shape, (64, 128, 256), n_classes)
    
    elif model_type == 'corred':
        return build_model_corred(input_shape, n_classes)

    elif model_type == 'resunet chamorro pos':
         return build_resunet_chamorro_semdropout(input_shape, (64, 128, 256), n_classes)
     
    elif model_type == 'resunet chamorro pos curta':
         return build_resunet_chamorro_semdropout_curta(input_shape, (64, 128, 256), n_classes)
     
    elif model_type == 'unetr':
        return build_unetr_2d(input_shape, config_dict)
    
    elif model_type == 'segformer_b0':
        return SegFormer_B0(input_shape, n_classes)

    elif model_type == 'segformer_b1':
        return SegFormer_B1(input_shape, n_classes)  

    elif model_type == 'segformer_b2':
        return SegFormer_B2(input_shape, n_classes)    
    
    elif model_type == 'segformer_b3':
        return SegFormer_B3(input_shape, n_classes)  
    
    elif model_type == 'segformer_b4':
        return SegFormer_B4(input_shape, n_classes)  
    
    elif model_type == 'segformer_b5':
        return SegFormer_B5(input_shape, n_classes)  
    
    else:
        raise Exception("Model options are 'unet' and 'resunet' and 'resunet chamorro' and 'corred' and "
                        "'resunet chamorro pos' and 'resunet chamorro pos curta' and 'unetr'")