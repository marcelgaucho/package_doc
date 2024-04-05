

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:01:23 2023

@author: marce
"""

# Importação das bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os
import math
import pickle
from functions_extract import normalization
from functions_bib import salva_arrays
from functions_extract import filtra_tiles_estradas, extract_patches_from_tiles


                                    
# Images Directories
train_test_dir = 'entrada/'
y_dir = 'y_directory/'

# %% Extrai patches (Info)

# Tamanho do patch. Patch é quadrado (altura=largura)
# patch_size = 224
patch_size = 256

# Stride do Patch. Com um stride menor que a largura do patch há sobreposição entre os patches
# 25% de sobreposição entre patches
patch_overlap = 0.25
# patch_overlap = 0.0625 # In Mnih the overlap is 14 pixels. 224*0.0625 = 14
patch_stride = patch_size - int(patch_size * patch_overlap)

# Número de canais da imagem/patch
# image_channels = 4
image_channels = 3


# Dimensões do patch
input_shape = (patch_size, patch_size, image_channels)




# %% Varre diretório de treino e de validação para abrir os tiles dos quais os patches serão extraídos

train_imgs_tiles_dir = r'dataset_massachusetts_mnih_mod/train/input'
valid_imgs_tiles_dir = r'dataset_massachusetts_mnih_mod/validation/input'
train_labels_tiles_dir = r'dataset_massachusetts_mnih_mod/train/maps'
valid_labels_tiles_dir = r'dataset_massachusetts_mnih_mod/validation/maps'
test_imgs_tiles_dir = r'dataset_massachusetts_mnih_mod/test/input'
test_labels_tiles_dir = r'dataset_massachusetts_mnih_mod/test/maps'
# train_imgs_tiles_dir = r'tiles_sentinel/imgs/train'
# valid_imgs_tiles_dir = r'tiles_sentinel/imgs/valid'
# train_labels_tiles_dir = r'tiles_sentinel/masks/2016/train'
# valid_labels_tiles_dir = r'tiles_sentinel/masks/2016/valid'
# test_imgs_tiles_dir = r'tiles_sentinel/imgs/test'
# test_labels_tiles_dir = r'tiles_sentinel/masks/2018/test'


#newroads_valid_labels_tiles_dir = r'new_teste_tiles\masks\valid_new_roads'
#newroads_test_labels_tiles_dir = r'new_teste_tiles\masks\test_new_roads'

train_imgs_tiles = [os.path.join(train_imgs_tiles_dir, arq) for arq in os.listdir(train_imgs_tiles_dir)]
valid_imgs_tiles = [os.path.join(valid_imgs_tiles_dir, arq) for arq in os.listdir(valid_imgs_tiles_dir)]
test_imgs_tiles = [os.path.join(test_imgs_tiles_dir, arq) for arq in os.listdir(test_imgs_tiles_dir)]

train_labels_tiles = [os.path.join(train_labels_tiles_dir, arq) for arq in os.listdir(train_labels_tiles_dir)]
valid_labels_tiles = [os.path.join(valid_labels_tiles_dir, arq) for arq in os.listdir(valid_labels_tiles_dir)]
test_labels_tiles = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
#newroads_valid_labels_tiles = [os.path.join(newroads_valid_labels_tiles_dir, arq) for arq in os.listdir(newroads_valid_labels_tiles_dir)]
#newroads_test_labels_tiles = [os.path.join(newroads_test_labels_tiles_dir, arq) for arq in os.listdir(newroads_test_labels_tiles_dir)]

# Build Dictionaries
train_imgs_labels_dict = dict(zip(train_imgs_tiles, train_labels_tiles))
valid_imgs_labels_dict = dict(zip(valid_imgs_tiles, valid_labels_tiles))
test_imgs_labels_dict = dict(zip(test_imgs_tiles, test_labels_tiles))
#newroads_valid_labels_dict = dict(zip(valid_imgs_tiles, newroads_valid_labels_tiles)) # Tanto faz colocar valid_imgs_tiles ou test_imgs_tiles,
                                                                                      # porque os tiles das imagens serão descartados, já que eles terão
                                                                                      # sido criados de qualquer maneira
#newroads_test_labels_dict = dict(zip(test_imgs_tiles, newroads_test_labels_tiles))





# %% Extrai patches de treino


x_train, y_train, _, _ = extract_patches_from_tiles(train_imgs_labels_dict, patch_size, patch_stride)

x_valid, y_valid, _, _ = extract_patches_from_tiles(valid_imgs_labels_dict, patch_size, patch_stride)

x_test, y_test, len_tiles_test, shape_tiles_test = extract_patches_from_tiles(test_imgs_labels_dict, patch_size, patch_stride, border_patches=True)

#_, newroads_y_valid = extract_patches_from_tiles(newroads_valid_labels_dict, patch_size, patch_stride, border_patches=True)

#_, newroads_y_test = extract_patches_from_tiles(newroads_test_labels_dict, patch_size, patch_stride, border_patches=True)


# %% Teste com 1 tile apenas

test_imgs_labels_dict0 = {r'new_teste_tiles\\imgs\\test\\tile_0_2.tif': r'new_teste_tiles\\masks\\test\\reftile_0_2.tif'}

x_test0, y_test0 = extract_patches_from_tiles(test_imgs_labels_dict0, patch_size, patch_stride, border_patches=True)

# %% Seleciona patches de x_train que não contenham NODATA

nodata_value = -9999
patches_with_no_nodata_indexes = [i for i in range(len(x_train)) if not np.any(x_train[i] == nodata_value)]
x_train = x_train[patches_with_no_nodata_indexes]
y_train = y_train[patches_with_no_nodata_indexes]



# %% Seleciona patches de x_valid que não contenham NODATA

nodata_value = -9999
patches_with_no_nodata_indexes = [i for i in range(len(x_valid)) if not np.any(x_valid[i] == nodata_value)]
x_valid = x_valid[patches_with_no_nodata_indexes]
y_valid = y_valid[patches_with_no_nodata_indexes]


# %% Seleciona patches de x_train que não contenham NODATA

# Descomentar se vai se usar a soma (255 + 255 + 255) como valor de Nodata
# nodata_value = 765
# patches_with_no_nodata_indexes1 = [i for i in range(len(x_valid)) if not np.any(x_valid[i].sum(axis=2) == nodata_value)]
nodata_value = (255, 255, 255)
patches_with_no_nodata_indexes = [i for i in range(len(x_train)) if not np.any((x_train[i] == nodata_value).all(axis=2))]
x_train = x_train[patches_with_no_nodata_indexes]
y_train = y_train[patches_with_no_nodata_indexes]


# %% Seleciona patches de x_valid que não contenham NODATA

# Descomentar se vai se usar a soma (255 + 255 + 255) como valor de Nodata
# nodata_value = 765
# patches_with_no_nodata_indexes1 = [i for i in range(len(x_valid)) if not np.any(x_valid[i].sum(axis=2) == nodata_value)]
nodata_value = (255, 255, 255)
patches_with_no_nodata_indexes = [i for i in range(len(x_valid)) if not np.any((x_valid[i] == nodata_value).all(axis=2))]
x_valid = x_valid[patches_with_no_nodata_indexes]
y_valid = y_valid[patches_with_no_nodata_indexes]



# %% Filtra Patches que contenham mais que X% de pixels de estrada

x_train, y_train, indices_maior_train = filtra_tiles_estradas(x_train, y_train, 1)
x_valid, y_valid, indices_maior_valid = filtra_tiles_estradas(x_valid, y_valid, 1)

# %% For Masachussets Dataset

# Set 1 where y = 255 (1=Estrada, 0=Não-Estrada)
y_train[y_train==255] = 1
y_valid[y_valid==255] = 1
y_test[y_test==255] = 1



# %% For Masachussets Dataset

# Normaliza dados
x_train = x_train/255
x_valid = x_valid/255
x_test = x_test/255


# %% For Masachussets Dataset

# x_train_filter, y_train_filter, indices_maior_train = filtra_tiles_estradas(x_train, y_train, 5)

# x_valid_filter, y_valid_filter, indices_maior_valid = filtra_tiles_estradas(x_valid, y_valid, 0.5)

x_train_filter, y_train_filter, indices_maior_train = filtra_tiles_estradas(x_train, y_train, 3)

x_valid_filter, y_valid_filter, indices_maior_valid = filtra_tiles_estradas(x_valid, y_valid, 1)

x_test_filter, y_test_filter, indices_maior_test = filtra_tiles_estradas(x_test, y_test, 0)





# %% Faz aumento de dados (Função)

def augment_data(images: np.ndarray) -> np.ndarray:
    augmented_images = []
    for image in images:
        # Rotação 90 graus
        augmented_images.append(np.rot90(image, 1))
        # Rotação 180 graus
        augmented_images.append(np.rot90(image, 2))
        # Rotação 270 graus
        augmented_images.append(np.rot90(image, 3))
        # Flip horizontal
        augmented_images.append(np.fliplr(image))
        # Flip vertical
        augmented_images.append(np.flipud(image))
    
    return np.stack(augmented_images)


# Faz aumento de dados para arrays x e y na forma
# (batches, heigth, width, channels)
# Rotações sentido anti-horário 90, 180, 270
# Espelhamento Vertical e Horizontal
def aumento_dados(x, y):
    # Rotações Sentido Anti-Horário
    # Rotação 90 graus
    x_rot90 = np.rot90(x, k=1, axes=(1,2))
    y_rot90 = np.rot90(y, k=1, axes=(1,2))
    # Rotação 180 graus 
    x_rot180 = np.rot90(x, k=2, axes=(1,2))
    y_rot180 = np.rot90(y, k=2, axes=(1,2))
    # Rotação 270 graus 
    x_rot270 = np.rot90(x, k=3, axes=(1,2))
    y_rot270 = np.rot90(y, k=3, axes=(1,2))
    
    # Espelhamento Vertical (Mirror)
    x_mirror = np.flip(x, axis=2)
    y_mirror = np.flip(y, axis=2)
    
    # Espelhamento Horizontal (Flip)
    x_flip = np.flip(x, axis=1)
    y_flip = np.flip(y, axis=1)

    x_aum = np.concatenate((x, x_rot90, x_rot180, x_rot270, x_mirror, x_flip))
    y_aum = np.concatenate((y, y_rot90, y_rot180, y_rot270, y_mirror, y_flip))

    return x_aum, y_aum                
    
    
    

# %% Faz aumento de dados

# x_train, y_train = aumento_dados(x_train, y_train)
x_train_filter, y_train_filter = aumento_dados(x_train_filter, y_train_filter)


#x_valid, y_valid = aumento_dados(x_valid, y_valid)


# %% Seleciona Patches que contenham estrada

# Exemplo de um patch de treinamento que contenha estrada
image_index = 0

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Rotacionado 90 graus)
image_index = 1430

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Rotated 90 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Rotated 90 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Rotacionado 180 graus)
image_index = 2860

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Rotated 180 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Rotated 180 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Rotacionado 270 graus)
image_index = 4290

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Rotated 270 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Rotated 270 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Espelhamento Vertical)
image_index = 5720

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Espelhamento Vertical)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Espelhamento Vertical)', fontsize=20)
ax2.axis('off')

# Exemplo de um patch de treinamento que contenha estrada (Espelhamento Horizontal)
image_index = 7150

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Espelhamento Horizontal)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Espelhamento Horizontal)', fontsize=20)
ax2.axis('off')        


# Exemplo de um patch de validação que contenha estrada
image_index = 0

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Rotacionado 90 graus)
image_index = 358

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Rotated 90 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Rotated 90 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Rotacionado 180 graus)
image_index = 716

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Rotated 180 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Rotated 180 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Rotacionado 270 graus)
image_index = 1074

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Rotated 270 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Rotated 270 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Espelhamento Vertical)
image_index = 1432

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Espelhamento Vertical)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Espelhamento Vertical)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Espelhamento Vertical)
image_index = 1790

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Espelhamento Horizontal)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Espelhamento Horizontal)', fontsize=20)
ax2.axis('off')       

# %% Salva dados de treino e validação para poderem ser usados por outro script

x_train = x_train.astype(np.float16)
x_valid = x_valid.astype(np.float16)

salva_arrays(train_test_dir, x_train=x_train, y_train=y_train, 
             x_valid=x_valid, y_valid=y_valid)

# %% Salva dados de teste para serem usados em outro script 

x_test = x_test.astype(np.float16)
y_test = y_test.astype(np.uint8)
#newroads_y_test = newroads_y_test.astype(np.uint8)

#salva_arrays(train_test_dir, x_test=x_test, y_test=y_test, newroads_y_test = newroads_y_test)
salva_arrays(train_test_dir, x_test=x_test, y_test=y_test)


info_tiles_test = {'patch_stride_test':patch_stride, 'len_tiles_test':len_tiles_test, 'shape_tiles_test':shape_tiles_test}

with open(os.path.join(y_dir, 'info_tiles_test.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
    pickle.dump(info_tiles_test, fp)
    
# with open(os.path.join(y_dir, 'len_tiles_test.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
#     pickle.dump(len_tiles_test, fp)
    
# with open(os.path.join(y_dir, 'shape_tiles_test.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
#     pickle.dump(shape_tiles_test, fp)

# %% Salva dados como dataset

import tensorflow as tf
from pathlib import Path

# Path para salvar os datasets
train_test_dir_path = Path(train_test_dir)

# Build Datasets
# Train Dataset
train_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Valid Dataset
valid_dataset= tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

# Test Dataset
test_dataset= tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Save Datasets
# Cria diretório para salvar Datasets
train_dataset_path = train_test_dir_path / 'train_dataset'
train_dataset_path.mkdir()

valid_dataset_path = train_test_dir_path / 'valid_dataset'
valid_dataset_path.mkdir()

test_dataset_path = train_test_dir_path / 'test_dataset'
test_dataset_path.mkdir()

# Save Datasets
train_dataset.save(str(train_dataset_path))
valid_dataset.save(str(valid_dataset_path))
test_dataset.save(str(test_dataset_path))
