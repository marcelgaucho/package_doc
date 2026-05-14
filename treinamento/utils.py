# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:17:47 2025

@author: marce
"""

# Functions used in other modules inside treinamento

# %% Imports

import random
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# %% One-Hot codify an y array

def onehot_numpy(np_array, num_classes=2, ignore_index=255):
    '''
    Parameters
    ----------
    np_array : numpy.ndarray
        Array of labels that Must not contain channels dimension.
        Values must be integers between 0 and n-1, to encode n classes.

    Returns
    -------
    np_array_onehot : numpy.ndarray
        Output will contain channel-last dimension with 
        length equal to number of classes.

    '''
    assert len(np_array.shape) == 4 and np_array.shape[-1] == 1, 'Patches must be in shape (B, H, W, 1)'
    
    np_array = np_array.squeeze(axis=3) # Squeeze patches in last dimension (channel dimension)
    
    # If ignore_index is set, first change ignore_index to the last value and 
    # increment the number of classes
    if ignore_index is not None:
        mask = (np_array == ignore_index)
        np_array = np_array.copy()
        np_array[mask] = num_classes
        num_classes = num_classes + 1  
        
    np_array = np.eye(num_classes, dtype=np.uint8)[np_array] # One-Hot codification

    # Consider classes only up to the ignore index
    if ignore_index is not None:
        np_array = np_array[..., :num_classes-1]
        
    return np_array

# %% Function that build training plot

def show_training_plot(history, metric_name='accuracy', save=False, save_path=r'', save_name='plotagem.png'):
    # Prepare to new function
    history_train = {}
    history_train['loss'] = list(np.array(history[0]).squeeze()[:, 0])
    history_train[metric_name] = list(np.array(history[0]).squeeze()[:, 1])
    history_valid = {}
    history_valid['loss'] = list(np.array(history[1]).squeeze()[:, 0])
    history_valid[metric_name] = list(np.array(history[1]).squeeze()[:, 1])
    
    # Training epochs and steps in x ticks
    total_epochs_training = len(history_train['loss'])
    x_ticks_step = 5
    
    # Create Figure
    plt.figure(figsize=(15,6))
    
    # There are 2 subplots in a row
    
    # X and X ticks for all subplots
    x = list(range(1, total_epochs_training+1)) # Could be length of other parameter too
    x_ticks = list(range(0, total_epochs_training+x_ticks_step, x_ticks_step))
    x_ticks.insert(1, 1)
    
    # First subplot (Metric)
    plt.subplot(1, 2, 1)
    
    plt.title(f'{metric_name} per Epoch', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 19}) # Title, with font name and size
    
    plt.xlabel('Epochs', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) # Label of X axis, with font
    plt.ylabel(f'{metric_name}', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) # Label of X axis, with font
    
    plt.plot(x, history_train[metric_name]) # Plot Train Metric
    plt.plot(x, history_valid[metric_name]) # Plot Valid Metric
    
    plt.ylim(bottom=0, top=1) # Set y=0 on horizontal axis, and for maximum y=1
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # Set y ticks
    plt.xticks(x_ticks) # Set x ticks
    
    plt.legend(['Train', 'Valid'], loc='upper left', fontsize=12) # Legend, with position and fontsize
    plt.grid(True) # Create grid
    
    # Second subplot (Loss)
    plt.subplot(1, 2, 2)
    
    plt.title('Loss per Epoch', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 19})
     
    plt.xlabel('Epochs', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) 
    plt.ylabel('Loss', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) 
    
    plt.plot(x, history_train['loss'])
    plt.plot(x, history_valid['loss'])
    
    
    plt.ylim(bottom=0, top=1) 
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(x_ticks)
    
    plt.legend(['Train', 'Valid'], loc='upper right', fontsize=12)
    plt.grid(True) 
    
    # Adjust layout
    plt.tight_layout()
    
    # Show or save plot
    if save:
        plt.savefig(save_path / save_name)
        plt.close()
    else:
        plt.show(block=False)
        
# %% Options for operations

# =============================================================================
# 0. Mantém original (maintain)                                                # x
# 1. Espelhamento Vertical (Flip)                                              # tf.image.flip_up_down(x) 
# 2. Espelhamento Horizontal (Mirror)                                          # tf.image.flip_left_right(x)
# 3. Rotação 90 graus (90 degrees rotation)                                    # tf.image.rot90(x, k=1)
# 4. Rotação 180 graus (180 degrees rotation)                                  # tf.image.rot90(x, k=2)
# 5. Rotação 270 graus (270 degrees rotation)                                  # tf.image.rot90(x, k=3)
# 6. Espelhamento Vertical e Rotação 90 graus (Flip and 90 degrees rotation)   # tf.image.rot90(tf.image.flip_up_down(x), k=1)
# 7. Espelhamento Vertical e Rotação 270 graus (Flip and 270 degrees rotation) # tf.image.rot90(tf.image.flip_up_down(x), k=3)
# =============================================================================
        
# %% Reference Function to the next optimized ones - Augment or maintain the data by applying a random transformation to a patch and its reference

def transform_augment_or_maintain(x, y):
    # Draw option
    lista_opcoes = [0, 1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide what to do
    # Mantém original
    if opcao == 0:
        return x, y
    # Espelhamento Vertical (Flip)
    elif opcao == 1:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
        return x, y
    # Espelhamento Horizontal (Mirror)
    elif opcao == 2:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
        return x, y
    # Rotação 90 graus
    elif opcao == 3:
        x = tf.image.rot90(x, k=1)
        y = tf.image.rot90(y, k=1)
        return x, y
    # Rotação 180 graus
    elif opcao == 4:
        x = tf.image.rot90(x, k=2)
        y = tf.image.rot90(y, k=2)
        return x, y
    # Rotação 270 graus
    elif opcao == 5:
        x = tf.image.rot90(x, k=3)
        y = tf.image.rot90(y, k=3)
        return x, y
    # Espelhamento Vertical e Rotação 90 graus
    elif opcao == 6:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=1)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=1)
        return x, y
    # Espelhamento Vertical e Rotação 270 graus
    elif opcao == 7:
        x = tf.image.rot90(tf.image.flip_up_down(x), k=3)
        y = tf.image.rot90(tf.image.flip_up_down(y), k=3)
        return x, y
    
# %% Augment the data by applying a random transformation to a patch, its reference and the entropy (Google IA)

@tf.function
def transform_augment_tf_xye(items):
    # vectorized_map expects a single tuple/list of tensors
    x, y, e = items

    # 1 to 7: Excludes 0 (which was your 'stay the same' case)
    opcao = tf.random.uniform([], minval=1, maxval=8, dtype=tf.int32)
    
    # 1. Flip Vertical (opcao 1, 6, 7)
    cond_flip_v = tf.reduce_any([tf.equal(opcao, 1), tf.equal(opcao, 6), tf.equal(opcao, 7)])
    x = tf.where(cond_flip_v, tf.image.flip_up_down(x), x)
    y = tf.where(cond_flip_v, tf.image.flip_up_down(y), y)
    e = tf.where(cond_flip_v, tf.image.flip_up_down(e), e)
    
    # 2. Flip Horizontal (opcao 2)
    cond_flip_h = tf.equal(opcao, 2)
    x = tf.where(cond_flip_h, tf.image.flip_left_right(x), x)
    y = tf.where(cond_flip_h, tf.image.flip_left_right(y), y)
    e = tf.where(cond_flip_h, tf.image.flip_left_right(e), e)
        
    # 3. Rotações
    # k=1 (3,6), k=2 (4), k=3 (5,7)
    k = tf.where(tf.math.logical_or(tf.equal(opcao, 3), tf.equal(opcao, 6)), 1,
        tf.where(tf.equal(opcao, 4), 2,
        tf.where(tf.math.logical_or(tf.equal(opcao, 5), tf.equal(opcao, 7)), 3, 0)))
    
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)
    e = tf.image.rot90(e, k=k)
        
    return x, y, e

@tf.function
def transform_augment_tf_xy(items):
    # vectorized_map expects a single tuple/list of tensors
    x, y = items

    # 1 to 7: Excludes 0 (which was your 'stay the same' case)
    opcao = tf.random.uniform([], minval=1, maxval=8, dtype=tf.int32)
    
    # 1. Flip Vertical (opcao 1, 6, 7)
    cond_flip_v = tf.reduce_any([tf.equal(opcao, 1), tf.equal(opcao, 6), tf.equal(opcao, 7)])
    x = tf.where(cond_flip_v, tf.image.flip_up_down(x), x)
    y = tf.where(cond_flip_v, tf.image.flip_up_down(y), y)
    
    # 2. Flip Horizontal (opcao 2)
    cond_flip_h = tf.equal(opcao, 2)
    x = tf.where(cond_flip_h, tf.image.flip_left_right(x), x)
    y = tf.where(cond_flip_h, tf.image.flip_left_right(y), y)
        
    # 3. Rotações
    # k=1 (3,6), k=2 (4), k=3 (5,7)
    k = tf.where(tf.math.logical_or(tf.equal(opcao, 3), tf.equal(opcao, 6)), 1,
        tf.where(tf.equal(opcao, 4), 2,
        tf.where(tf.math.logical_or(tf.equal(opcao, 5), tf.equal(opcao, 7)), 3, 0)))
    
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)
        
    return x, y



# %% 

@tf.function
def transform_augment_tf(items):
    # vectorized_map expects a single tuple/list of tensors
    x, y, e = items
    
    # Options 1 to 7
    opcao = tf.random.uniform([], minval=1, maxval=8, dtype=tf.int32)
    
    # 1. Flip Vertical (opcao 1, 6, 7) or Flip Horizontal (opcao 2)
    is_flip_v = tf.reduce_any([tf.equal(opcao, 1), tf.equal(opcao, 6), tf.equal(opcao, 7)])
    is_flip_h = tf.equal(opcao, 2)
    
    # 2. Look-up table for rotations
    mapa_k = tf.constant([ 0,  0,  0,  1,  2,  3,  1,  3], dtype=tf.int32)
    k = tf.gather(mapa_k, opcao)

    # 3. Flips using tf.where (required for vectorized_map)
    x = tf.where(is_flip_v, tf.image.flip_up_down(x), x)
    y = tf.where(is_flip_v, tf.image.flip_up_down(y), y)
    e = tf.where(is_flip_v, tf.image.flip_up_down(e), e)

    x = tf.where(is_flip_h, tf.image.flip_left_right(x), x)
    y = tf.where(is_flip_h, tf.image.flip_left_right(y), y)
    e = tf.where(is_flip_h, tf.image.flip_left_right(e), e)

    # 4. Rotação vetorizada
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)
    e = tf.image.rot90(e, k=k)

    return x, y, e


# %% Correction by Google IA

@tf.function
def transform_augment_batch(items):
    """
    Applies unique D4 transformations to each item in a tuple (x, y, and optionally e).
    Uses vectorized_map for batch-level efficiency.
    """
    def _apply_single_sample(tensors):
        # 1 a 7 aleatório para cada amostra do lote
        opcao = tf.random.uniform([], minval=1, maxval=8, dtype=tf.int32)
        
        # Condições booleanas para os flips
        cond_flip_v = tf.reduce_any([tf.equal(opcao, 1), tf.equal(opcao, 6), tf.equal(opcao, 7)])
        cond_flip_h = tf.equal(opcao, 2)
        
        # Correção e Otimização: Tabela de busca direta substituindo os tf.where aninhados
        # Índices mapeados:          0  1  2  3  4  5  6  7
        mapa_k = tf.constant([ 0,  0,  0,  1,  2,  3,  1,  3], dtype=tf.int32)
        k = tf.gather(mapa_k, opcao)

        def transform(t):
            # tf.where puramente matemático obrigatório para o vectorized_map
            t = tf.where(cond_flip_v, tf.image.flip_up_down(t), t)
            t = tf.where(cond_flip_h, tf.image.flip_left_right(t), t)
            return tf.image.rot90(t, k=k)

        return tf.nest.map_structure(transform, tensors)

    return tf.vectorized_map(_apply_single_sample, items)

# %% Made by Gemini


@tf.function
def transform_augment_batch(items):
    """
    Applies unique D4 transformations to each item in a tuple (x, y, and optionally e).
    Uses vectorized_map for batch-level efficiency.
    """
    def _apply_single_sample(tensors):
        # Generate a unique seed for this specific sample in the batch
        opcao = tf.random.uniform([], minval=1, maxval=8, dtype=tf.int32)
        
        cond_flip_v = tf.reduce_any([tf.equal(opcao, 1), tf.equal(opcao, 6), tf.equal(opcao, 7)])
        cond_flip_h = tf.equal(opcao, 2)
        
        k = tf.where(tf.math.logical_or(tf.equal(opcao, 3), tf.equal(opcao, 6)), 1,
            tf.where(tf.equal(opcao, 4), 2,
            tf.where(tf.math.logical_or(tf.equal(opcao, 5), tf.equal(opcao, 7)), 3, 0)))

        def transform(t):
            t = tf.where(cond_flip_v, tf.image.flip_up_down(t), t)
            t = tf.where(cond_flip_h, tf.image.flip_left_right(t), t)
            return tf.image.rot90(t, k=k)

        return tf.nest.map_structure(transform, tensors)

    return tf.vectorized_map(_apply_single_sample, items)