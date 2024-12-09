# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:40:04 2024

@author: Marcel
"""
from osgeo import gdal
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from .pred_functions import Test_Step

from .buffer_functions import array_buffer, buffer_patches_array

from tensorflow.keras import backend as K

import os, shutil, gc, json

from .save_functions import salva_arrays

from sklearn.utils.extmath import stable_cumsum





def compute_relaxed_metrics(y: np.ndarray, pred: np.ndarray, buffer_y: np.ndarray, buffer_pred: np.ndarray,
                            nome_conjunto: str, print_file=None):
    '''
    

    Parameters
    ----------
    y : np.ndarray
        array do formato (batches, heigth, width, channels).
    pred : np.ndarray
        array do formato (batches, heigth, width, channels).
    buffer_y : np.ndarray
        array do formato (batches, heigth, width, channels).
    buffer_pred : np.ndarray
        array do formato (batches, heigth, width, channels).

    Returns
    -------
    relaxed_precision : float
        Relaxed Precision (by 3 pixels).
    relaxed_recall : float
        Relaxed Recall (by 3 pixels).
    relaxed_f1score : float
        Relaxed F1-Score (by 3 pixels).

    '''
    
    
    # Flatten arrays to evaluate quality
    true_labels = np.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2]))
    predicted_labels = np.reshape(pred, (pred.shape[0] * pred.shape[1] * pred.shape[2]))
    
    buffered_true_labels = np.reshape(buffer_y, (buffer_y.shape[0] * buffer_y.shape[1] * buffer_y.shape[2]))
    buffered_predicted_labels = np.reshape(buffer_pred, (buffer_pred.shape[0] * buffer_pred.shape[1] * buffer_pred.shape[2]))
    
    # Calculate Relaxed Precision and Recall for test data
    relaxed_precision = 100*precision_score(buffered_true_labels, predicted_labels, pos_label=1)
    relaxed_recall = 100*recall_score(true_labels, buffered_predicted_labels, pos_label=1)
    relaxed_f1score = 2  *  (relaxed_precision*relaxed_recall) / (relaxed_precision+relaxed_recall)
    
    # Print result
    if print_file:
        print('\nRelaxed Metrics for %s' % (nome_conjunto), file=print_file)
        print('=======', file=print_file)
        print('Relaxed Precision: ', relaxed_precision, file=print_file)
        print('Relaxed Recall: ', relaxed_recall, file=print_file)
        print('Relaxed F1-Score: ', relaxed_f1score, file=print_file)
        print()        
    else:
        print('\nRelaxed Metrics for %s' % (nome_conjunto))
        print('=======')
        print('Relaxed Precision: ', relaxed_precision)
        print('Relaxed Recall: ', relaxed_recall)
        print('Relaxed F1-Score: ', relaxed_f1score)
        print()
    
    return relaxed_precision, relaxed_recall, relaxed_f1score



# Função que calcula algumas métricas: acurácia, f1-score, sensibilidade e precisão 
def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision


# TODO
# To average precision is prob array and not pred array
class RelaxedMetricCalculator:
    def __init__(self, y_array, pred_array=None, buffer_px=3, prob_array=None):
        self.y_array = y_array
        self.pred_array = pred_array
        self.prob_array = prob_array
        
        self.buffer_px = buffer_px
        
        self.buffer_y_array = buffer_patches_array(self.y_array, radius_px=self.buffer_px)
        self.buffer_pred_array = None
        
        self.metrics = None
    
    def _calculate_buffer_pred(self, print_interval):
        self.buffer_pred_array = buffer_patches_array(self.pred_array, radius_px=self.buffer_px, print_interval=print_interval)
        
    def _calculate_prec_recall_f1(self, value_zero_division, print_interval):
        # Give error if prediction isn't set
        assert self.pred_array is not None, "Prediction must be set, please set self.pred_array = array"
        
        # Set buffer of prediction
        self._calculate_buffer_pred(print_interval=print_interval)
        
        # Epsilon (to not divide by 0)
        epsilon = 1e-7
        
        # For relaxed precision (use buffer on Y to calculate true positives)
        true_positive_relaxed_precision = (self.buffer_y_array*self.pred_array).sum()
        predicted_positive = self.pred_array.sum()
        # false_positive_relaxed_precision = predicted_positive - true_positive_relaxed_precision
        
        # For relaxed recall (use buffer on Prediction to calculate true positives)
        true_positive_relaxed_recall = (self.y_array*self.buffer_pred_array).sum() 
        actual_positive = self.y_array.sum()
        # false_negative_relaxed_recall = actual_positive - true_positive_relaxed_recall

        # Calculate relaxed precision and relaxed recall
        relaxed_precision = true_positive_relaxed_precision / (predicted_positive + epsilon)
        relaxed_recall = true_positive_relaxed_recall / (actual_positive + epsilon)
        
        # Special case
        # If there are no actual positives, recall is 1 because "all" the positives will be discovered
        if actual_positive == 0:
            relaxed_recall = 1
                
        # Set the marked value in case of zero division, if it is setted
        if value_zero_division:
            if predicted_positive == 0:
                relaxed_precision = value_zero_division
            if actual_positive == 0:
                relaxed_recall = value_zero_division        
        
        # Calculate relaxed F1 from relaxed precision and relaxed recall
        relaxed_f1 = (2 * relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall + epsilon)
        
        # Dictionary of metrics
        metrics = {'relaxed_precision': relaxed_precision, 'relaxed_recall': relaxed_recall, 'relaxed_f1': relaxed_f1}
        
        return metrics
    
    def _calculate_avg_precision(self, print_interval):
        ''' Calculate Thresholds '''
        # Flatten prediction array and y_true buffer
        prob_flat = self.prob_array.flatten() # maintain original for use in recall calculus
        buffer_y_array_flat = self.buffer_y_array.flatten()
        
        # Sort predictions in descending order of probabilities
        desc_score_indices = np.argsort(prob_flat, kind="mergesort")[::-1]
        prob_flat = prob_flat[desc_score_indices]
        
        # Calculate Thresholds using threshold indexes, which are the change indexes plus the final element index
        # Change indexes are the diff indexes where probability difference with next probability> 0  
        diff_scores = np.diff(prob_flat)
        change_idxs = np.where(diff_scores)[0]
        threshold_idxs = np.r_[change_idxs, prob_flat.size - 1]
        thresholds = prob_flat[threshold_idxs]
        
        ''' Calculate Cumulative TP (for Precision) and Cumulative Predicted Positives for Precision '''
        # Cumulative sum of Trues Positives in the threshold indexes
        # Threshold used in index i is pred_scores[threshold_idxs[i]]
        buffer_y_array_flat = buffer_y_array_flat[desc_score_indices] # sort buffer of y
        cumtp_prec = stable_cumsum(buffer_y_array_flat)
        cumtp_thres_prec = cumtp_prec[threshold_idxs]
        
        # Cumulative (predicted) positives in thresholds
        # Index (1-based) is used as positives, as the predictions array is sorted
        cumpos_thres = threshold_idxs + 1
        
        ''' Calculate Cumulative TP (for Recall) and Actual Positives for Recall '''
        # We can't use the sorted y_true to calculate the TPs in relaxed recall
        # because the TPs depend on the prediction buffer
        
        # Cumulative TP (for thresholds) 
        cumtp_thres_recall = []     
    
        # Loop through thresholds
        for i, threshold in enumerate(thresholds):
            if print_interval:
                if i % print_interval == 0:
                    print(f'Calculating threshold {i:>6d}/{len(thresholds):>6d}')
            
            # Predictions for the current threshold
            pred = (self.prob_array >= threshold).astype(int)
            
            # Buffer for prediction
            pred_buffer = buffer_patches_array(pred, radius_px=self.buffer_px)
            
            # True Positive for Recall
            true_positive_relaxed_recall = (self.y_array * pred_buffer).sum()
            
            # Append value in array
            cumtp_thres_recall.append(true_positive_relaxed_recall)        
    
           
        # Transform in array
        cumtp_thres_recall = np.array(cumtp_thres_recall) 
        
        # Actual positives 
        actual_positive = self.y_array.sum()    
        
        ''' Calculate Precision and Recall '''
        # Initialize and calculate precision if posible (positives != 0)
        # If not posible, precision is set to 0
        precision = np.zeros_like(cumtp_thres_prec)
        np.divide(cumtp_thres_prec, cumpos_thres, out=precision, where=(cumpos_thres != 0))
        
        # Reverse order of precision (Increasing Precision) and append 1 to final
        precision = np.hstack((precision[::-1], 1))
        
        # Calculate recall. Set recall to 1 if there are no positive label in y_true
        if actual_positive == 0:
            print(
                "No positive class found in y_true, "
                "recall is set to one for all thresholds."
            )
            recall = np.ones_like(precision)
        else:
            recall = cumtp_thres_recall / actual_positive
            
        # Reverse order of recall (Decreasing Recall) and append 0 to final
        recall = np.hstack((recall[::-1], 0))
        
        # Reverse order of thresholds
        thresholds = thresholds[::-1] 
    
        # Calculate average precision
        avg_precision = np.sum(-np.diff(recall) * precision[:-1]) 
    
        
        return avg_precision
    
    def calculate_metrics(self, value_zero_division=None, include_avg_precision=False,
                          prob_array=None, print_interval=200):
        # Give error if probabilities isn't set when calculating average precision
        if include_avg_precision:
            assert self.prob_array is not None, ("Probabilities array must be set to calculate average precision, "
                                                 "please set self.prob_array = array")
        
        # Calculate relaxed precision, recall and f1
        metrics = self._calculate_prec_recall_f1(value_zero_division=value_zero_division, print_interval=print_interval)
        
        # Calculate average precision
        if include_avg_precision:
            metrics['relaxed_avg_precision'] = self._calculate_avg_precision(print_interval=print_interval)
            
        # Store metrics variable
        self.metrics = metrics        
            
        return metrics        
        
    def export_results(self, output_dir, group='test'):
        # Check if group is correct
        assert group in ('train', 'valid', 'test', 'mosaics'), "Parameter group must be 'train', 'valid', 'test' or 'mosaics'"
        
        if not hasattr(self, 'metrics'):
            raise Exception("Metrics weren't calculated. First is necessary to "
                            "calculate metrics with the method calculate_metrics")
        
        with open(output_dir + f'relaxed_metrics_{group}_{self.buffer_px}px.json', 'w') as f:
            json.dump(self.metrics, f)
    


# =============================================================================
# class TestRelaxedMetricCalculator(RelaxedMetricCalculator):
#     def export_results(self, output_dir):
#         if not hasattr(self, 'metrics'):
#             raise Exception("Metrics weren't calculated. First is necessary to "
#                             "calculate metrics with the method calculate_metrics")
#         
#         with open(output_dir + f'relaxed_metrics_test_{self.buffer_px}px.json', 'w') as f:
#             json.dump(self.metrics, f)
#             
#             
# class ValidRelaxedMetricCalculator(RelaxedMetricCalculator):
#     def export_results(self, output_dir):
#         if not hasattr(self, 'metrics'):
#             raise Exception("Metrics weren't calculated. First is necessary to "
#                             "calculate metrics with the method calculate_metrics")
#         
#         with open(output_dir + f'relaxed_metrics_valid_{self.buffer_px}px.json', 'w') as f:
#             json.dump(self.metrics, f)
# =============================================================================


def adiciona_prob_0(prob_1):
    ''' Add first channel (background) to a probability array with only road channel (prob_1) '''   
    zeros = np.zeros(prob_1.shape, dtype=np.uint8)
    prob_1 = np.concatenate((zeros, prob_1), axis=-1)
    
    # Percorre dimensão dos batches e adiciona imagens (arrays) que são o complemento
    # da probabilidade de estradas
    for i in range(prob_1.shape[0]):
        prob_1[i, :, :, 0] = 1 - prob_1[i, :, :, 1]
        
    return prob_1


# Função para cálculo da entropia preditiva para um array de probabilidade com várias predições
def calcula_predictive_entropy(prob_array):
    # Calcula probabilidade média    
    prob_mean = np.mean(prob_array, axis=0)
    
    # Calcula Entropia Preditiva
    pred_entropy = np.zeros(prob_mean.shape[0:-1] + (1,)) # Inicia com zeros
    
    K = prob_mean.shape[-1] # número de classes
    epsilon = 1e-7 # usado para evitar log 0
    
    '''
    # Com log base 2 porque tem 2 classes
    for k in range(K):
        pred_entropy_valid = pred_entropy_valid + prob_valid_mean[..., k:k+1] * np.log2(prob_valid_mean[..., k:k+1] + epsilon) 
        
    for k in range(K):
        pred_entropy_train = pred_entropy_train + prob_train_mean[..., k:k+1] * np.log2(prob_train_mean[..., k:k+1] + epsilon) 
    '''    

    # Calculando com log base e e depois escalonando para entre 0 e 1
    for k in range(K):
        pred_entropy = pred_entropy + prob_mean[..., k:k+1] * np.log(prob_mean[..., k:k+1] + epsilon) 
        
    pred_entropy = - pred_entropy / np.log(K) # Escalona entre 0 e 1, já que o máximo da entropia é log(K),
                                              # onde K é o número de classes
                                              
    pred_entropy = np.clip(pred_entropy, 0, 1) # Clipa valores entre 0 e 1

    return pred_entropy  

# Calcula predição a partir da probabilidade média do ensemble
def calcula_pred_from_prob_ensemble_mean(prob_array):
    # Calcula probabilidade média    
    prob_mean = np.mean(prob_array, axis=0)

    # Calcula predição (é a classe que, entre as 2 probabilidades, tem probabilidade maior)
    predicted_classes = np.argmax(prob_mean, axis=-1)[..., np.newaxis] 
    
    return predicted_classes



# https://github.com/tensorflow/tensorflow/issues/6743
def patches_to_images_tf(
    patches: np.ndarray, image_shape: tuple,
    overlap: float = 0.5, stitch_type='average', indices=None) -> np.ndarray:
    """Reconstructs images from patches.

    Args:
        patches (ndarray): Array with batch of patches to convert to batch of images.
            [batch_size, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]
        image_shape (Tuple): Shape of output image. (y, x, n_channels) or (y, x)
        overlap (float, optional): Overlap factor between patches. Defaults to 0.5.
        stitch_type (str, optional): Type of stitching to use. Defaults to 'average'.
            Options: 'average', 'replace'.
        indices (ndarray, optional): Indices of patches in image. Defaults to None.
            If provided, indices are used to stitch patches together and not recomputed
            to save time. Has same shape as patches shape but with added index axis (last).
    Returns:
        images (ndarray): Reconstructed batch of images from batch of patches.

    """
    assert len(image_shape) == 3, 'image_shape should have 3 dimensions, namely: ' \
        '(#image_y, #image_x, (n_channels))'
    assert len(patches.shape) == 6 , 'patches should have 6 dimensions, namely: ' \
        '[batch, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]'
    assert overlap >= 0 and overlap < 1, 'overlap should be between 0 and 1'
    assert stitch_type in ['average', 'replace'], 'stitch_type should be either ' \
        '"average" or "replace"'

    batch_size, n_patches_y, n_patches_x, *patch_shape = patches.shape
    n_channels = image_shape[-1]
    dtype = patches.dtype

    assert len(patch_shape) == len(image_shape)

    # Kernel for counting overlaps
    if stitch_type == 'average':
        kernel_ones = tf.ones((batch_size, n_patches_y, n_patches_x, *patch_shape), dtype=tf.int32)
        mask = tf.zeros((batch_size, *image_shape), dtype=tf.int32)

    if indices is None:
        if overlap:
            nonoverlap = np.array([1 - overlap, 1 - overlap, 1 / patch_shape[-1]])
            stride = (np.array(patch_shape) * nonoverlap).astype(int)
        else:
            stride = (np.array(patch_shape)).astype(int)

        channel_idx = tf.reshape(tf.range(n_channels), (1, 1, 1, 1, 1, n_channels, 1))
        channel_idx = (tf.ones((batch_size, n_patches_y, n_patches_x, *patch_shape, 1), dtype=tf.int32) * channel_idx)

        batch_idx = tf.reshape(tf.range(batch_size), (batch_size, 1, 1, 1, 1, 1, 1))
        batch_idx = (tf.ones((batch_size, n_patches_y, n_patches_x, *patch_shape, 1), dtype=tf.int32) * batch_idx)

        # TODO: create indices without looping possibly
        indices = []
        for j in range(n_patches_y):
            for i in range(n_patches_x):
                # Make indices from meshgrid
                _indices = tf.meshgrid(
                    tf.range(stride[0] * j, # row start
                            patch_shape[0] + stride[0] * j), # row end
                    tf.range(stride[1] * i, # col_start
                            patch_shape[1] + stride[1] * i), indexing='ij') # col_end

                _indices = tf.stack(_indices, axis=-1)
                indices.append(_indices)

        indices = tf.reshape(tf.stack(indices, axis=0), (n_patches_y, n_patches_x, *patch_shape[:2], 2))

        indices = tf.repeat(indices[tf.newaxis, ...], batch_size, axis=0)
        indices = tf.repeat(indices[..., tf.newaxis, :], n_channels, axis=-2)

        indices = tf.concat([batch_idx, indices, channel_idx], axis=-1)

    # create output image tensor
    images = tf.zeros([batch_size, *image_shape], dtype=dtype)

    # Add sliced image to recovered image indices
    if stitch_type == 'replace':
        images = tf.tensor_scatter_nd_update(images, indices, patches)
    else:
        images = tf.tensor_scatter_nd_add(images, indices, patches)
        mask = tf.tensor_scatter_nd_add(mask, indices, kernel_ones)
        images = tf.cast(images, tf.float32) / tf.cast(mask, tf.float32)

    return images, indices





def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]








def blur_x_patches(x_train, y_train, dim, k, blur, model):
    '''
    Faz o blur, em cada patch, em k subpaches. Preferencialmente naqueles que contêm pelo menos um pixel de estrada.
    Caso não houver subpatches suficientes subpatches com pelos um pixel de estrada, então faz o blur naqueles que não 
    tem estrada mesmo.

    Parameters
    ----------
    x_train : TYPE
        Numpy array (patches, height, width, channels).
    y_train : TYPE
        Numpy array (patches, height, width, channels).
    dim : TYPE
        Tamanho do subpatch
    k : TYPE
        Quantos subpatches aplica o blur.
    blur : TYPE
        Desvio Padrão do blur.

    Returns
    -------
    x_train_blur : TYPE
        x_train com blur aplicado.
    pred_train_blur : TYPE
        predições de x_train com blur aplicado.
    '''
    
    # Tamanho do patch e número de canais dos patches e número de patches e Quantidade de pixels em um subpatch
    patch_size = x_train.shape[1] # Patch Quadrado
    n_channels = x_train.shape[-1]
    n_patches = x_train.shape[0]
    pixels_patch = patch_size*patch_size
    pixels_subpatch = dim*dim

    if k > pixels_patch//pixels_subpatch:
        raise Exception('k é maior que o número total de subpatches com o tamanho subpatch_size. '
                        'Diminua o tamanho do k ou diminua o tamanho do subpatch')
        
    # Variáveis para extração dos subpatches (tiles) a partir dos patches
    sizes = [1, dim, dim, 1]
    strides = [1, dim, dim, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    
    # Extrai subpatches e faz o reshape para estar 
    # na forma (patch, número total de subpatches, altura do subpatch, 
    # largura do subpatch, número de canais do subpatch)
    # Na forma como é extraído, o subpatch fica achatado na forma
    # (patch, número vertical de subpatches, número horizontal de subpatches, número de pixels do subpatch achatado)
    subpatches_x_train = tf.image.extract_patches(x_train, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()
    n_vertical_subpatches = subpatches_x_train.shape[1]
    n_horizontal_subpatches = subpatches_x_train.shape[2]
    subpatches_x_train = subpatches_x_train.reshape((n_patches, n_vertical_subpatches*n_horizontal_subpatches, 
                                                     dim, dim, n_channels))
    
    subpatches_y_train = tf.image.extract_patches(y_train, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()
    subpatches_y_train = subpatches_y_train.reshape((n_patches, n_vertical_subpatches*n_horizontal_subpatches, 
                                                     dim, dim, 1)) # Só um canal para referência
    

    
    # Serão selecionados, preferencialmente, subpatches com número de pixels de estrada maior que pixels_corte
    pixels_corte = 0

    # Aplica blur nos respectivos subpatches    
    for i_patch in range(len(subpatches_y_train)):
        # Conta número de pixels de estrada em cada subpatch do presente patch
        contagem_estrada = np.array([np.count_nonzero(subpatches_y_train[i_patch, i_subpatch] == 1) 
                                     for i_subpatch in range(subpatches_y_train.shape[1])])
        
        # Array com índices dos subpatches em ordem decrescente de número de pixels de estrada
        indices_sorted_desc = np.argsort(contagem_estrada)[::-1]
        
        # Deixa somente os indices cujos subpatches tem número de pixels maior que o limiar pixels_corte
        indices_maior = np.array([indice for indice in indices_sorted_desc 
                                  if contagem_estrada[indice] > pixels_corte])
        
        # Pega os k subpatches que tem mais pixels de estrada e 
        # que tem quantidade de pixels de estrada maior que 0
        indices_selected_subpatches = indices_maior[:k]
        
        # Converte em lista e coloca em ordem crescente
        indices_selected_subpatches = list(indices_selected_subpatches)
        indices_selected_subpatches.sort()
            
        # Cria array com subpatches escolhidos
        selected_subpatches_x_train_i_patch = subpatches_x_train[i_patch][indices_selected_subpatches]
        
        # Aplica filtro aos subpatches 
        selected_subpatches_x_train_i_patch_blurred = gaussian_filter(selected_subpatches_x_train_i_patch.astype(np.float32), sigma=(0, blur, blur, 0))
        
        # Substitui subpatches pelos respectivos subpatches com blur no array original de subpatches
        subpatches_x_train[i_patch][indices_selected_subpatches] = selected_subpatches_x_train_i_patch_blurred
        
    
    # Agora faz reconstituição dos patches originais, com o devido blur nos subpatches
    # Coloca no formato aceito da função de reconstituição
    x_train_blur = np.zeros(x_train.shape, dtype=np.float16)
    
    for i_patch in range(len(subpatches_x_train)):
        # Pega subpatches do patch
        sub = subpatches_x_train[i_patch]
    
        # Divide em linhas    
        rows = np.split(sub, patch_size//dim, axis=0)
        
        # Concatena linhas
        rows = [tf.concat(tf.unstack(x), axis=1).numpy() for x in rows]
        
        # Reconstroi
        reconstructed = (tf.concat(rows, axis=0)).numpy()
        
        # Atribui
        x_train_blur[i_patch] = reconstructed
    
    # Make predictions
    pred_train_blur, _ = Test_Step(model, x_train_blur)

    # Retorna patches com blur e predições
    return x_train_blur, pred_train_blur
    


def occlude_y_patches(y_train, dim, k):
    # Tamanho do patch e número de canais dos patches e número de patches e Quantidade de pixels em um subpatch
    patch_size = y_train.shape[1] # Patch Quadrado
    n_channels = 1 # Só um canal, pois é máscara
    n_patches = y_train.shape[0]
    pixels_patch = patch_size*patch_size
    pixels_subpatch = dim*dim
    
    if k > pixels_patch//pixels_subpatch:
        raise Exception('k é maior que o número total de subpatches com o tamanho subpatch_size. '
                        'Diminua o tamanho do k ou diminua o tamanho do subpatch')
        
    # Variáveis para extração dos subpatches (tiles) a partir dos patches
    sizes = [1, dim, dim, 1]
    strides = [1, dim, dim, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    
    
    # Extrai subpatches e faz o reshape para estar 
    # na forma (patch, número total de subpatches, altura do subpatch, 
    # largura do subpatch, número de canais do subpatch)
    # Na forma como é extraído, o subpatch fica achatado na forma
    # (patch, número vertical de subpatches, número horizontal de subpatches, número de pixels do subpatch achatado)
    subpatches_y_train = tf.image.extract_patches(y_train, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()
    n_vertical_subpatches = subpatches_y_train.shape[1]
    n_horizontal_subpatches = subpatches_y_train.shape[2]
    subpatches_y_train = subpatches_y_train.reshape((n_patches, n_vertical_subpatches*n_horizontal_subpatches, 
                                                     dim, dim, n_channels)) # Só um canal para referência
    
    
    # Serão selecionados, preferencialmente, subpatches com número de pixels de estrada maior que pixels_corte
    pixels_corte = 0
    
    # Aplica blur nos respectivos subpatches    
    for i_patch in range(len(subpatches_y_train)):
        # Conta número de pixels de estrada em cada subpatch do presente patch
        contagem_estrada = np.array([np.count_nonzero(subpatches_y_train[i_patch, i_subpatch] == 1) 
                                     for i_subpatch in range(subpatches_y_train.shape[1])])
        
        # Array com índices dos subpatches em ordem decrescente de número de pixels de estrada
        indices_sorted_desc = np.argsort(contagem_estrada)[::-1]
        
        # Deixa somente os indices cujos subpatches tem número de pixels maior que o limiar pixels_corte
        indices_maior = np.array([indice for indice in indices_sorted_desc 
                                  if contagem_estrada[indice] > pixels_corte])
        # indices_maior = indices_sorted_desc[contagem_estrada > pixels_corte]
        
        # Pega os k subpatches que tem mais pixels de estrada e 
        # que tem quantidade de pixels de estrada maior que 0
        indices_selected_subpatches = indices_maior[:k]
        
        # Converte em lista e coloca em ordem crescente
        indices_selected_subpatches = list(indices_selected_subpatches)
        indices_selected_subpatches.sort()
            
        # Cria array com subpatches escolhidos
        selected_subpatches_y_train_i_patch = subpatches_y_train[i_patch][indices_selected_subpatches]
        
        # Aplica filtro aos subpatches 
        selected_subpatches_y_train_i_patch_occluded = np.zeros(selected_subpatches_y_train_i_patch.shape)
        
        # Substitui subpatches pelos respectivos subpatches com blur no array original de subpatches
        subpatches_y_train[i_patch][indices_selected_subpatches] = selected_subpatches_y_train_i_patch_occluded
        
    
    # Agora faz reconstituição dos patches originais, com o devido blur nos subpatches
    # Coloca no formato aceito da função de reconstituição
    y_train_occluded = np.zeros(y_train.shape, dtype=np.uint8)
    
    for i_patch in range(len(subpatches_y_train)):
        # Pega subpatches do patch
        sub = subpatches_y_train[i_patch]
    
        # Divide em linhas    
        rows = np.split(sub, patch_size//dim, axis=0)
        
        # Concatena linhas
        rows = [tf.concat(tf.unstack(x), axis=1).numpy() for x in rows]
        
        # Reconstroi
        reconstructed = (tf.concat(rows, axis=0)).numpy()
        
        # Atribui
        y_train_occluded[i_patch] = reconstructed
        
    return y_train_occluded
    
    

# intersection between line(p1, p2) and line(p3, p4)
# Extraído de https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)



def extract_difference_reftiles(tile_before: str, tile_after: str, out_raster_path:str, buffer_px: int=3):
    '''
    Extrai diferença entre tiles de referência de diferentes épocas. 
    Ela é computada assim: Tile Depois - Buffer(Tile Antes).
    
    Exemplo: extract_difference_reftiles('tiles\\masks\\2016\\test\\reftile_2016_15.tif', 
                                         'tiles\\masks\\2018\\test\\reftile_2018_15.tif',
                                         'tiles\\masks\\Diff\\test\\diff_reftile_2018_2016_15.tif')

    Parameters
    ----------
    tile_before : str
        DESCRIPTION.
    tile_after : str
        DESCRIPTION.
    out_raster_path : str
        DESCRIPTION.
    buffer_px : int, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    None, but save the tile difference as raster .tif in the out_raster_path.

    '''
    # Open Images with GDAL
    gdal_header_tile_before = gdal.Open(str(tile_before))
    gdal_header_tile_after = gdal.Open(str(tile_after)) 
    
    # X and Y size of image
    xsize = gdal_header_tile_before.RasterXSize
    ysize = gdal_header_tile_after.RasterYSize
    
    # Read as Rasters as Numpy Array
    tile_before_arr = gdal_header_tile_before.ReadAsArray()
    tile_after_arr = gdal_header_tile_after.ReadAsArray()
    
    # Replace NODATA with 0s to make calculations
    # Replace only Array After because the Buffer of Array Before already eliminates Nodata
    tile_after_arr_0 = tile_after_arr.copy()
    tile_after_arr_0[tile_after_arr_0==255] = 0
    
    # Make Buffer on Image Before and Subtract it from Image After
    # Calculate Difference: Tile After - Buffer(Tile Before)
    # Consider only Positive Values (New Roads)
    dist_buffer_px = 3
    buffer_before = array_buffer(tile_before_arr, dist_buffer_px)
    diff_after_before = np.clip(tile_after_arr_0.astype(np.int16) - buffer_before.astype(np.int16), 0, 1)
    diff_after_before = diff_after_before.astype(np.uint8)
    diff_after_before[tile_after_arr==255] = 0 # Set Original Nodata values to 0, output doesn't contain Nodata
    
    # Export Raster
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver
    file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Byte) # Tipo Unsigned Int 8 bits
    
    file_band = file.GetRasterBand(1) 
    file_band.WriteArray(diff_after_before)
    
    file.SetGeoTransform(gdal_header_tile_after.GetGeoTransform())
    file.SetProjection(gdal_header_tile_after.GetProjection())    
    
    file.FlushCache()


# Etapa 1 se refere ao processamento no qual a entrada do treinamento do pós-processamento é a predição do modelo
# Etapa 2 ao processamento no qual a entrada do treinamento do pós-processamento é a imagem original mais a predição do modelo
# Etapa 3 ao processamento no qual a entrada do treinamento do pós-processamento é o rótulo degradado
# Etapa 5 se refere ao pós-processamento
def gera_dados_segunda_rede(input_dir, y_dir, output_dir, etapa=2):
    x_train = np.load(input_dir + 'x_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    
    y_train = np.load(y_dir + 'y_train.npy')
    y_valid = np.load(y_dir + 'y_valid.npy')
    
    # Copia y_train e y_valid para diretório de saída, pois eles são mantidos como rótulos da Segunda Rede
    # para o pós-processamento
    # if etapa==1 or etapa==2 or etapa==3 or etapa==4:
    #     shutil.copy(input_dir + 'y_train.npy', output_dir + 'y_train.npy')
    #     shutil.copy(input_dir + 'y_valid.npy', output_dir + 'y_valid.npy')
        
    # Predições
    # if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')):
    #     # Nome do modelo salvo
    #     best_model_filename = 'best_model'
        
    #     # Load model
    #     model = load_model(output_dir + best_model_filename + '.keras', compile=False)        
        
    #     # Faz predição
    #     pred_train, _ = Test_Step(model, x_train, 2)
        
    #     # Converte para tipo que ocupa menos espaço e salva
    #     pred_train = pred_train.astype(np.uint8)
    #     salva_arrays(output_dir, pred_train=pred_train)
    
    if etapa != 3:
        pred_train = np.load(output_dir + 'pred_train.npy')
        pred_valid = np.load(output_dir + 'pred_valid.npy')
        
    # Forma e Copia para Nova Pasta o Novo x_train e x_valid
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            shutil.copy(output_dir + 'pred_train.npy', output_dir + 'x_train.npy')
            shutil.copy(output_dir + 'pred_valid.npy', output_dir + 'x_valid.npy')
    
    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            x_train_new = np.concatenate((x_train, pred_train), axis=-1)
            x_valid_new = np.concatenate((x_valid, pred_valid), axis=-1)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
            
    if etapa == 3:
        dim = 14
        k = 10
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            x_train_new = occlude_y_patches(y_train, dim, k)
            x_valid_new = occlude_y_patches(y_valid, dim, k)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
            
    # Libera memória
    if all(pred in locals() for pred in ('pred_train', 'pred_valid')):
        del x_train, x_valid, y_train, y_valid, pred_train, pred_valid
    else:
        del x_train, x_valid, y_train, y_valid
        
    gc.collect()
            
            
    # Lê arrays referentes aos patches de teste
    x_test = np.load(input_dir + 'x_test.npy')
    
    
    # Copia y_test para diretório de saída, pois ele será usado como entrada (rótulo) para o pós-processamento
    # if etapa==1 or etapa==2 or etapa==3 or etapa==4:
    #     shutil.copy(input_dir + 'y_test.npy', output_dir + 'y_test.npy')
        
    # Predições
    pred_test = np.load(output_dir + 'pred_test.npy')
        
    # Forma e Copia para Nova Pasta o Novo x_test
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            shutil.copy(output_dir + 'pred_test.npy', output_dir + 'x_test.npy')

    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            x_test_new = np.concatenate((x_test, pred_test), axis=-1)
            salva_arrays(output_dir, x_test=x_test_new)
            
    if etapa == 3: # No caso o X teste será igual ao da etapa 1
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            shutil.copy(output_dir + 'pred_test.npy', output_dir + 'x_test.npy')
            
    # Libera memória
    del x_test, pred_test
    gc.collect()
    
    
# Source: https://stackoverflow.com/questions/44951624/numpy-stack-with-unequal-shapes
def stack_uneven(arrays, fill_value=0):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result