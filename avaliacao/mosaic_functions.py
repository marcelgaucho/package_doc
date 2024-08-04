# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:44:56 2024

@author: Marcel
"""

import numpy as np
import os
from .save_functions import save_raster_reference

# Calcula os limites em x e y do patch, para ser usado no caso de um patch de borda
def calculate_xp_yp_limits(p_size, x, y, xmax, ymax, left_half, up_half, right_shift, down_shift):
    # y == 0
    if y == 0:
        if x == 0:
            if y + p_size >= ymax:
                yp_limit = ymax - y 
            else:
                yp_limit = p_size
                
            if x + p_size >= xmax:
                xp_limit = xmax - x
            else:
                xp_limit = p_size
                
        else:
            if y + p_size >= ymax:
                yp_limit = ymax - y
            else:
                yp_limit = p_size
                
            if x + right_shift >= xmax:
                xp_limit = xmax - x + left_half
            else:
                xp_limit = p_size
    # y != 0
    else:
        if x == 0:
            if y + down_shift >= ymax:
                yp_limit = ymax - y + up_half
            else:
                yp_limit = p_size
                
            if x + p_size >= xmax:
                xp_limit = xmax - x
            else:
                xp_limit = p_size
        
        else:
            if y + down_shift >= ymax:
                yp_limit = ymax - y + up_half
            else:
                yp_limit = p_size
                
            if x + right_shift >= xmax:
                xp_limit = xmax - x + left_half
            else:
                xp_limit = p_size

    return xp_limit, yp_limit

# Cria o mosaico a partir dos batches extraídos da Imagem de Teste
def unpatch_reference(reference_batches, stride, reference_shape, border_patches=False):
    '''
    Function: unpatch_reference
    -------------------------
    Unpatch the patches of a reference to form a mosaic
    
    Input parameters:
      reference_batches  = array containing the batches (batches, h, w)
      stride = stride used to extract the patches
      reference_shape = shape of the target mosaic. Is the same shape from the labels image. (h, w)
      border_patches = include patches overlaping image borders, as extracted with the function extract_patches
    
    Returns: 
      mosaic = array in shape (h, w) with the mosaic build from the patches.
               Array is builded so that, in overlap part, half of the patch is from the left patch,
               and the other half is from the right patch. This is done to lessen border effects.
               This happens also with the patches in the vertical.
    '''
    
    # Objetivo é reconstruir a imagem de forma que, na parte da sobreposição dos patches,
    # metade fique a cargo do patch da esquerda e a outra metade, a cargo do patch da direita.
    # Isso é para diminuir o efeito das bordas

    # Trabalhando com patch quadrado (reference_batches.shape[1] == reference_batches.shape[2])
    # ovelap e notoverlap são o comprimento de pixels dos patches que tem ou não sobreposição 
    # Isso depende do stride com o qual os patches foram extraídos
    p_size = reference_batches.shape[1]
    overlap = p_size - stride
    notoverlap = stride
    
    # Cálculo de metade da sobreposição. Cada patch receberá metade da área de sobreposição
    # Se número for fracionário, a metade esquerda será diferente da metade direita
    half_overlap = overlap/2
    left_half = round(half_overlap)
    right_half = overlap - left_half
    up_half = left_half
    down_half = right_half
    
    # É o quanto será determinado, em pixels, pelo primeiro patch (patch da esquerda),
    # já que metade da parte de sobreposição
    # (a metade da direita) é desprezada
    # right_shift é o quanto é determinado pelo patch da direita
    # up_shift é o análogo da esquerda na vertical, portanto seria o patch de cima,
    # enquanto down_shift é o análogo da direita na vertical, portanto seria o patch de baixo
    left_shift = notoverlap + left_half
    right_shift = notoverlap + right_half
    up_shift = left_shift 
    down_shift = right_shift
    
    
    # Cria mosaico que será usado para escrever saída
    # Mosaico tem mesmas dimensões da referência usada para extração 
    # dos patches
    pred_test_mosaic = np.zeros(reference_shape)  

    # Dimensões máximas na vertical e horizontal
    ymax, xmax = pred_test_mosaic.shape
    
    # Linha e coluna de referência que serão atualizados no loop
    y = 0 
    x = 0 
    
    # Loop para escrever o mosaico    
    for patch in reference_batches:
        # Parte do mosaico a ser atualizada (debug)
        mosaico_parte_atualizada = pred_test_mosaic[y : y + p_size, x : x + p_size]
        print(mosaico_parte_atualizada)
        
        # Se reconstituímos os patches de borda (border_patches=True)
        # e se o patch é um patch que transborda a imagem, então vamos usar apenas
        # o espaço necessário no patch, com demarcação final por yp_limit e xp_limit
        if border_patches:
            xp_limit, yp_limit = calculate_xp_yp_limits(p_size, x, y, xmax, ymax, left_half, up_half, right_shift, down_shift)
        
        else:
            # Patches sempre são considerados por inteiro, pois não há patches de borda (border_patches=False) 
            yp_limit = p_size
            xp_limit = p_size
            
                    
                    
        # Se é primeiro patch, então vai usar todo patch
        # Do segundo patch em diante, vai ser usado a parte direita do patch, 
        # sobreescrevendo o patch anterior na área correspondente
        # Isso também acontece para os patches na vertical, sendo que o de baixo sobreescreverá o de cima
        if y == 0:
            if x == 0:
                pred_test_mosaic[y : y + p_size, x : x + p_size] = patch[0 : yp_limit, 0 : xp_limit]
            else:
                pred_test_mosaic[y : y + p_size, x : x + right_shift] = patch[0 : yp_limit, left_half : xp_limit]
        # y != 0
        else:
            if x == 0:
                pred_test_mosaic[y : y + down_shift, x : x + p_size] = patch[up_half : yp_limit, 0 : xp_limit]
            else:
                pred_test_mosaic[y : y + down_shift, x : x + right_shift] = patch[up_half : yp_limit, left_half : xp_limit]
            
            
        print(pred_test_mosaic) # debug
        
        
        # Incrementa linha de referência, de forma análoga ao incremento da coluna, se ela já foi esgotada 
        
        # No caso de não haver patches de bordas, é preciso testar se o próximo patch ultrapassará a borda 
        if not border_patches:
            if x == 0 and x + left_shift + right_shift > xmax:
                x = 0
        
                if y == 0:
                    y = y + up_shift
                else:
                    y = y + notoverlap
        
                continue
            
            else:
                if x + right_shift + right_shift > xmax:
                    x = 0
            
                    if y == 0:
                        y = y + up_shift
                    else:
                        y = y + notoverlap
            
                    continue
                
        
        
        if x == 0 and x + left_shift >= xmax:
            x = 0
            
            if y == 0:
                y = y + up_shift
            else:
                y = y + notoverlap
            
            continue
            
        elif x + right_shift >= xmax:
            x = 0
            
            if y == 0:
                y = y + up_shift
            else:
                y = y + notoverlap
            
            continue
        
        # Incrementa coluna de referência
        # Se ela for o primeiro patch (o mais a esquerda), então será considerada como patch da esquerda
        # Do segundo patch em diante, eles serão considerados como patch da direita
        if x == 0:
            x = x + left_shift
        else:
            x = x + notoverlap
            
    
    return pred_test_mosaic


# pred_array é o array de predição e labels_paths é uma lista com o nome dos caminhos dos tiles
# que deram origem aos patches e que serão usados para informações de georreferência
# patch_test_stride é o stride com o qual foram extraídos os patches, necessário para a reconstrução dos tiles
# labels_test_shape é a dimensão de cada tile, também necessário para suas reconstruções
# len_tiles_test é a quantidade de patches por tile
def gera_mosaicos(output_dir, pred_array, labels_paths, prefix='outmosaic', patch_test_stride=96, labels_test_shape=(1408, 1280), len_tiles_test=[], is_float=False):
    # Stride e Dimensões do Tile
    patch_test_stride = patch_test_stride # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches
    labels_test_shape = labels_test_shape # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
    
    # Lista com os mosaicos previstos
    pred_test_mosaic_list = []
    
    # n_test_tiles = n_test_tiles
    n_test_tiles = len(len_tiles_test)
    # n_patches_tile = int(pred_array.shape[0]/n_test_tiles) # Quantidade de patches por tile. Supõe tiles com igual número de patches
    n = len(pred_array) # Quantidade de patches totais, contando todos os tiles, é igual ao comprimento 
    
    i = 0 # Índice onde o tile começa
    i_mosaic = 0 # Índice que se refere ao número do mosaico (primeiro, segundo, terceiro, ...)
    
    while i < n:
        print('Making Mosaic {}/{}'.format(i_mosaic+1, n_test_tiles ))
        
        pred_test_mosaic = unpatch_reference(pred_array[i:i+len_tiles_test[i_mosaic], ..., 0], patch_test_stride, labels_test_shape[i_mosaic], border_patches=True)
        
        pred_test_mosaic_list.append(pred_test_mosaic) 
        
        # Incremento
        i = i+len_tiles_test[i_mosaic]
        i_mosaic += 1  
        
        
    # Salva mosaicos
    labels_paths = labels_paths
    output_dir = output_dir
    
    for i in range(len(pred_test_mosaic_list)):
        pred_test_mosaic = pred_test_mosaic_list[i]
        labels_path = labels_paths[i]
        
        filename_wo_ext = labels_path.split(os.path.sep)[-1].split('.tif')[0]
        '''
        tile_line = int(filename_wo_ext.split('_')[1])
        tile_col = int(filename_wo_ext.split('_')[2])
        
        out_mosaic_name = prefix + '_' + str(tile_line) + '_' + str(tile_col) + r'.tif'
        '''
        out_mosaic_name = prefix + '_' + filename_wo_ext + r'.tif'
        out_mosaic_path = os.path.join(output_dir, out_mosaic_name)
        
        if is_float:
            save_raster_reference(labels_path, out_mosaic_path, pred_test_mosaic, is_float=True)
        else:
            save_raster_reference(labels_path, out_mosaic_path, pred_test_mosaic, is_float=False)
    
            
            
    # Retorna lista de mosaicos
    return pred_test_mosaic_list