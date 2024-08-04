# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:39:30 2024

@author: Marcel
"""

from osgeo import gdal
import pickle
from ..treinamento.plot_training import show_training_plot

from ..treinamento.arquiteturas.unetr_2d import Patches
from ..treinamento.arquiteturas.segformer_tf_k2.models.modules import MixVisionTransformer
from ..treinamento.arquiteturas.segformer_tf_k2.models.Head import SegFormerHead
from ..treinamento.arquiteturas.segformer_tf_k2.models.utils import ResizeLayer

from tensorflow.keras.models import load_model

import numpy as np
from scipy.ndimage import gaussian_filter

import os, gc, shutil
from pathlib import Path

import tensorflow as tf

from transformers import TFSegformerForSemanticSegmentation


from .pred_functions import Test_Step, Test_Step_Ensemble

from .save_functions import salva_arrays

from .buffer_functions import buffer_patches

from .compute_functions import (compute_relaxed_metrics, stack_uneven, 
                               extract_difference_reftiles,
                               calcula_pred_from_prob_ensemble_mean)

from .mosaic_functions import gera_mosaicos





# Avalia um modelo segundo conjuntos de treino, validação, teste e mosaicos de teste
def avalia_modelo(input_dir: str, y_dir: str, output_dir: str, metric_name = 'F1-Score', 
                  dist_buffers=[3], avalia_train=False, avalia_diff=False, avalia_ate_teste=False):
    metric_name = metric_name
    dist_buffers = dist_buffers
    
    # Nome do modelo salvo
    best_model_filename = 'best_model'
    
    # Lê histórico
    with open(output_dir + 'history_pickle_' + best_model_filename + '.pickle', "rb") as fp:   
        history = pickle.load(fp)
    
    # Mostra histórico em gráfico
    show_training_plot(history, metric_name = metric_name, save=True, save_path=output_dir)

    # Load model
    model = load_model(output_dir + best_model_filename + '.keras', compile=False, custom_objects={"Patches": Patches, 
                                                                                                   "MixVisionTransformer": MixVisionTransformer,
                                                                                                   "SegFormerHead": SegFormerHead,
                                                                                                   "ResizeLayer": ResizeLayer})

    # Avalia treino    
    if avalia_train:
        x_train = np.load(input_dir + 'x_train.npy')
        y_train = np.load(y_dir + 'y_train.npy')

        if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')) or \
           not os.path.exists(os.path.join(output_dir, 'prob_train.npy')): 
    
            pred_train, prob_train = Test_Step(model, x_train, 2)
            
            # Probabilidade apenas da classe 1 (de estradas)
            prob_train = prob_train[..., 1:2] 
            
            # Converte para tipos que ocupam menos espaço
            pred_train = pred_train.astype(np.uint8)
            prob_train = prob_train.astype(np.float16)

            # Salva arrays de predição do Treinamento e Validação    
            salva_arrays(output_dir, pred_train=pred_train, prob_train=prob_train)
        
        pred_train = np.load(output_dir + 'pred_train.npy')
        prob_train = np.load(output_dir + 'prob_train.npy')

        # Faz os Buffers necessários, para treino, nas imagens 
        
        # Precisão Relaxada - Buffer na imagem de rótulos
        # Sensibilidade Relaxada - Buffer na imagem extraída
        # F1-Score Relaxado - É obtido através da Precisão e Sensibilidade Relaxadas
        
        buffers_y_train = {}
        buffers_pred_train = {}
    
        for dist in dist_buffers:
            # Buffers para Precisão Relaxada
            if not os.path.exists(os.path.join(y_dir, f'buffer_y_train_{dist}px.npy')): 
                buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
                np.save(y_dir + f'buffer_y_train_{dist}px.npy', buffers_y_train[dist])
                
            # Buffers para Sensibilidade Relaxada
            if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}px.npy')):
                buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
                np.save(output_dir + f'buffer_pred_train_{dist}px.npy', buffers_pred_train[dist])
            

        # Lê buffers de arrays de predição do Treinamento
        for dist in dist_buffers:
            buffers_y_train[dist] = np.load(y_dir + f'buffer_y_train_{dist}px.npy')            
            buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}px.npy')
  
        
        # Relaxed Metrics for training
        relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    
    
        for dist in dist_buffers:
            with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
                relaxed_precision_train[dist], relaxed_recall_train[dist], relaxed_f1score_train[dist] = compute_relaxed_metrics(y_train, 
                                                                                                                                     pred_train, buffers_y_train[dist],
                                                                                                                                     buffers_pred_train[dist], 
                                                                                                                                     nome_conjunto = 'Treino', 
                                                                                                                                     print_file=f)        
           
    
        # Release Memory
        x_train = None
        y_train = None
        pred_train = None
        prob_train = None
        
        buffers_y_train = None
        buffers_pred_train = None
            
        gc.collect()
    
    # Avalia Valid
    x_valid = np.load(input_dir + 'x_valid.npy')
    y_valid = np.load(y_dir + 'y_valid.npy')
    
    if not os.path.exists(os.path.join(output_dir, 'pred_valid.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_valid.npy')):
           
           pred_valid, prob_valid = Test_Step(model, x_valid, 2)
           
           # Probabilidade apenas da classe 1 (de estradas)
           prob_valid = prob_valid[..., 1:2]
           
           # Converte para tipos que ocupam menos espaço
           pred_valid = pred_valid.astype(np.uint8)
           prob_valid = prob_valid.astype(np.float16)
           
           # Salva arrays de predição do Treinamento e Validação    
           salva_arrays(output_dir, pred_valid=pred_valid, prob_valid=prob_valid)
           
    pred_valid = np.load(output_dir + 'pred_valid.npy')
    prob_valid = np.load(output_dir + 'prob_valid.npy')
    
    buffers_y_valid = {}
    buffers_pred_valid = {}
    
    for dist in dist_buffers:
        # Buffers para Precisão Relaxada
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_valid_{dist}px.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(y_dir + f'buffer_y_valid_{dist}px.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}px.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}px.npy', buffers_pred_valid[dist])
            
            
    for dist in dist_buffers:
        buffers_y_valid[dist] = np.load(y_dir + f'buffer_y_valid_{dist}px.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}px.npy')
        
        
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_valid[dist], relaxed_recall_valid[dist], relaxed_f1score_valid[dist] = compute_relaxed_metrics(y_valid, 
                                                                                                           pred_valid, buffers_y_valid[dist],
                                                                                                           buffers_pred_valid[dist],
                                                                                                           nome_conjunto = 'Validação',
                                                                                                           print_file=f) 
            

    x_valid = None
    y_valid = None
    pred_valid = None
    prob_valid = None   
        
    buffers_y_valid = None
    buffers_pred_valid = None
        
    gc.collect()


    # Avalia teste
    x_test = np.load(input_dir + 'x_test.npy')
    y_test = np.load(y_dir + 'y_test.npy')
    
    if not os.path.exists(os.path.join(output_dir, 'pred_test.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_test.npy')):
        pred_test, prob_test = Test_Step(model, x_test, 2)
        
        prob_test = prob_test[..., 1:2] # Essa é a probabilidade de prever estrada (valor de pixel 1)
    
        # Converte para tipos que ocupam menos espaço
        pred_test = pred_test.astype(np.uint8)
        prob_test = prob_test.astype(np.float16)
        
        # Salva arrays de predição do Teste. Arquivos da Predição (pred) são salvos na pasta de arquivos de saída (resultados_dir)
        salva_arrays(output_dir, pred_test=pred_test, prob_test=prob_test)
        
    pred_test = np.load(output_dir + 'pred_test.npy')
    prob_test = np.load(output_dir + 'prob_test.npy')
    
    buffers_y_test = {}
    buffers_pred_test = {}
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_test_{dist}px.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(y_dir + f'buffer_y_test_{dist}px.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}px.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}px.npy', buffers_pred_test[dist])
            
            
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(y_dir + f'buffer_y_test_{dist}px.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}px.npy')
        
        
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_test[dist], relaxed_recall_test[dist], relaxed_f1score_test[dist] = compute_relaxed_metrics(y_test, 
                                                                                                       pred_test, buffers_y_test[dist],
                                                                                                       buffers_pred_test[dist], 
                                                                                                       nome_conjunto = 'Teste', 
                                                                                                       print_file=f)
            
    
    x_test = None
    y_test = None
    prob_test = None   
        
    buffers_y_test = None
    buffers_pred_test = None
        
    gc.collect()
    
    # Se avalia_ate_teste = True, a função para por aqui e não gera mosaicos e nem avalia a diferença
    # Os resultados no dicionário são atualizados até o que se tem, outras informações tem o valor None
    if avalia_ate_teste:
        if not avalia_train:
            relaxed_precision_train = None
            relaxed_recall_train = None
            relaxed_f1score_train = None

        relaxed_precision_mosaics = None
        relaxed_recall_mosaics = None
        relaxed_f1score_mosaics = None
        
        relaxed_precision_diff = None
        relaxed_recall_diff = None
        relaxed_f1score_diff = None
            
        dict_results = {
            'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
            'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
            'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
            'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics,
            'relaxed_precision_diff': relaxed_precision_diff, 'relaxed_recall_diff': relaxed_recall_diff, 'relaxed_f1score_diff': relaxed_f1score_diff
            }
        
        with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
            pickle.dump(dict_results, fp)
        
        return dict_results
    
    # Gera Mosaicos de Teste
    # Avalia Mosaicos de Teste
    if not os.path.exists(os.path.join(y_dir, 'y_mosaics.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_mosaics.npy')):

        # Stride e Dimensões do Tile
        with open(y_dir + 'info_tiles_test.pickle', "rb") as fp:   
            info_tiles_test = pickle.load(fp)
            len_tiles_test = info_tiles_test['len_tiles_test']
            shape_tiles_test = info_tiles_test['shape_tiles_test']
            patch_test_stride = info_tiles_test['patch_stride_test']
    
        # patch_test_stride = 210 # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches

        # labels_test_shape = (1500, 1500) # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
        labels_test_shape = shape_tiles_test

        # n_test_tiles = 49 # Número de tiles de teste
    
        # Pasta com os tiles de teste para pegar informações de georreferência
        test_labels_tiles_dir = r'dataset_massachusetts_mnih_mod/test/maps'
        # test_labels_tiles_dir = r'tiles/masks/2018/test'
        labels_paths = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
        labels_paths.sort()
    
        # Gera mosaicos e lista com os mosaicos previstos
        pred_mosaics = gera_mosaicos(output_dir, pred_test, labels_paths, 
                                     patch_test_stride=patch_test_stride,
                                     labels_test_shape=labels_test_shape,
                                     len_tiles_test=len_tiles_test, is_float=False)
    
        # Lista e Array dos Mosaicos de Referência
        y_mosaics = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
        #y_mosaics = np.array(y_mosaics)[..., np.newaxis]
        y_mosaics = stack_uneven(y_mosaics)[..., np.newaxis]
        
        # Transforma valor NODATA de y_mosaics em 0
        y_mosaics[y_mosaics==255] = 0
        
        # Array dos Mosaicos de Predição 
        #pred_mosaics = np.array(pred_mosaics)[..., np.newaxis]
        pred_mosaics = stack_uneven(pred_mosaics)[..., np.newaxis]
        pred_mosaics = pred_mosaics.astype(np.uint8)
        
        # Salva Array dos Mosaicos de Predição
        salva_arrays(y_dir, y_mosaics=y_mosaics)
        salva_arrays(output_dir, pred_mosaics=pred_mosaics)


    # Libera memória se for possível
    pred_test = None
    gc.collect()
    
    # Lê Mosaicos 
    y_mosaics = np.load(y_dir + 'y_mosaics.npy')
    pred_mosaics = np.load(output_dir + 'pred_mosaics.npy')
        
    # Buffer dos Mosaicos de Referência e Predição
    buffers_y_mosaics = {}
    buffers_pred_mosaics = {}
    
    for dist in dist_buffers:
        # Buffer dos Mosaicos de Referência
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_mosaics_{dist}px.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(y_dir + f'buffer_y_mosaics_{dist}px.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}px.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}px.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(y_dir + f'buffer_y_mosaics_{dist}px.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}px.npy')  
        
        
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_mosaics[dist], relaxed_recall_mosaics[dist], relaxed_f1score_mosaics[dist] = compute_relaxed_metrics(y_mosaics, 
                                                                                                            pred_mosaics, buffers_y_mosaics[dist],
                                                                                                            buffers_pred_mosaics[dist], 
                                                                                                            nome_conjunto = 'Mosaicos de Teste', 
                                                                                                            print_file=f)
            
    # Libera memória
    y_mosaics = None
    pred_mosaics = None
    
    buffers_y_mosaics = None
    buffers_pred_mosaics = None
    
    gc.collect()
    
    
    # Gera Diferença Referente a Novas Estradas e
    # Avalia Novas Estradas
    if avalia_diff:
        if not os.path.exists(os.path.join(y_dir, 'y_tiles_diff.npy')) or \
           not os.path.exists(os.path.join(output_dir, 'pred_tiles_diff.npy')):
                        
            # Lista de Tiles de Referência de Antes
            test_labels_tiles_before_dir = Path(r'tiles/masks/2016/test')
            test_labels_tiles_before = list(test_labels_tiles_before_dir.glob('*.tif'))
        
            # Lista de Tiles Preditos (referente a Depois)
            test_labels_tiles_predafter = list(Path(output_dir).glob('outmosaic*.tif'))
        
            # Extrai Diferenca entre Predição e Referência de Antes para
            # Computar Novas Estradas
            for tile_before, tile_after in zip(test_labels_tiles_before, test_labels_tiles_predafter):
                print(f'Extraindo Diferença entre {tile_after.name} e {tile_before.name}')
                suffix_extension = Path(tile_after).name.replace('outmosaic_', '', 1)
                out_raster_path = Path(output_dir) / f"diffnewroad_{suffix_extension}"
                extract_difference_reftiles(str(tile_before), str(tile_after), str(out_raster_path), buffer_px=3)
                
            # Lista de caminhos dos rótulos dos Tiles de Diferança de Teste
            test_labels_tiles_diff_dir = Path(r'tiles/masks/Diff/test')
            test_labels_tiles_diff = list(test_labels_tiles_diff_dir.glob('*.tif'))
            
            # Lista e Array da referência para os Tiles de Diferença
            y_tiles_diff = [gdal.Open(str(tile_diff)).ReadAsArray() for tile_diff in test_labels_tiles_diff]
            y_tiles_diff = stack_uneven(y_tiles_diff)[..., np.newaxis]
            
            # Lista e Array da predição da Diferença
            test_labels_tiles_preddiff = list(Path(output_dir).glob('diffnewroad*.tif'))
            pred_tiles_diff = [gdal.Open(str(tile_preddiff)).ReadAsArray() for tile_preddiff in test_labels_tiles_preddiff]
            pred_tiles_diff = stack_uneven(pred_tiles_diff)[..., np.newaxis]
            
            # Salva Arrays dos Rótulos e Predições das Diferenças
            salva_arrays(y_dir, y_tiles_diff=y_tiles_diff)
            salva_arrays(output_dir, pred_tiles_diff=pred_tiles_diff)
            
            
        # Lê Arrays dos Rótulos e Predições das Diferenças
        y_tiles_diff = np.load(y_dir + 'y_tiles_diff.npy')
        pred_tiles_diff = np.load(output_dir + 'pred_tiles_diff.npy')
        
        # Buffer dos Tiles de Referência e Predição da Diferença
        buffers_y_tiles_diff = {}
        buffers_pred_tiles_diff = {}
        
        for dist in dist_buffers:
            # Buffer da referência dos Tiles de Diferença
            if not os.path.exists(os.path.join(y_dir, f'buffer_y_tiles_diff_{dist}px.npy')):
                buffers_y_tiles_diff[dist] = buffer_patches(y_tiles_diff, dist_cells=dist)
                np.save(y_dir + f'buffer_y_tiles_diff_{dist}px.npy', buffers_y_tiles_diff[dist])
                
            # Buffer da predição da Diferença   
            if not os.path.exists(os.path.join(output_dir, f'buffer_pred_tiles_diff_{dist}px.npy')):
                buffers_pred_tiles_diff[dist] = buffer_patches(pred_tiles_diff, dist_cells=dist)
                np.save(output_dir + f'buffer_pred_tiles_diff_{dist}px.npy', buffers_pred_tiles_diff[dist])
                
        
        # Lê buffers das Diferenças
        for dist in dist_buffers:
            buffers_y_tiles_diff[dist] = np.load(y_dir + f'buffer_y_tiles_diff_{dist}px.npy')
            buffers_pred_tiles_diff[dist] = np.load(output_dir + f'buffer_pred_tiles_diff_{dist}px.npy')
            
        # Avaliação da Qualidade para a Diferença
        # Relaxed Metrics for difference tiles
        relaxed_precision_diff, relaxed_recall_diff, relaxed_f1score_diff = {}, {}, {}
          
        for dist in dist_buffers:
            with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
                relaxed_precision_diff[dist], relaxed_recall_diff[dist], relaxed_f1score_diff[dist] = compute_relaxed_metrics(y_tiles_diff, 
                                                                                                            pred_tiles_diff, buffers_y_tiles_diff[dist],
                                                                                                            buffers_pred_tiles_diff[dist], 
                                                                                                            nome_conjunto = 'Mosaicos de Diferenca', 
                                                                                                            print_file=f)
                
        # Libera memória
        y_tiles_diff = None
        pred_tiles_diff = None
        
        buffers_y_tiles_diff = None
        buffers_pred_tiles_diff = None
        
        gc.collect()
            
    
    # Save and Return dictionary with the values of precision, recall and F1-Score for all the groups (Train, Validation, Test, Mosaics of Test)
    if not avalia_train:
        relaxed_precision_train = None
        relaxed_recall_train = None
        relaxed_f1score_train = None

    if not avalia_diff:
        relaxed_precision_diff = None
        relaxed_recall_diff = None
        relaxed_f1score_diff = None           
        
    dict_results = {
        'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
        'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
        'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
        'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics,
        'relaxed_precision_diff': relaxed_precision_diff, 'relaxed_recall_diff': relaxed_recall_diff, 'relaxed_f1score_diff': relaxed_f1score_diff
        }
    
    with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(dict_results, fp)
    
    return dict_results









# Avalia um ensemble de modelos segundo conjuntos de treino, validação, teste e mosaicos de teste
# Retorna as métricas de precisão, recall e f1-score relaxados
# Além disso constroi os mosaicos de resultado 
# Etapa 1 se refere ao processamento no qual a entrada do pós-processamento é a predição do modelo
# Etapa 2 ao processamento no qual a entrada do pós-processamento é a imagem original mais a predição do modelo
# Etapa 3 ao processamento no qual a entrada do pós-processamento é a imagem original com blur gaussiano mais a predição desses dados ruidosos com o modelo
# Etapa 4 ao processamento no qual a entrada do pós-processamento é a imagem original mais a predição dela aplicando um blur gaussiano em apenas alguns subpatches
# Etapa 5 se refere ao pós-processamento   
def avalia_modelo_ensemble(input_dir: str, output_dir: str, metric_name = 'F1-Score', 
                           etapa=3, dist_buffers = [3], std_blur = 0.4, n_members=5):
    metric_name = metric_name
    etapa = etapa
    dist_buffers = dist_buffers
    std_blur = std_blur
    
    # Lê arrays referentes aos patches de treino e validação
    x_train = np.load(input_dir + 'x_train.npy')
    y_train = np.load(input_dir + 'y_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    y_valid = np.load(input_dir + 'y_valid.npy')
    
    # Copia y_train e y_valid para diretório de saída, pois eles serão usados depois como entrada (rótulos)
    # para o pós-processamento
    if etapa==1 or etapa==2 or etapa==3:
        shutil.copy(input_dir + 'y_train.npy', output_dir + 'y_train.npy')
        shutil.copy(input_dir + 'y_valid.npy', output_dir + 'y_valid.npy')
    
    
    # Nome base do modelo salvo e número de membros do ensemble    
    best_model_filename = 'best_model'
    n_members = n_members
    
    # Show and save history
    for i in range(n_members):
        with open(output_dir + 'history_pickle_' + best_model_filename + '_' + str(i+1) + '.pickle', "rb") as fp:   
            history = pickle.load(fp)
            
        # Show and save history
        show_training_plot(history, metric_name = metric_name, save=True, save_path=output_dir,
                                 save_name='plotagem' + '_' + str(i+1) + '.png')
    
    # Load model
    model_list = []
    
    for i in range(n_members):
        model = load_model(output_dir + best_model_filename + '_' + str(i+1) + '.keras', compile=False)
        model_list.append(model)
        
    # Test the model over training and validation data (outputs result if at least one of the files does not exist)
    if not os.path.exists(os.path.join(output_dir, 'pred_train_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_train_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_valid_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_valid_ensemble.npy')):
        pred_train_ensemble, prob_train_ensemble = Test_Step_Ensemble(model_list, x_train, 2)
        pred_valid_ensemble, prob_valid_ensemble = Test_Step_Ensemble(model_list, x_valid, 2)
            
        # Converte para tipos que ocupam menos espaço
        pred_train_ensemble = pred_train_ensemble.astype(np.uint8)
        prob_train_ensemble = prob_train_ensemble.astype(np.float16)
        pred_valid_ensemble = pred_valid_ensemble.astype(np.uint8)
        prob_valid_ensemble = prob_valid_ensemble.astype(np.float16)
            
        # Salva arrays de predição do Treinamento e Validação   
        salva_arrays(output_dir, pred_train_ensemble=pred_train_ensemble, prob_train_ensemble=prob_train_ensemble, 
                     pred_valid_ensemble=pred_valid_ensemble, prob_valid_ensemble=prob_valid_ensemble)
       
        
    # Lê arrays de predição do Treinamento e Validação (para o caso deles já terem sido gerados)  
    pred_train_ensemble = np.load(output_dir + 'pred_train_ensemble.npy')
    prob_train_ensemble = np.load(output_dir + 'prob_train_ensemble.npy')
    pred_valid_ensemble = np.load(output_dir + 'pred_valid_ensemble.npy')
    prob_valid_ensemble = np.load(output_dir + 'prob_valid_ensemble.npy')
    
    # Calcula e salva predição a partir da probabilidade média do ensemble
    if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_valid.npy')):
        pred_train = calcula_pred_from_prob_ensemble_mean(prob_train_ensemble)
        pred_valid = calcula_pred_from_prob_ensemble_mean(prob_valid_ensemble)
        
        pred_train = pred_train.astype(np.uint8)
        pred_valid = pred_valid.astype(np.uint8)
        
        salva_arrays(output_dir, pred_train=pred_train, pred_valid=pred_valid)
        
    
    # Lê arrays de predição do Treinamento e Validação (para o caso deles já terem sido gerados)  
    pred_train = np.load(output_dir + 'pred_train.npy')
    pred_valid = np.load(output_dir + 'pred_valid.npy')
        
        
    
    # Antigo: Copia predições pred_train e pred_valid com o nome de x_train e x_valid no diretório de saída,
    # Antigo: pois serão usados como dados de entrada para o pós-processamento
    # Faz concatenação de x_train com pred_train e salva no diretório de saída, com o nome de x_train,
    # para ser usado como entrada para o pós-processamento. Faz procedimento semelhante com x_valid
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
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            print('Fazendo Blur nas imagens de treino e validação e gerando as predições')
            x_train_blur = gaussian_filter(x_train.astype(np.float32), sigma=(0 ,std_blur, std_blur, 0)).astype(np.float16)
            x_valid_blur = gaussian_filter(x_valid.astype(np.float32), sigma=(0 ,std_blur, std_blur, 0)).astype(np.float16)
            _, prob_train_ensemble_blur = Test_Step_Ensemble(model, x_train_blur, 2)
            _, prob_valid_ensemble_blur = Test_Step_Ensemble(model, x_valid_blur, 2)
            pred_train_blur = calcula_pred_from_prob_ensemble_mean(prob_train_ensemble_blur)
            pred_valid_blur = calcula_pred_from_prob_ensemble_mean(prob_valid_ensemble_blur)
            x_train_new = np.concatenate((x_train_blur, pred_train_blur), axis=-1).astype(np.float16)
            x_valid_new = np.concatenate((x_valid_blur, pred_valid_blur), axis=-1).astype(np.float16)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
            


        
        
    # Faz os Buffers necessários, para treino e validação, nas imagens 
    
    # Precisão Relaxada - Buffer na imagem de rótulos
    # Sensibilidade Relaxada - Buffer na imagem extraída
    # F1-Score Relaxado - É obtido através da Precisão e Sensibilidade Relaxadas
    
    buffers_y_train = {}
    buffers_y_valid = {}
    buffers_pred_train = {}
    buffers_pred_valid = {}
    
    for dist in dist_buffers:
        # Buffers para Precisão Relaxada
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_train_{dist}px.npy')): 
            buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
            np.save(input_dir + f'buffer_y_train_{dist}px.npy', buffers_y_train[dist])
            
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_valid_{dist}px.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(input_dir + f'buffer_y_valid_{dist}px.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}px.npy')):
            buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_train_{dist}px.npy', buffers_pred_train[dist])
            
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}px.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}px.npy', buffers_pred_valid[dist])
    
    
    # Lê buffers de arrays de predição do Treinamento e Validação
    for dist in dist_buffers:
        buffers_y_train[dist] = np.load(input_dir + f'buffer_y_train_{dist}px.npy')
        buffers_y_valid[dist] = np.load(input_dir + f'buffer_y_valid_{dist}px.npy')
        buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}px.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}px.npy')
    
    
    # Relaxed Metrics for training and validation
    relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_train[dist], relaxed_recall_train[dist], relaxed_f1score_train[dist] = compute_relaxed_metrics(y_train, 
                                                                                                       pred_train, buffers_y_train[dist],
                                                                                                       buffers_pred_train[dist], 
                                                                                                       nome_conjunto = 'Treino', 
                                                                                                       print_file=f)        
            # Relaxed Metrics for validation
            relaxed_precision_valid[dist], relaxed_recall_valid[dist], relaxed_f1score_valid[dist] = compute_relaxed_metrics(y_valid, 
                                                                                                           pred_valid, buffers_y_valid[dist],
                                                                                                           buffers_pred_valid[dist],
                                                                                                           nome_conjunto = 'Validação',
                                                                                                           print_file=f) 
        
        
    # Lê arrays referentes aos patches de teste
    x_test = np.load(input_dir + 'x_test.npy')
    y_test = np.load(input_dir + 'y_test.npy')
    
    
    # Copia y_test para diretório de saída, pois ele será usado como entrada (rótulo) para o pós-processamento
    if etapa==1 or etapa==2 or etapa==3:
        shutil.copy(input_dir + 'y_test.npy', output_dir + 'y_test.npy')
        
    
    # Test the model over test data (outputs result if at least one of the files does not exist)
    if not os.path.exists(os.path.join(output_dir, 'pred_test_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_test_ensemble.npy')):
        pred_test_ensemble, prob_test_ensemble = Test_Step_Ensemble(model_list, x_test, 2)
        
        # Converte para tipos que ocupam menos espaço
        pred_test_ensemble = pred_test_ensemble.astype(np.uint8)
        prob_test_ensemble = prob_test_ensemble.astype(np.float16)
        
        # Salva arrays de predição do Teste. Arquivos da Predição (pred) são salvos na pasta de arquivos de saída (resultados_dir)
        salva_arrays(output_dir, pred_test_ensemble=pred_test_ensemble, prob_test_ensemble=prob_test_ensemble)
        
        
    # Lê arrays de predição do Teste
    pred_test_ensemble = np.load(output_dir + 'pred_test_ensemble.npy')
    prob_test_ensemble = np.load(output_dir + 'prob_test_ensemble.npy')
    
    
    # Calcula e salva predição a partir da probabilidade média do ensemble
    if not os.path.exists(os.path.join(output_dir, 'pred_test.npy')):
        pred_test = calcula_pred_from_prob_ensemble_mean(prob_test_ensemble)
        
        pred_test = pred_test.astype(np.uint8)
        
        salva_arrays(output_dir, pred_test=pred_test)
        
    
    # Lê arrays de predição do Teste
    pred_test = np.load(output_dir + 'pred_test.npy')
    
    # Antigo: Copia predição pred_test com o nome de x_test no diretório de saída,
    # Antigo: pois será usados como dado de entrada no pós-processamento
    # Faz concatenação de x_test com pred_test e salva no diretório de saída, com o nome de x_test,
    # para ser usado como entrada para o pós-processamento.
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            shutil.copy(output_dir + 'pred_test.npy', output_dir + 'x_test.npy')
    
    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            x_test_new = np.concatenate((x_test, pred_test), axis=-1)
            salva_arrays(output_dir, x_test=x_test_new)
        
    if etapa == 3:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            print('Fazendo Blur nas imagens de teste e gerando as predições')
            x_test_blur = gaussian_filter(x_test.astype(np.float32), sigma=(0 ,std_blur, std_blur, 0)).astype(np.float16)
            _, prob_test_ensemble_blur = Test_Step_Ensemble(model, x_test_blur, 2)
            pred_test_blur = calcula_pred_from_prob_ensemble_mean(prob_test_ensemble_blur)
            x_test_new = np.concatenate((x_test_blur, pred_test_blur), axis=-1).astype(np.float16)
            salva_arrays(output_dir, x_test=x_test_new)
    
    
    # Faz os Buffers necessários, para teste, nas imagens
    buffers_y_test = {}
    buffers_pred_test = {}
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_test_{dist}px.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(input_dir + f'buffer_y_test_{dist}px.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}px.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}px.npy', buffers_pred_test[dist])
            
            
    # Lê buffers de arrays de predição do Teste
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(input_dir + f'buffer_y_test_{dist}px.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}px.npy')
    
    # Relaxed Metrics for test
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_test[dist], relaxed_recall_test[dist], relaxed_f1score_test[dist] = compute_relaxed_metrics(y_test, 
                                                                                                       pred_test, buffers_y_test[dist],
                                                                                                       buffers_pred_test[dist], 
                                                                                                       nome_conjunto = 'Teste', 
                                                                                                       print_file=f)
        
    
    # Stride e Dimensões do Tile
    patch_test_stride = 96 # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches
    labels_test_shape = (1408, 1280) # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
    n_test_tiles = 10 # Número de tiles de teste
    
    # Pasta com os tiles de teste para pegar informações de georreferência
    test_labels_tiles_dir = r'new_teste_tiles\masks\test'
    labels_paths = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
    
    
    
    # Gera mosaicos e lista com os mosaicos previstos
    pred_test_mosaic_list = gera_mosaicos(output_dir, pred_test, labels_paths, 
                                          patch_test_stride=patch_test_stride,
                                          labels_test_shape=labels_test_shape,
                                          n_test_tiles=n_test_tiles, is_float=False)
    
    
    # Lista e Array dos Mosaicos de Referência
    y_mosaics_list = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
    y_mosaics = np.array(y_mosaics_list)[..., np.newaxis]
    
    # Array dos Mosaicos de Predição 
    pred_mosaics = np.array(pred_test_mosaic_list)[..., np.newaxis]
    pred_mosaics = pred_mosaics.astype(np.uint8)
    
    # Salva Array dos Mosaicos de Predição
    if not os.path.exists(os.path.join(input_dir, 'y_mosaics.npy')): salva_arrays(input_dir, y_mosaics=y_mosaics)
    if not os.path.exists(os.path.join(output_dir, 'pred_mosaics.npy')): salva_arrays(output_dir, pred_mosaics=pred_mosaics)
    
    # Lê Mosaicos 
    y_mosaics = np.load(input_dir + 'y_mosaics.npy')
    pred_mosaics = np.load(output_dir + 'pred_mosaics.npy')
    
    
    # Buffer dos Mosaicos de Referência e Predição
    buffers_y_mosaics = {}
    buffers_pred_mosaics = {}
    
    for dist in dist_buffers:
        # Buffer dos Mosaicos de Referência
        if not os.path.exists(os.path.join(input_dir, f'buffers_y_mosaics_{dist}px.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(input_dir + f'buffer_y_mosaics_{dist}px.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}px.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}px.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(input_dir + f'buffer_y_mosaics_{dist}px.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}px.npy')  
    
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_mosaics[dist], relaxed_recall_mosaics[dist], relaxed_f1score_mosaics[dist] = compute_relaxed_metrics(y_mosaics, 
                                                                                                       pred_mosaics, buffers_y_mosaics[dist],
                                                                                                       buffers_pred_mosaics[dist], 
                                                                                                       nome_conjunto = 'Mosaicos de Teste', 
                                                                                                       print_file=f)
            
    
    # Save and Return dictionary with the values of precision, recall and F1-Score for all the groups (Train, Validation, Test, Mosaics of Test)
    dict_results = {
        'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
        'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
        'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
        'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics             
        }
    
    with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(dict_results, fp)
    
    return dict_results




    
def avalia_transfer_learning_segformer(input_dir, y_dir, output_dir, model_checkpoint, model_weights_filename, metric_name = 'F1-Score', 
                                       dist_buffers=[3], avalia_train=False, 
                                       avalia_ate_teste=False, avalia_diff=False):
    model_weights_filename = model_weights_filename
    metric_name = metric_name
    dist_buffers = dist_buffers    
    
    # Lê histórico
    with open(output_dir + 'history_pickle_' + Path(model_weights_filename).stem + '.pickle', "rb") as fp:   
        history = pickle.load(fp)
    
    # Mostra histórico em gráfico
    show_training_plot(history, metric_name = metric_name, save=True, save_path=output_dir)

    # Load Model
    id2label = {0: "background", 1: "road"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Load Saved Weights of the Model
    model.load_weights(output_dir + model_weights_filename)

    # # Load model
    # model = load_model(output_dir + model_weights_filename, compile=False, custom_objects={"TFSegformerMainLayer": TFSegformerMainLayer, 
    #                                                                                       "TFSegformerDecodeHead": TFSegformerDecodeHead})
    
    # Avalia treino    
    if avalia_train:
        x_train = np.load(input_dir + 'x_train_fine.npy')
        y_train = np.load(y_dir + 'y_train_fine.npy')
        
        if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')) or \
           not os.path.exists(os.path.join(output_dir, 'prob_train.npy')): 
    
            prob_train = model.predict(x_train, batch_size=2, verbose=1).logits
            prob_train = tf.transpose(prob_train, [0, 2, 3, 1]) # Transpose dimensions to channel-last
            prob_train = tf.image.resize(prob_train, size=(x_train.shape[2], x_train.shape[3])).numpy()
            pred_train = np.argmax(prob_train, axis=-1)[..., np.newaxis]
            
            # Probabilidade apenas da classe 1 (de estradas)
            prob_train = prob_train[..., 1:2] 
            
            # Converte para tipos que ocupam menos espaço
            pred_train = pred_train.astype(np.uint8)
            prob_train = prob_train.astype(np.float16)

            # Salva arrays de predição do Treinamento e Validação    
            salva_arrays(output_dir, pred_train=pred_train, prob_train=prob_train)
        
        pred_train = np.load(output_dir + 'pred_train.npy')
        prob_train = np.load(output_dir + 'prob_train.npy')

        # Faz os Buffers necessários, para treino, nas imagens 
        
        # Precisão Relaxada - Buffer na imagem de rótulos
        # Sensibilidade Relaxada - Buffer na imagem extraída
        # F1-Score Relaxado - É obtido através da Precisão e Sensibilidade Relaxadas
        
        buffers_y_train = {}
        buffers_pred_train = {}
        
        # Add dimension to Y to buffers calculations
        y_train = np.expand_dims(y_train, 3)
    
        for dist in dist_buffers:
            # Buffers para Precisão Relaxada
            if not os.path.exists(os.path.join(y_dir, f'buffer_y_train_{dist}px.npy')): 
                buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
                np.save(y_dir + f'buffer_y_train_{dist}px.npy', buffers_y_train[dist])
                
            # Buffers para Sensibilidade Relaxada
            if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}px.npy')):
                buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
                np.save(output_dir + f'buffer_pred_train_{dist}px.npy', buffers_pred_train[dist])
            

        # Lê buffers de arrays de predição do Treinamento
        for dist in dist_buffers:
            buffers_y_train[dist] = np.load(y_dir + f'buffer_y_train_{dist}px.npy')            
            buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}px.npy')
  
        
        # Relaxed Metrics for training
        relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    
    
        for dist in dist_buffers:
            with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
                relaxed_precision_train[dist], relaxed_recall_train[dist], relaxed_f1score_train[dist] = compute_relaxed_metrics(y_train, 
                                                                                                                                 pred_train, buffers_y_train[dist],
                                                                                                                                 buffers_pred_train[dist], 
                                                                                                                                 nome_conjunto = 'Treino', 
                                                                                                                                 print_file=f)
                
        # Release Memory
        x_train = None
        y_train = None
        pred_train = None
        prob_train = None
        
        buffers_y_train = None
        buffers_pred_train = None
            
        gc.collect()
        
    # Avalia Valid
    x_valid = np.load(input_dir + 'x_valid_fine.npy')
    y_valid = np.load(y_dir + 'y_valid_fine.npy')
    
    if not os.path.exists(os.path.join(output_dir, 'pred_valid.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_valid.npy')):
           
        prob_valid = model.predict(x_valid, batch_size=2, verbose=1).logits
        prob_valid = tf.transpose(prob_valid, [0, 2, 3, 1]) # Transpose dimensions
        prob_valid = tf.image.resize(prob_valid, size=(x_valid.shape[2], x_valid.shape[3])).numpy()
        pred_valid = np.argmax(prob_valid, axis=-1)[..., np.newaxis]
           
        # Probabilidade apenas da classe 1 (de estradas)
        prob_valid = prob_valid[..., 1:2]
           
        # Converte para tipos que ocupam menos espaço
        pred_valid = pred_valid.astype(np.uint8)
        prob_valid = prob_valid.astype(np.float16)
           
        # Salva arrays de predição do Treinamento e Validação    
        salva_arrays(output_dir, pred_valid=pred_valid, prob_valid=prob_valid)
           
    pred_valid = np.load(output_dir + 'pred_valid.npy')
    prob_valid = np.load(output_dir + 'prob_valid.npy')
    
    buffers_y_valid = {}
    buffers_pred_valid = {}
    
    # Add dimension to Y to buffers calculations
    y_valid = np.expand_dims(y_valid, 3)
    
    for dist in dist_buffers:
        # Buffers para Precisão Relaxada
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_valid_{dist}px.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(y_dir + f'buffer_y_valid_{dist}px.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}px.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}px.npy', buffers_pred_valid[dist])
            
            
    for dist in dist_buffers:
        buffers_y_valid[dist] = np.load(y_dir + f'buffer_y_valid_{dist}px.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}px.npy')
        
        
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_valid[dist], relaxed_recall_valid[dist], relaxed_f1score_valid[dist] = compute_relaxed_metrics(y_valid, 
                                                                                                           pred_valid, buffers_y_valid[dist],
                                                                                                           buffers_pred_valid[dist],
                                                                                                           nome_conjunto = 'Validação',
                                                                                                           print_file=f) 
            

    x_valid = None
    y_valid = None
    pred_valid = None
    prob_valid = None   
        
    buffers_y_valid = None
    buffers_pred_valid = None
        
    gc.collect()
    
    # Avalia teste
    x_test = np.load(input_dir + 'x_test_fine.npy')
    y_test = np.load(y_dir + 'y_test_fine.npy')
    
    if not os.path.exists(os.path.join(output_dir, 'pred_test.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_test.npy')):
        
        prob_test = model.predict(x_test, batch_size=2, verbose=1).logits
        prob_test = tf.transpose(prob_test, [0, 2, 3, 1]) # Transpose dimensions
        prob_test = tf.image.resize(prob_test, size=(x_test.shape[2], x_test.shape[3])).numpy()
        pred_test = np.argmax(prob_test, axis=-1)[..., np.newaxis]
                
        prob_test = prob_test[..., 1:2] # Essa é a probabilidade de prever estrada (valor de pixel 1)
    
        # Converte para tipos que ocupam menos espaço
        pred_test = pred_test.astype(np.uint8)
        prob_test = prob_test.astype(np.float16)
        
        # Salva arrays de predição do Teste. Arquivos da Predição (pred) são salvos na pasta de arquivos de saída (resultados_dir)
        salva_arrays(output_dir, pred_test=pred_test, prob_test=prob_test)
        
    pred_test = np.load(output_dir + 'pred_test.npy')
    prob_test = np.load(output_dir + 'prob_test.npy')
    
    buffers_y_test = {}
    buffers_pred_test = {}
    
    # Add dimension to Y to buffers calculations
    y_test = np.expand_dims(y_test, 3)
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_test_{dist}px.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(y_dir + f'buffer_y_test_{dist}px.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}px.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}px.npy', buffers_pred_test[dist])
            
            
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(y_dir + f'buffer_y_test_{dist}px.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}px.npy')
        
        
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_test[dist], relaxed_recall_test[dist], relaxed_f1score_test[dist] = compute_relaxed_metrics(y_test, 
                                                                                                       pred_test, buffers_y_test[dist],
                                                                                                       buffers_pred_test[dist], 
                                                                                                       nome_conjunto = 'Teste', 
                                                                                                       print_file=f)
            
    
    x_test = None
    y_test = None
    prob_test = None   
        
    buffers_y_test = None
    buffers_pred_test = None
        
    gc.collect()
    
    # Se avalia_ate_teste = True, a função para por aqui e não gera mosaicos e nem avalia a diferença
    # Os resultados no dicionário são atualizados até o que se tem, outras informações tem o valor None
    if avalia_ate_teste:
        if not avalia_train:
            relaxed_precision_train = None
            relaxed_recall_train = None
            relaxed_f1score_train = None

        relaxed_precision_mosaics = None
        relaxed_recall_mosaics = None
        relaxed_f1score_mosaics = None
        
        relaxed_precision_diff = None
        relaxed_recall_diff = None
        relaxed_f1score_diff = None
            
        dict_results = {
            'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
            'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
            'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
            'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics,
            'relaxed_precision_diff': relaxed_precision_diff, 'relaxed_recall_diff': relaxed_recall_diff, 'relaxed_f1score_diff': relaxed_f1score_diff
            }
        
        with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
            pickle.dump(dict_results, fp)
        
        return dict_results
    
    # Gera Mosaicos de Teste
    # Avalia Mosaicos de Teste
    if not os.path.exists(os.path.join(y_dir, 'y_mosaics.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_mosaics.npy')):

        # Stride e Dimensões do Tile
        with open(y_dir + 'info_tiles_test.pickle', "rb") as fp:   
            info_tiles_test = pickle.load(fp)
            len_tiles_test = info_tiles_test['len_tiles_test']
            shape_tiles_test = info_tiles_test['shape_tiles_test']
            patch_test_stride = info_tiles_test['patch_stride_test']
    
        # patch_test_stride = 210 # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches

        # labels_test_shape = (1500, 1500) # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
        labels_test_shape = shape_tiles_test

        # n_test_tiles = 49 # Número de tiles de teste
    
        # Pasta com os tiles de teste para pegar informações de georreferência
        #test_labels_tiles_dir = r'dataset_massachusetts_mnih/test/maps'
        test_labels_tiles_dir = r'tiles/masks/2018/test'
        labels_paths = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
        labels_paths.sort()
    
        # Gera mosaicos e lista com os mosaicos previstos
        pred_mosaics = gera_mosaicos(output_dir, pred_test, labels_paths, 
                                     patch_test_stride=patch_test_stride,
                                     labels_test_shape=labels_test_shape,
                                     len_tiles_test=len_tiles_test, is_float=False)
    
        # Lista e Array dos Mosaicos de Referência
        y_mosaics = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
        #y_mosaics = np.array(y_mosaics)[..., np.newaxis]
        y_mosaics = stack_uneven(y_mosaics)[..., np.newaxis]
        
        # Transforma valor NODATA de y_mosaics em 0
        y_mosaics[y_mosaics==255] = 0
        
        # Array dos Mosaicos de Predição 
        #pred_mosaics = np.array(pred_mosaics)[..., np.newaxis]
        pred_mosaics = stack_uneven(pred_mosaics)[..., np.newaxis]
        pred_mosaics = pred_mosaics.astype(np.uint8)
        
        # Salva Array dos Mosaicos de Predição
        salva_arrays(y_dir, y_mosaics=y_mosaics)
        salva_arrays(output_dir, pred_mosaics=pred_mosaics)


    # Libera memória se for possível
    pred_test = None
    gc.collect()
    
    # Lê Mosaicos 
    y_mosaics = np.load(y_dir + 'y_mosaics.npy')
    pred_mosaics = np.load(output_dir + 'pred_mosaics.npy')
        
    # Buffer dos Mosaicos de Referência e Predição
    buffers_y_mosaics = {}
    buffers_pred_mosaics = {}
    
    for dist in dist_buffers:
        # Buffer dos Mosaicos de Referência
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_mosaics_{dist}px.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(y_dir + f'buffer_y_mosaics_{dist}px.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}px.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}px.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(y_dir + f'buffer_y_mosaics_{dist}px.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}px.npy')  
        
        
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_mosaics[dist], relaxed_recall_mosaics[dist], relaxed_f1score_mosaics[dist] = compute_relaxed_metrics(y_mosaics, 
                                                                                                            pred_mosaics, buffers_y_mosaics[dist],
                                                                                                            buffers_pred_mosaics[dist], 
                                                                                                            nome_conjunto = 'Mosaicos de Teste', 
                                                                                                            print_file=f)
            
    # Libera memória
    y_mosaics = None
    pred_mosaics = None
    
    buffers_y_mosaics = None
    buffers_pred_mosaics = None
    
    gc.collect()
    
    
    # Gera Diferença Referente a Novas Estradas e
    # Avalia Novas Estradas
    if avalia_diff:
        if not os.path.exists(os.path.join(y_dir, 'y_tiles_diff.npy')) or \
           not os.path.exists(os.path.join(output_dir, 'pred_tiles_diff.npy')):
                        
            # Lista de Tiles de Referência de Antes
            test_labels_tiles_before_dir = Path(r'tiles/masks/2016/test')
            test_labels_tiles_before = list(test_labels_tiles_before_dir.glob('*.tif'))
        
            # Lista de Tiles Preditos (referente a Depois)
            test_labels_tiles_predafter = list(Path(output_dir).glob('outmosaic*.tif'))
        
            # Extrai Diferenca entre Predição e Referência de Antes para
            # Computar Novas Estradas
            for tile_before, tile_after in zip(test_labels_tiles_before, test_labels_tiles_predafter):
                print(f'Extraindo Diferença entre {tile_after.name} e {tile_before.name}')
                suffix_extension = Path(tile_after).name.replace('outmosaic_', '', 1)
                out_raster_path = Path(output_dir) / f"diffnewroad_{suffix_extension}"
                extract_difference_reftiles(str(tile_before), str(tile_after), str(out_raster_path), buffer_px=3)
                
            # Lista de caminhos dos rótulos dos Tiles de Diferança de Teste
            test_labels_tiles_diff_dir = Path(r'tiles/masks/Diff/test')
            test_labels_tiles_diff = list(test_labels_tiles_diff_dir.glob('*.tif'))
            
            # Lista e Array da referência para os Tiles de Diferença
            y_tiles_diff = [gdal.Open(str(tile_diff)).ReadAsArray() for tile_diff in test_labels_tiles_diff]
            y_tiles_diff = stack_uneven(y_tiles_diff)[..., np.newaxis]
            
            # Lista e Array da predição da Diferença
            test_labels_tiles_preddiff = list(Path(output_dir).glob('diffnewroad*.tif'))
            pred_tiles_diff = [gdal.Open(str(tile_preddiff)).ReadAsArray() for tile_preddiff in test_labels_tiles_preddiff]
            pred_tiles_diff = stack_uneven(pred_tiles_diff)[..., np.newaxis]
            
            # Salva Arrays dos Rótulos e Predições das Diferenças
            salva_arrays(y_dir, y_tiles_diff=y_tiles_diff)
            salva_arrays(output_dir, pred_tiles_diff=pred_tiles_diff)
            
            
        # Lê Arrays dos Rótulos e Predições das Diferenças
        y_tiles_diff = np.load(y_dir + 'y_tiles_diff.npy')
        pred_tiles_diff = np.load(output_dir + 'pred_tiles_diff.npy')
        
        # Buffer dos Tiles de Referência e Predição da Diferença
        buffers_y_tiles_diff = {}
        buffers_pred_tiles_diff = {}
        
        for dist in dist_buffers:
            # Buffer da referência dos Tiles de Diferença
            if not os.path.exists(os.path.join(y_dir, f'buffer_y_tiles_diff_{dist}px.npy')):
                buffers_y_tiles_diff[dist] = buffer_patches(y_tiles_diff, dist_cells=dist)
                np.save(y_dir + f'buffer_y_tiles_diff_{dist}px.npy', buffers_y_tiles_diff[dist])
                
            # Buffer da predição da Diferença   
            if not os.path.exists(os.path.join(output_dir, f'buffer_pred_tiles_diff_{dist}px.npy')):
                buffers_pred_tiles_diff[dist] = buffer_patches(pred_tiles_diff, dist_cells=dist)
                np.save(output_dir + f'buffer_pred_tiles_diff_{dist}px.npy', buffers_pred_tiles_diff[dist])
                
        
        # Lê buffers das Diferenças
        for dist in dist_buffers:
            buffers_y_tiles_diff[dist] = np.load(y_dir + f'buffer_y_tiles_diff_{dist}px.npy')
            buffers_pred_tiles_diff[dist] = np.load(output_dir + f'buffer_pred_tiles_diff_{dist}px.npy')
            
        # Avaliação da Qualidade para a Diferença
        # Relaxed Metrics for difference tiles
        relaxed_precision_diff, relaxed_recall_diff, relaxed_f1score_diff = {}, {}, {}
          
        for dist in dist_buffers:
            with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
                relaxed_precision_diff[dist], relaxed_recall_diff[dist], relaxed_f1score_diff[dist] = compute_relaxed_metrics(y_tiles_diff, 
                                                                                                            pred_tiles_diff, buffers_y_tiles_diff[dist],
                                                                                                            buffers_pred_tiles_diff[dist], 
                                                                                                            nome_conjunto = 'Mosaicos de Diferenca', 
                                                                                                            print_file=f)
                
        # Libera memória
        y_tiles_diff = None
        pred_tiles_diff = None
        
        buffers_y_tiles_diff = None
        buffers_pred_tiles_diff = None
        
        gc.collect()
            
    
    # Save and Return dictionary with the values of precision, recall and F1-Score for all the groups (Train, Validation, Test, Mosaics of Test)
    if not avalia_train:
        relaxed_precision_train = None
        relaxed_recall_train = None
        relaxed_f1score_train = None

    if not avalia_diff:
        relaxed_precision_diff = None
        relaxed_recall_diff = None
        relaxed_f1score_diff = None           
        
    dict_results = {
        'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
        'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
        'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
        'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics,
        'relaxed_precision_diff': relaxed_precision_diff, 'relaxed_recall_diff': relaxed_recall_diff, 'relaxed_f1score_diff': relaxed_f1score_diff
        }
    
    with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(dict_results, fp)
    
    return dict_results