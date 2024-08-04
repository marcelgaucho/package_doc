# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:54:18 2024

@author: Marcel
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt 
from .buffer_functions import buffer_patches
from .compute_functions import intersect
from sklearn.metrics import precision_score, recall_score

def plota_resultado_experimentos(lista_precisao, lista_recall, lista_f1score, lista_nomes_exp, nome_conjunto,
                                 save=False, save_path=r''):
    '''
    

    Parameters
    ----------
    lista_precisao : TYPE
        Lista das Precisões na ordem dada pela lista dos experimentos lista_nomes_exp.
    lista_recall : TYPE
        Lista dos Recalls na ordem dada pela lista dos experimentos lista_nomes_exp.
    lista_f1score : TYPE
        Lista dos F1-Scores na ordem dada pela lista dos experimentos lista_nomes_exp.
    lista_nomes_exp : TYPE
        Lista dos nomes dos experimentos.
    nome_conjunto : TYPE
        Nome do Conjunto sendo testado, por exemplo, "Treino", "Validação", "Teste", "Mosaicos de Teste", etc. 
        Esse nome será usado no título do gráfico e para nome do arquivo do gráfico.

    Returns
    -------
    None.

    '''
    # Figura, posição dos labels e largura da barra
    n_exp = len(lista_nomes_exp)
    fig = plt.figure(figsize = (15, 10))
    x = np.arange(n_exp) # the label locations
    width = 0.2
    
    # Cria barras
    plt.bar(x-0.2, lista_precisao, width, color='cyan')
    plt.bar(x, lista_recall, width, color='orange')
    plt.bar(x+0.2, lista_f1score, width, color='green')
    
    # Nomes dos grupos em X
    plt.xticks(x, lista_nomes_exp)
    
    # Nomes dos eixos
    plt.xlabel("Métricas")
    plt.ylabel("Valor Percentual (%)")
    
    # Título do gráfico
    plt.title("Resultados das Métricas para %s" % (nome_conjunto))
    
    # Legenda
    legend = plt.legend(["Precisão", "Recall", "F1-Score"], ncol=3, framealpha=0.5)
    #legend.get_frame().set_alpha(0.9)

    # Salva figura se especificado
    if save:
        plt.savefig(save_path + nome_conjunto + ' plotagem_resultado.png')
    
    # Exibe gráfico
    plt.show()
    

# Gera gráficos de Treino, Validação, Teste e Mosaicos de Teste para os experimentos desejados,
# devendo informar os diretórios com os resultados desses experimentos, o nome desses experimentos e
# o diretório onde salvar os gráficos
def gera_graficos(metrics_dirs_list, lista_nomes_exp, save_path=r''):
    # Open metrics and add to list
    resultados_metricas_list = []
    
    for mdir in metrics_dirs_list:
        with open(mdir + "resultados_metricas.pickle", "rb") as fp:   # Unpickling
            metrics = pickle.load(fp)
            
        resultados_metricas_list.append(metrics)
        
    # Resultados para Treino
    precision_treino = [] 
    recall_treino = []
    f1score_treino = []
    
    # Resultados para Validação
    precision_validacao = []
    recall_validacao = []
    f1score_validacao = []
    
    # Resultados para Teste
    precision_teste = []
    recall_teste = []
    f1score_teste = []
    
    # Resultados para Mosaicos de Teste
    precision_mosaicos_teste = []
    recall_mosaicos_teste = [] 
    f1score_mosaicos_teste = []
    
    # Loop inside list of metrics and add to respective list
    for resultado in resultados_metricas_list:
        # Append results to Train
        precision_treino.append(resultado['relaxed_precision_train'])
        recall_treino.append(resultado['relaxed_recall_train'])
        f1score_treino.append(resultado['relaxed_f1score_train'])
        
        # Append results to Valid
        precision_validacao.append(resultado['relaxed_precision_valid'])
        recall_validacao.append(resultado['relaxed_recall_valid'])
        f1score_validacao.append(resultado['relaxed_f1score_valid'])
        
        # Append results to Test
        precision_teste.append(resultado['relaxed_precision_test'])
        recall_teste.append(resultado['relaxed_recall_test'])
        f1score_teste.append(resultado['relaxed_f1score_test'])
        
        # Append results to Mosaics of Test
        precision_mosaicos_teste.append(resultado['relaxed_precision_mosaics'])
        recall_mosaicos_teste.append(resultado['relaxed_recall_mosaics'])
        f1score_mosaicos_teste.append(resultado['relaxed_f1score_mosaics'])
    
    
    # Gera gráficos
    plota_resultado_experimentos(precision_treino, recall_treino, f1score_treino, lista_nomes_exp, 
                             'Treino', save=True, save_path=save_path)
    plota_resultado_experimentos(precision_validacao, recall_validacao, f1score_validacao, lista_nomes_exp, 
                             'Validação', save=True, save_path=save_path)
    plota_resultado_experimentos(precision_teste, recall_teste, f1score_teste, lista_nomes_exp, 
                                 'Teste', save=True, save_path=save_path)
    plota_resultado_experimentos(precision_mosaicos_teste, recall_mosaicos_teste, f1score_mosaicos_teste, lista_nomes_exp, 
                                 'Mosaicos de Teste', save=True, save_path=save_path)
    
    
def plota_curva_precision_recall_relaxada(y, prob, buffer_y, buffer_px=3, num_pontos=10, output_dir='',
                                          save_figure=True):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION. Array de Referência
    prob : TYPE
        DESCRIPTION. Array de Probabilidades gerado a partir do modelo
    buffer_y : TYPE
        DESCRIPTION. Buffer de  buffer_px do Array de Referência 
    buffer_px : TYPE
        DESCRIPTION. Valor do buffer em pixels, exemplo: 1, 2, 3.
    num_pontos : TYPE, optional
        DESCRIPTION. The default is 100. Número de pontos para gerar no gráfico. 
        A ligação desses pontos formará a curva precision-recall relaxada

    Returns
    -------
    precision_scores
        DESCRIPTION. sequência de precisões para cada limiar
    recall_scores
        DESCRIPTION. sequência de recalls para cada limiar
    intersection_found
        DESCRIPTION. ponto em que precision=recall, ou seja, que intercepta a reta em que
        precision=recall, chamado de breakeven point.        


    '''    
    # Achata arrays
    y_flat = np.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2]))
    buffer_y_flat = np.reshape(buffer_y, (buffer_y.shape[0] * buffer_y.shape[1] * buffer_y.shape[2]))
    
    # Define sequências de limiares que serão usados para gerar 
    # resultados de precision e recall
    probability_thresholds = np.linspace(0, 1, num=num_pontos)
    
    # Lista de Precisões e Recalls
    precision_scores = []
    recall_scores = []
    
    # Cria diretório para armazenar os buffers (por questão de performance)
    # Salva e carrega o buffer
    #prob_temp_dir = os.path.join(output_dir, 'prob_temp')
    #os.makedirs(prob_temp_dir, exist_ok=True)
    
    # Percorre probabilidades e calcula precisão e recall para cada uma delas
    # Adiciona cada uma a sua respectiva lista
    for i, p in enumerate(probability_thresholds):
        print(f'Calculando resultados {i+1}/{num_pontos}')
        # Predição e Buffer da Predição
        pred_for_prob = (prob > p).astype('uint8')
        #np.save(os.path.join(prob_temp_dir, f'prob_temp_list{i}.npy'), pred_for_prob)
        #pred_for_prob = np.load(os.path.join(prob_temp_dir, f'prob_temp_list{i}.npy')) 
        buffer_for_prob = buffer_patches(pred_for_prob, dist_cells=buffer_px)
       
        
        # Achatamento da Predição e Buffer da Predição, para entrada na função
        pred_for_prob_flat = np.reshape(pred_for_prob, (pred_for_prob.shape[0] * pred_for_prob.shape[1] * pred_for_prob.shape[2]))
        buffer_for_prob_flat = np.reshape(buffer_for_prob, (buffer_for_prob.shape[0] * buffer_for_prob.shape[1] * buffer_for_prob.shape[2]))
        
        # Cálculo da precisão e recall relaxados
        relaxed_precision = precision_score(buffer_y_flat, pred_for_prob_flat, pos_label=1, zero_division=1)
        relaxed_recall = recall_score(y_flat, buffer_for_prob_flat, pos_label=1, zero_division=1)
        
        # Adiciona precisão e recall às suas listas
        precision_scores.append(relaxed_precision)
        recall_scores.append(relaxed_recall)
        
        
    # Segmento de Linha (p1, p2) pertencerá à reta precision=recall
    # ou seja, está na diagonal do gráfico 
    p1 = (0,0)
    p2 = (1,1)
    
    # Acha algum segmento que faz interseção com a linha em que precision=recall        
    intersections = []
        
    for i in range(len(precision_scores)-1):
        print(i)
        # p3 vai ser o ponto atual da sequência e p4 será o próximo ponto
        p3 = (precision_scores[i], recall_scores[i])
        p4 = (precision_scores[i+1], recall_scores[i+1])
        
        print(p3, p4)
        
        # Adiciona à lists de interseções
        intersection = intersect(p1, p2, p3, p4)
        if intersection:
            intersections.append(intersection)
        else:
            intersections.append(False)
            
    # Pega a primeira interseção da lista caso múltiplas forem achadas        
    intersection_found = [inter for inter in intersections if inter is not False][0]
        
    # Exporta figura
    fig, ax = plt.subplots()
    ax.plot(precision_scores, recall_scores)
    ax.plot(intersection_found[0], intersection_found[1], marker="o", markersize=10)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Curva Precision x Recall')

    if save_figure:    
        fig.savefig(output_dir + f'curva_precision_recall_{buffer_px}px.png')
    
    plt.show()

    # Salva resultados em um dicionário
    curva_precision_recall_results = {'precision_scores':precision_scores,
                                      'recall_scores':recall_scores,
                                      'intersection_found':intersection_found}


    with open(output_dir + 'curva_precision_recall_results' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(curva_precision_recall_results, fp)
        
    # Imprime em arquivo de resultado
    with open(output_dir + f'relaxed_metrics_{buffer_px}px.txt', 'a') as f:
        print('\nBreakeven point in Precision-Recall Curve for Teste', file=f)
        print('=======', file=f)
        print('Precision=Recall in point: (%.4f, %.4f)' % (intersection_found[0], intersection_found[1]), file=f)
        print() 
        
        
    return precision_scores, recall_scores, intersection_found