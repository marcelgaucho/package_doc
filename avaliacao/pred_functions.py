# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:58:35 2024

@author: Marcel
"""

import numpy as np
    
# Faz a previsão de todos os patches de treinamento
def Test(model, patch_test):
    result = model.predict(patch_test)
    predicted_classes = np.argmax(result, axis=-1)
    return predicted_classes

# Faz a previsão de todos os patches de treinamento aos poucos (em determinados lotes).
# Isso é para poupar o processamento do computador
# Retorna também a probabilidade das classes, além da predição
def Test_Step(model, patch_test, step=2, out_sigmoid=False, threshold_sigmoid=0.5):
    result = model.predict(patch_test, batch_size=step, verbose=1)

    if out_sigmoid:
        predicted_classes = np.where(result > threshold_sigmoid, 1, 0)
    else:
        predicted_classes = np.argmax(result, axis=-1)[..., np.newaxis]
        
    return predicted_classes, result


# Faz a previsão de todos os patches de treinamento aos poucos (em determinados lotes).
# Isso é para poupar o processamento do computador
# Retorna também a probabilidade das classes, além da predição
# Retorna uma previsões para todo o ensemble de modelos, dado pela lista de modelos passada
# como parâmetro
def Test_Step_Ensemble(model_list, patch_test, step, out_sigmoid=False, threshold_sigmoid=0.5):
    # Lista de predições, cada item é a predição para um modelo do ensemble  
    pred_list = []
    prob_list = []
        
    # Loop no modelo 
    for i, model in enumerate(model_list):
        print(f'Predizendo usando modelo número {i} \n\n')
        predicted_classes, result = Test_Step(model, patch_test, step, out_sigmoid=out_sigmoid, threshold_sigmoid=threshold_sigmoid)
        pred_list.append(predicted_classes)
        prob_list.append(result)
        
    # Transforma listas em arrays e retorna arrays como resultado
    pred_array = np.array(pred_list)
    prob_array = np.array(prob_list)
    
    return pred_array, prob_array