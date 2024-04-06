# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:25:39 2024

@author: Marcel
"""

import numpy as np
import os, shutil
import math
import tensorflow as tf
import pickle
import time
import types
import gc
from pathlib import Path

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Metric
# from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras import backend as K

from transformers import SegformerFeatureExtractor, TFSegformerForSemanticSegmentation

from .arquiteturas.unetr_2d import config_dict
print(config_dict)
from .arquiteturas.modelos import build_model

from tensorflow.keras.optimizers import Adam, SGD
#from focal_loss import SparseCategoricalFocalLoss

import matplotlib.pyplot as plt





# Função que mostra gráfico
def show_graph_loss_accuracy(history, accuracy_position, metric_name = 'accuracy', save=False, save_path=r'', save_name='plotagem.png'):
    plt.rcParams['axes.facecolor']='white'
    plt.figure(num=1, figsize=(14,6))

    config = [ { 'title': 'model %s' % (metric_name), 'ylabel': '%s' % (metric_name), 'legend_position': 'upper left', 'index_position': accuracy_position },
               { 'title': 'model loss', 'ylabel': 'loss', 'legend_position': 'upper right', 'index_position': 0 } ]

    for i in range(len(config)):
        
        plot_number = 120 + (i+1)
        plt.subplot(plot_number)
        plt.plot(history[0,:,0, config[i]['index_position']])
        plt.plot(history[1,:,0, config[i]['index_position']])
        plt.title(config[i]['title'])
        plt.ylabel(config[i]['ylabel'])
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc=config[i]['legend_position'])
        plt.tight_layout()
        
    if save:
        plt.savefig(save_path + save_name)
        plt.close()
    else:
        plt.show(block=False)


def train_unet(net, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, filepath, 
               filename, early_stopping=True, early_loss=False, metric_name='f1score', lr_decay=False, train_with_dataset=False, 
               tensorboard_log=False):
    print('Start the training...')
    
    early_stop = None
    filepath_name = os.path.join(filepath, filename+'.keras')

    
    # Para log no tensorboard, precisa ter callback
    if tensorboard_log:
        log_dir = os.path.join(filepath, 'logs', 'fit') #, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        shutil.rmtree(log_dir)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # Com Learning Rate Decay
    if lr_decay:
        initial_lrate = 0.001
        drop = 0.1
        epochs_drop = 30.0
        print(f'Initial Learning Rate={initial_lrate}, Drop={drop}, Epochs Drop={epochs_drop}')        
        def step_decay(epoch):
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    
    # Com Early Stopping
    if early_stopping:
        # Early Stopping por Loss
        if early_loss:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                          min_delta=early_stopping_delta, verbose=1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_weights_only=False,
                                                              verbose=1,
                                                              save_freq='epoch',
                                                              save_best_only=True)
            
        # Early Stopping por Métrica
        else:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_'+metric_name, patience=early_stopping_epochs, mode='max', restore_best_weights=True,
                                                          min_delta=early_stopping_delta, verbose=1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name,
                                                              monitor='val_'+metric_name,
                                                              mode='max',
                                                              save_weights_only=False,
                                                              verbose=1,
                                                              save_freq='epoch',
                                                              save_best_only=True)
    
    # Sem Early Stopping salva apenas o último modelo
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name, 
                                                         monitor='val_loss',
                                                         mode='auto',
                                                         save_weights_only=False,
                                                         verbose=1,
                                                         save_freq='epoch',
                                                         save_best_only=False)
    
    # Constroi lista de callbacks
    if tensorboard_log:
        if lr_decay:
            if early_stop:
                callbacks = [early_stop, cp_callback, tensorboard_callback, lrate]
            else:
                callbacks = [cp_callback, tensorboard_callback, lrate]
        
        else:
            if early_stop:
                callbacks = [early_stop, cp_callback, tensorboard_callback]
            else:
                callbacks = [cp_callback, tensorboard_callback]
    else:
        if lr_decay:
            if early_stop:
                callbacks = [early_stop, cp_callback, lrate]
            else:
                callbacks = [cp_callback, lrate]
        
        else:
            if early_stop:
                callbacks = [early_stop, cp_callback]
            else:
                callbacks = [cp_callback]
        
        
    # Treina Modelo
    if train_with_dataset:
        pass
    else:
        history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose='auto',
                          callbacks=callbacks, validation_data=(x_valid, y_valid))
    
    # Retorna resultados de Treino e Validação
    historia = history.history
    
    lista_loss = historia['loss']
    lista_metric = historia[metric_name]
    lista_val_loss = historia['val_loss']
    lista_val_metric = historia['val_' + metric_name]
    
    history_train = [np.array([dupla]) for dupla in zip(lista_loss, lista_metric)]
    history_valid = [np.array([dupla]) for dupla in zip(lista_val_loss, lista_val_metric)]
    resultado = [ history_train, history_valid ]
    
    return resultado


class F1Score(Metric):
    def __init__(self, name='f1score', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
        # initializing an object of the super class
        super(F1Score, self).__init__(name=name, **kwargs)
          
        # initializing state variables
        self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives 
        self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
        self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives
          
        # initializing other atrributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2 
        self.threshold = threshold
        self.epsilon = epsilon
    
    def update_state(self, ytrue, ypred, sample_weight=None):
        # Pega só referente a classe 1
        ypred = ypred[..., 1:2] 
        ytrue = ytrue[..., 1:2] 
          
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)
        
        #print(f'Shape Shape de Y True é {ytrue.shape}')
        #print(f'Shape Shape de Y Pred é {ytrue.shape}')
          
        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
            
        self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
        self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
        self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute
    
    def result(self):
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
          
        # calculating fbeta
        self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
        
        return self.fb
    
    def reset_state(self):
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero
        




def dice_loss(y_true, y_pred, smooth=1e-6):
    '''
    Dice Coef = (2*Inter)/(Union+Inter)
    Dice Loss = 1 - Dice Coef    
    '''
    # Cast y_true as float 32
    y_true = tf.cast(y_true, tf.float32)
    
    # Flatten Arrays        
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    
    # Intersection and Union
    intersection = K.sum(y_true_f * y_pred_f)
    union_plus_inter = K.sum(y_true_f) + K.sum(y_pred_f)
    
    # Compute Dice Loss
    dice_coef = (2. * intersection + smooth) / (union_plus_inter + smooth)
    #dice_loss = 1 - dice_coef
    dice_loss = -K.log(dice_coef)
    
    return dice_loss

        
def weighted_focal_loss(alpha, gamma=0):
    """
    compute focal loss according to the prob of the sample.
    loss= -(1-p)^gamma*log(p)

    Variables:
        gamma: integer
        alpha: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        gamma = 2 #The larger the gamma value, the less importance of easily classified. Typical values: 0 to 5
        alpha = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_focal_loss(gamma, alpha)
        model.compile(loss=focal_loss,optimizer='adam')
    """
    
    alpha = K.variable(alpha)
        
    def focal_loss(y_true, y_pred):
        print('Tipo y_true =', y_true.dtype)
        print('Tipo y_pred =', y_pred.dtype)
        print('Tipo alpha =', alpha.dtype)
        # Cast y_true as float 32
        y_true = tf.cast(y_true, tf.float32)
        
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred_inv = 1.0 - y_pred
        y_pred_inv = K.pow(y_pred_inv, gamma)
        # calc
        focal_loss = y_true * K.log(y_pred) * y_pred_inv * alpha
        focal_loss = K.mean(-K.sum(focal_loss, -1))
        return focal_loss
    
    return focal_loss


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    #weights = K.variable(weights)
    weights = tf.Variable(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        #loss = y_true * K.log(y_pred) * weights
        loss = y_true * tf.math.log(y_pred) * weights
        #loss = K.mean(-K.sum(loss, -1))
        loss = tf.reduce_mean(-tf.reduce_sum(loss, axis=-1))
        
        return loss

    return loss


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Função para treinar o modelo conforme os dados (arrays numpy) em uma pasta de entrada, salvando o modelo e o 
# histórico na pasta de saída
def treina_modelo(input_dir: str, y_dir: str, output_dir: str, model_type: str ='resunet chamorro', epochs=150, early_stopping=True, 
                  early_loss=False, loss='cross', weights=[0.25, 0.75], gamma=2, metric=F1Score(), best_model_filename = 'best_model',
                  train_with_dataset=False, lr_decay=False):
    '''
    

    e talvez correction=False
    Parameters
    ----------
    input_dir : str
        Relative Path for Input Folder, e.g, r"entrada/".
    output_dir : str
        Relative Path for Output Folder, e.g, r"resultados/".
    model_type : str, optional
        DESCRIPTION. The default is 'resunet'.
     : TYPE
        DESCRIPTION.
    lr_decay=Dicionário, por exemplo, {initial_lrate = 0.1,
             drop = 0.5,
             epochs_drop = 10.0}, ou None, para sem learning rate decay 

    Returns
    -------
    None.

    '''
    # TODO
    # Completar código para train_with_dataset
    
    # Marca tempo total do treinamento. Começa Contagem
    start = time.time()
    
    # Lê arrays salvos
    '''
    if correction:
        x_train = np.load(input_dir + 'pred_train.npy')
        x_valid = np.load(input_dir + 'pred_valid.npy')
    else:
        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
    '''
    if train_with_dataset:
        # Load Datasets
        train_dataset = tf.data.Dataset.load(input_dir + 'train_dataset/')
        valid_dataset = tf.data.Dataset.load(input_dir + 'valid_dataset/')
        
        print('Shape dos arrays:')
        print('Shape X Train Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(train_dataset).next()[0].numpy().shape)
        print('Shape Y Train Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(train_dataset).next()[1].numpy().shape)
        print('Shape X Valid Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(valid_dataset).next()[0].numpy().shape)
        print('Shape Y Valid Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(valid_dataset).next()[1].numpy().shape)
        print('')
        
    else:
        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
        y_train = np.load(y_dir + 'y_train.npy')
        y_valid = np.load(y_dir + 'y_valid.npy')
        
        # Faz codificação One-Hot dos Ys
        y_train = to_categorical(y_train, num_classes=2)
        y_valid = to_categorical(y_valid, num_classes=2)
        
        print('Shape dos arrays:')
        print('Shape x_train: ', x_train.shape)
        print('Shape y_train: ', y_train.shape)
        print('Shape x_valid: ', x_valid.shape)
        print('Shape y_valid: ', y_valid.shape)
        print('')

        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        if not train_with_dataset:
            with tf.device('/CPU:0'):
                x_train = tf.convert_to_tensor(x_train)
                y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                x_valid = tf.convert_to_tensor(x_valid)
                y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float32)
        
        # Livra memória no que for possível
        gc.collect()  

    
    # Constroi modelo
    if train_with_dataset:
        # input_shape = (patch_size, patch_size, image_channels)
        input_shape = iter(train_dataset).next()[0].numpy().shape
        num_classes = 2
        model = build_model(input_shape, num_classes, model_type=model_type)
        # model.summary()
        print('Input Patch Shape = ', input_shape)
        print()
        
    else:
        # input_shape = (patch_size, patch_size, image_channels)
        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        num_classes = 2
        model = build_model(input_shape, num_classes, model_type=model_type)
        # model.summary()
        print('Input Patch Shape = ', input_shape)
        print()
        
    # Compila o modelo
    learning_rate = 0.001 # Learning Rate
    #learning_rate = 0.01 # Learning Rate # não estava convergindo na primeira rede
    optimizer = Adam(learning_rate = learning_rate , beta_1=0.9) # Otimizador
    #optimizer = SGD(learning_rate = learning_rate , momentum=0.9) # Otimizador

    if loss == 'focal':
        # focal_loss = CategoricalFocalCrossentropy(alpha=weights, gamma=gamma)
        focal_loss = weighted_focal_loss(alpha=weights, gamma=gamma)
        model.compile(loss = focal_loss, optimizer=optimizer , metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'cross':
        #model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=[metric])
        model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'mse':
        optimizer = SGD(learning_rate = learning_rate)
        model.compile(loss = 'mse', optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'dice':
        model.compile(loss = dice_loss, optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'jaccard':
        model.compile(loss = jaccard_distance_loss, optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'wcross':
        model.compile(loss = weighted_categorical_crossentropy(weights), optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
        
        
     # Definição dos Outros hiperparâmetros
    batch_size = 16
    
    epochs = epochs

    # Parâmetros do Early Stopping
    early_stopping = early_stopping
    early_stopping_epochs = 50
    #early_stopping_epochs = 2
    #early_stopping_delta = 0.001 # aumento delta (percentual de diminuição da perda) equivalente a 0.1%
    early_stopping_delta = 0.001 # aumento delta (percentual de diminuição da perda) equivalente a 0.1%

    if train_with_dataset:
        shuffle_buffer = 100
        train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer).batch(batch_size=batch_size).prefecth(buffer_size=1)
    
    print('Hiperparâmetros:')
    print('Modelo:', model_type)
    print('Batch Size:', batch_size)
    print('Epochs:', epochs)
    print('Early Stopping:', early_stopping)
    print('Early Stopping Epochs:', early_stopping_epochs)
    print('Early Stopping Delta:', early_stopping_delta)
    print('Otimizador:', 'Adam')
    print('Learning Rate:', learning_rate)
    print('Beta 1:', 0.9)
    print('Função de Perda:', loss)
    print('Gamma para Focal Loss:', gamma)
    print()
        
    # Nome do modelo a ser salvo
    best_model_filename = best_model_filename
    

    
    # Treina o modelo
    if train_with_dataset:
        pass
    else:
        # Testa se a métrica é string
        if isinstance(metric, str):
            history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                         output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                         metric_name=metric, lr_decay=lr_decay, train_with_dataset=False)
        
        # Testa se é instância
        elif isinstance(metric, object):
            history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                             output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                             metric_name=metric.__class__.__name__.lower(), lr_decay=lr_decay, train_with_dataset=False)
            
        # Testa se é função
        elif isinstance(metric, (types.FunctionType, types.BuiltinFunctionType)):
            history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                             output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                             metric_name=metric.__name__, lr_decay=lr_decay, train_with_dataset=False)                                                            

        

    # Imprime e salva história do treinamento
    print('history = \n', history)
    
    # Marca tempo total do treinamento. Encerra Contagem
    end = time.time()
    
    # Salva histórico e inclui no arquivo texto a contagem de tempo gasto
    with open(os.path.join(output_dir, 'history_' + best_model_filename + '.txt'), 'w') as f:
        f.write('history = \n')
        f.write(str(history))
        f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos')
        
    with open(os.path.join(output_dir, 'history_pickle_' +  best_model_filename + '.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
        pickle.dump(history, fp)
        
    # Mostra histórico em gráfico
    show_graph_loss_accuracy(np.asarray(history), 1, metric_name='F1-Score', save=True, save_path=output_dir)
    
    # Escreve hiperparâmetreos e modelo usados no Diretório
    with open(os.path.join(output_dir, 'model_configuration_used.txt'), 'w') as f:
        f.writelines([f'Model Type = {model_type}\n', f'Batch Size = {batch_size}\n', f'Epochs = {epochs}\n',
                      f'Early Stopping Epochs = {early_stopping_epochs}\n', f'Early Stopping Delta = {early_stopping_delta}\n',
                      f'Learning Rate = {learning_rate}\n', f'Função de perda = {loss}\n', f'Optimizer = {str(type(optimizer))}\n'])
        if model_type == 'unet transformer':
            f.write(str(config_dict) + '\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    
    
        




def treina_modelo_ensemble(input_dir: str, output_dir_ensemble: str, n_members: int = 10, model_type: str ='resunet', epochs=150, 
                           early_stopping=True, early_loss=True, loss='focal', gamma=2, metric=F1Score(), 
                           best_model_filename = 'best_model'):
    # Loop para treinar vários modelos diversas vezes
    for i in range(n_members):
        treina_modelo(input_dir, output_dir_ensemble, model_type=model_type, epochs=epochs, early_stopping=early_stopping, 
                      early_loss=early_loss, loss=loss, gamma=gamma, metric=metric, 
                      best_model_filename=best_model_filename + '_' + str(i+1))
        

def transfer_learning_segformer(input_dir, y_dir, output_dir, model_checkpoint, model_weights_filename, learning_rate=0.001, epochs=2000, 
                                early_stopping_epochs=50, batch_size=16):
    # Marca tempo total do treinamento. Começa Contagem
    start = time.time()
    
    # Load Arrays
    x_train = np.load(input_dir + 'x_train_fine.npy')
    x_valid = np.load(input_dir + 'x_valid_fine.npy')
    y_train = np.load(y_dir + 'y_train_fine.npy')
    y_valid = np.load(y_dir + 'y_valid_fine.npy')
    
    # Convert them to tensors
    with tf.device('/CPU:0'):
        x_train = tf.convert_to_tensor(x_train)
        x_valid = tf.convert_to_tensor(x_valid)
        y_train = tf.convert_to_tensor(y_train)
        y_valid = tf.convert_to_tensor(y_valid)
        
    # Livra memória no que for possível
    gc.collect()  
    
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
    
    # Freeze Encoder
    model.layers[0].trainable = False
    
    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    
    # Build callbacks
    # Com Early Stopping
    # Early Stopping por Loss
    early_stopping_delta = 0.001
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                  min_delta=early_stopping_delta, verbose=1)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(Path(output_dir) / model_weights_filename),
                                                      monitor='val_loss',
                                                      mode='min',
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      save_freq='epoch',
                                                      save_best_only=True)    
    callbacks = [early_stop, cp_callback]
    
    
    # Fit Model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        callbacks=callbacks,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Retorna resultados de Treino e Validação
    historia = history.history
    
    lista_loss = historia['loss']
    lista_metric = [0]*len(lista_loss)
    lista_val_loss = historia['val_loss']
    lista_val_metric = [0]*len(lista_loss)
    
    history_train = [np.array([dupla]) for dupla in zip(lista_loss, lista_metric)]
    history_valid = [np.array([dupla]) for dupla in zip(lista_val_loss, lista_val_metric)]
    history = [ history_train, history_valid ]
    
    # Imprime e salva história do treinamento
    print('history = \n', history)
    
    # Marca tempo total do treinamento. Encerra Contagem
    end = time.time()
    
    # Salva histórico e inclui no arquivo texto a contagem de tempo gasto
    with open(os.path.join(output_dir, 'history_' + Path(model_weights_filename).stem + '.txt'), 'w') as f:
        f.write('history = \n')
        f.write(str(history))
        f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos')
        
    with open(os.path.join(output_dir, 'history_pickle_' +  Path(model_weights_filename).stem + '.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
        pickle.dump(history, fp)
        
    # Mostra histórico em gráfico
    show_graph_loss_accuracy(np.asarray(history), 1, metric_name='F1-Score', save=True, save_path=output_dir)
    
    # Escreve hiperparâmetreos e modelo usados no Diretório
    with open(os.path.join(output_dir, 'model_configuration_used.txt'), 'w') as f:
        f.writelines([f'Model Type = {type(model)}\n', f'Batch Size = {batch_size}\n', f'Epochs = {epochs}\n',
                      f'Early Stopping Epochs = {early_stopping_epochs}\n', f'Early Stopping Delta = {early_stopping_delta}\n',
                      f'Learning Rate = {learning_rate}\n', 'Função de perda = Loss do Modelo\n', f'Optimizer = {str(type(optimizer))}\n'])
        model.summary(print_fn=lambda x: f.write(x + '\n'))
                
        
    return history

        



