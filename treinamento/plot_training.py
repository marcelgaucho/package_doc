# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:04:53 2024

@author: Marcel
"""

import numpy as np
from matplotlib import pyplot as plt

# Função que mostra gráfico
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
        plt.savefig(save_path+save_name)
        plt.close()
    else:
        plt.show(block=False)