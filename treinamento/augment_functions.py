# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:38:37 2024

@author: Marcel
"""

import random
import tensorflow as tf

def transform_augment_or_maintain(x, y):
    # Sorteia opção
    lista_opcoes = [0, 1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
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
    
    
def transform_augment(x_y):
    x, y = x_y
    
    # Sorteia opção
    lista_opcoes = [1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
    # Espelhamento Vertical (Flip)
    if opcao == 1:
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
        return (x, y)
    

def transform_augment_2arg(x, y):
    # Sorteia opção
    lista_opcoes = [1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
    # Espelhamento Vertical (Flip)
    if opcao == 1:
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