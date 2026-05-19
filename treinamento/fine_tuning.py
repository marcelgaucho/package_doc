#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:57:24 2026

@author: rotunno
"""

# %% Import libraries

from abc import ABC, abstractmethod
import tensorflow as tf

# %% Fine Tuning strategy

class FineTuneStrategy(ABC):
    """Abstract base class defining the contract for any fine-tuning behavior."""
    def __init__(self, learning_rate: float = 1e-5):
        self.learning_rate = learning_rate

    @abstractmethod
    def apply(self, model: tf.keras.Model) -> tf.keras.Model:
        """Modifies layer trainability (freezing/unfreezing)."""
        pass


class LayerIndexStrategy(FineTuneStrategy):
    """Fine-tunes the model from a specific layer index onward."""
    def __init__(self, fine_tune_at: int, learning_rate: float = 1e-5):
        super().__init__(learning_rate)
        self.fine_tune_at = fine_tune_at

    def apply(self, model: tf.keras.Model) -> tf.keras.Model:
        # Unfreeze the whole model first
        model.trainable = True
        
        # Freeze all layers up to the specified index
        for layer in model.layers[:self.fine_tune_at]:
            layer.trainable = False
            
        print(f"[{self.__class__.__name__}] Unfroze layers from index {self.fine_tune_at} onwards.")
        return model


class BlockNameStrategy(FineTuneStrategy):
    """Fine-tunes specific blocks by matching strings in layer names (e.g., 'block5')."""
    def __init__(self, target_block_names: list, learning_rate: float = 1e-5):
        super().__init__(learning_rate)
        self.target_block_names = target_block_names

    def apply(self, model: tf.keras.Model) -> tf.keras.Model:
        model.trainable = True
        
        # Freeze layers unless their name matches our target blocks
        for layer in model.layers:
            if any(block in layer.name for block in self.target_block_names):
                layer.trainable = True
            else:
                layer.trainable = False
                
        print(f"[{self.__class__.__name__}] Unfroze layers matching blocks: {self.target_block_names}")
        return model