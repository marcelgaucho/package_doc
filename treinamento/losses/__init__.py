#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:16:54 2026

@author: rotunno
"""

# Standard Losses
from .region import get_generalized_dice_loss

# Uncertainty-weighted Losses
from .uce_distribution import get_u_categorical_crossentropy
from .uce_region import get_u_dice_loss
from .uce_compound import get_u_combo_loss