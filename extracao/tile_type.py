# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:07:13 2026

@author: Marcel
"""




# %% Import Libraries

from enum import Enum



# %% Enumerated constant for tile types

class TileType(str, Enum):
    X = 'X'
    Y = 'Y'