# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:49:01 2026

@author: Marcel
"""

# %% Import Libraries

from typing import List
from .tile import Tile



# %%

class TileGroup:
    def __init__(self, tiles: List[Tile]):
        self.tiles = tiles
        
