# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:39:22 2024

@author: Marcel
"""

from osgeo import gdal
from package_doc.functions_bib import preprocessa_fine_tuning

# %%

input_dir = 'entrada/'
y_dir = 'y_directory/'

# %% 

preprocessa_fine_tuning(input_dir, y_dir)