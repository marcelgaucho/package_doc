# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:09:12 2024

@author: Marcel
"""

import h5py
from osgeo import gdal
import numpy as np
from pathlib import Path
from functions_bib import extract_difference_reftiles

gdal.UseExceptions()

# %% Directories

reftiles_test_2016_dir = Path('tiles_sentinel/masks/2016/test')
reftiles_test_2018_dir = Path('tiles_sentinel/masks/2018/test')
reftiles_test_Diff_dir = Path('tiles_sentinel/masks/Diff/test')

# %%

reftiles_test_2016_list = list(reftiles_test_2016_dir.iterdir())
reftiles_test_2018_list = list(reftiles_test_2018_dir.iterdir())
reftiles_test_Diff_list = list(reftiles_test_Diff_dir.iterdir())

# %%

# Extrai Diferenca
for reftile_2016, reftile_2018 in zip(reftiles_test_2016_list, reftiles_test_2018_list):
    print(f'Extraindo Diferença de {reftile_2016} e {reftile_2018}')

    out_raster_path = reftiles_test_Diff_dir / f"diff_reftile_2018_2016_{(str(reftile_2016).split('_')[-1]).split('.tif')[0]}.tif"
    print(out_raster_path)
    
    extract_difference_reftiles(str(reftile_2016), str(reftile_2018), str(out_raster_path))
    print()
