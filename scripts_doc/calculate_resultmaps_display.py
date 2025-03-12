# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:37:37 2025

@author: marcel.rotunno
"""

# %% Import libraries

from package_doc.avaliacao.mapdisplay import resultmap, MapType
from pathlib import Path

# %% Extracted images for tile

tile_id = '23278915_15'

imgs_extraction = {r'ResUnet': r'backup_notebook_185/saida_resunet_loop_2x_16b_3/outmosaic_pred__23278915_15.tif',
                   r'UNETR': r'backup_notebook_185/saida_unetr_loop_2x_16b_0/outmosaic_pred__23278915_15.tif',
                   r'SegFormer': r'backup_notebook_185/saida_segformer_b5_loop_2x_16b_4/outmosaic_pred__23278915_15.tif',
                   r'ResUnet/ResUnet': r'backup_notebook_185/saida_resunet_entropy_road_loop_2x_16b_3/outmosaic_pred__23278915_15.tif',
                   r'UNETR/UNETR': r'backup_notebook_185/saida_unetr_entropy_road_loop_2x_16b_4/outmosaic_pred__23278915_15.tif',
                   r'SegFormer/SegFormer': r'backup_notebook_185/saida_segformer_b5_entropy_road_loop_2x_16b_3/outmosaic_pred__23278915_15.tif',
                   r'ResUnet/SegFormer': r'backup_notebook_185/saida_resunet_segformer_b5_entropy_road_loop_2x_16b_3/outmosaic_pred__23278915_15.tif'}

# %% Label

img_label = r'dataset_massachusetts_mnih_mod/test/maps/23278915_15.tiff'


# %% Output maps

output_folder = r'testes1/out_result_maps'
output_folder_path = Path(output_folder)

output_folder_path.mkdir(exist_ok=True)

output_folder_prec_path = output_folder_path / 'precision' 
output_folder_recall_path = output_folder_path / 'recall' 

output_folder_prec_path.mkdir(exist_ok=True)
output_folder_recall_path.mkdir(exist_ok=True)

output_precision_maps_paths = [output_folder_prec_path / f"{net.replace('/', '_')}_{tile_id}_precmap.tif"
                               for net in imgs_extraction]

output_recall_maps_paths = [output_folder_recall_path / f"{net.replace('/', '_')}_{tile_id}_recallmap.tif"
                               for net in imgs_extraction]


# %% Generate objects

resultmaps_prec = [resultmap(pred_path=extraction, y_path=img_label,
                             out_path=out, maptype=MapType.Precision) 
                   for extraction, out in zip(imgs_extraction.values(), output_precision_maps_paths)]

resultmaps_recall = [resultmap(pred_path=extraction, y_path=img_label,
                             out_path=out, maptype=MapType.Recall) 
                   for extraction, out in zip(imgs_extraction.values(), output_recall_maps_paths)]

# %% Export maps as tifs

for result_prec, result_recall in zip(resultmaps_prec, resultmaps_recall):
    result_prec.export()
    result_recall.export()


        