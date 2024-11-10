# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:01:06 2024

@author: Marcel
"""

# Extract x and y patches from directories of tiles

# %% Import Libraries

from package_doc.extracao.patches_extraction import DirPatchExtractor

# %% X and Y Input and Output Directories

group = 'train' # train, valid or test group
in_x_dir = fr'dataset_massachusetts_mnih_exp/{group}/input/'
in_y_dir = fr'dataset_massachusetts_mnih_exp/{group}/maps/'
out_x_dir = 'teste_x1/'
out_y_dir = 'teste_y1/'

# %% Parameters for extraction

filter_nodata = True
nodata_value = (255, 255, 255)
nodata_tolerance = 10

filter_object = True
object_value = 1
threshold_percentage = 1

overwrite_arrays = True # overwrite .npy files
overwrite_dir = True # overwrite tensorflow dataset dir
overwrite_info = True # overwrite metadata JSON file

# %% Extract patches from dir

p_extract = DirPatchExtractor(in_x_dir=in_x_dir, 
                              in_y_dir=in_y_dir, 
                              out_x_dir=out_x_dir, 
                              out_y_dir=out_y_dir,
                              filter_nodata=filter_nodata, nodata_value=nodata_value, nodata_tolerance=nodata_tolerance,
                              filter_object=filter_object, object_value=object_value, threshold_percentage=threshold_percentage,
                              group=group)
p_extract.extract_patches_from_tiles()

# %% Normalize the extracted X Patches
 
p_extract.normalize_x_patches()

# %% One-hot Y Patches

p_extract.onehot_y_patches()

# %% Save the extracted patches in Output Directories, in numpy and tensorflow dataset formats

p_extract.save_np_arrays(overwrite_arrays=overwrite_arrays)
p_extract.save_tf_dataset(overwrite_dir=overwrite_dir)

# %% Save metadata dictionary to JSON

p_extract.save_info_tiles(overwrite_info=overwrite_info)

 





