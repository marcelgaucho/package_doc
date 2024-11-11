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

patch_size=256

overlap=0.25 

# For train and valid groups, border patches aren't included
# For test group, the border patches are included to make the mosaic
if group == 'train' or group == 'valid':
    border_patches = False
else: # group == 'test'
    border_patches = True

# For train and valid groups, filter for patches without nodata and 
# filter for patches with road objects 
if group == 'train' or group == 'valid':
    filter_nodata = True
    nodata_value = (255, 255, 255)
    nodata_tolerance = 10
    
    filter_object = True
    object_value = 1
    threshold_percentage = 1
else: # group == 'test'
    filter_nodata = False
    nodata_value = (255, 255, 255)
    nodata_tolerance = 10
    
    filter_object = False
    object_value = 1
    threshold_percentage = 1

# Export parameters
overwrite_arrays = True # overwrite .npy files
overwrite_dir = True # overwrite tensorflow dataset dir
onehot_y_patches = True # One-hot y patches in tensorflow dataset export
overwrite_info = True # overwrite metadata JSON file

# %% Extract patches from dir

p_extract = DirPatchExtractor(in_x_dir=in_x_dir, 
                              in_y_dir=in_y_dir, 
                              out_x_dir=out_x_dir, 
                              out_y_dir=out_y_dir,
                              patch_size=patch_size,
                              overlap=overlap,
                              border_patches=border_patches,
                              filter_nodata=filter_nodata, nodata_value=nodata_value, nodata_tolerance=nodata_tolerance,
                              filter_object=filter_object, object_value=object_value, threshold_percentage=threshold_percentage,
                              group=group)
p_extract.extract_patches_from_tiles()

# %% Normalize the extracted X Patches
 
p_extract.normalize_x_patches()

# %% Save the extracted patches in Output Directories, in numpy and tensorflow dataset formats

p_extract.save_np_arrays(overwrite_arrays=overwrite_arrays)
if group == 'train' or group == 'valid':
    p_extract.save_tf_dataset(overwrite_dir=overwrite_dir, onehot_y_patches=onehot_y_patches)

# %% Save metadata dictionary to JSON

p_extract.save_info_tiles(overwrite_info=overwrite_info)

 





