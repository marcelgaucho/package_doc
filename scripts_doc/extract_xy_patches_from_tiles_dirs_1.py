# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:01:06 2024

@author: Marcel
"""

# Extract x and y patches from directories of tiles

# This is done according to parameters

# The patch_size is the size of the patch

# The overlap is the overlap between patches (in range [0, 1[), e.g.,
# 0.25 means 25% of overlap between patches (in horizontal and vertical directions)

# border patches extraction (with a maximum of one border patch per line/column)
# are regulated by border_patches parameter, that is set set as False for 
# train and validation groups, and True for test group, in order to build the mosaics 

# The filter_nodata is used to only extract patches without 
# pixels with nodata_value,
# with a certain tolerance set by nodata_tolerance, e.g.,
# nodata_tolerance = 10 and nodata_value = (255, 255, 255)
# only allows patches with the maximum of 10 pixels 
# with value (255, 255, 255). 
# The filter_nodata is set to True in train and validation groups, 
# to filter these patches, 
# and to False in test group, to not filter
# these patches, in order to build the mosaics 

# The filter_object is used to only extract patches with a
# certain amount of pixels of the object (road in this case).
# This inhibits the extraction of patches with only background.
# The object_value is the pixel value of the object class.
# The threshold_percentage is the maximum percentage, 
# relative to the patch's total pixels, 
# that excludes patches from the set. 
# Therefore, only patches with object percentage 
# above the threshold_percentage are selected.  
# The filter_object is set to True in train and validation groups, 
# to filter these patches, 
# and to False in test group, to not filter
# these patches, in order to build the mosaics 

# if overwrite_arrays is set to True, the arrays written in the output directories
# overwrite the existent arrays

# if overwrite_dir is set to True, the tensorflow dataset directory is overwritten in
# the output directory

# if onehot_y_patches is set to True, the y patches are one-hot codified when exporting
# to the tensorflow dataset format. This is the default behavior.

# if overwrite_info is set to True, the JSON metadata is overwritten 
# in the output directory

# %% Import Libraries

from package_doc.extracao.patches_extraction import DirPatchExtractor
from pathlib import Path

# %% X and Y Input and Output Directories

group = 'test' # train, valid or test group
in_x_dir = fr'dataset_massachusetts_mnih_exp/{group}/input/'
in_y_dir = fr'dataset_massachusetts_mnih_exp/{group}/maps/'
out_x_dir = 'teste_x2/'
out_y_dir = 'teste_y2/'

# %% Create output directories if they don't exist

if not Path(out_x_dir).exists():
    Path(out_x_dir).mkdir()
    
if not Path(out_y_dir).exists():
    Path(out_y_dir).mkdir()

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

 





