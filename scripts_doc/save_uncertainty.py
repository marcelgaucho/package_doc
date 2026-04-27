# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:25:09 2025

@author: Marcel
"""

# Calculate uncertainty and store it in ensemble dir

# %% Imports

from osgeo import gdal
from package_doc.entropy.uncertain import UncertaintyCalculator, DataGroups
from package_doc.entropy.dir_uncertain import UncertaintyMetric, XDirUncertain 
from package_doc.avaliacao.mosaics import MosaicGenerator
from package_doc.avaliacao.utils import stack_uneven

from pathlib import Path
import numpy as np
import re
import json

# %% Disable GPU and limit number of CPU threads (and thus the number of cpu cores) to use in evaluation

# from osgeo import gdal # First import gdal when it gives error
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Print 0 gpus (gpu disabled)

cpu_threads = 2

print(f"Number of CPU threads used in evaluation: {cpu_threads}") 

tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)


# %% Configuration dirs

outputs_dir = r'experimentos_deforestation/out_resunet/'
outputs_dir = Path(outputs_dir)

model_dirs = [d for d in outputs_dir.iterdir() if re.match('m_\d', d.name) and d.is_dir()]

folder_uncertainty = outputs_dir / 'uncertainty'

y_dir = Path('experimentos_deforestation/y_dir/')

label_tiles_dir =  'tiles_t2_preprocessed/test/' # dir of test labels
label_tiles_dir = Path(label_tiles_dir)

in_x_dir = r'experimentos_deforestation/x_dir/' # Original x_dir
out_x_dir = r'experimentos_deforestation/out_resunet/x_dir/' # New x_dir with uncertainty


# %% Parameters

min_target_scale = 0
max_target_scale = 1
perc_cut = None

data_group = DataGroups.Test

model_name = 'resunet'

metric_uncertainty = UncertaintyMetric.Entropy

prefix = f'mosaic_{metric_uncertainty.value}_'
export_mosaics = True

scale_result = True

# Load information
with open(y_dir / 'info_tiles_test.json') as fp:   
    info_tiles_test = json.load(fp)

ignore_index = 255

# %% Create directory

folder_uncertainty.mkdir(exist_ok=True)

# %% Create ensemble object

uncertainty_calc = UncertaintyCalculator(model_dirs=model_dirs, data_group=data_group, 
                                         scale_result=scale_result)

# %% Export metric

if metric_uncertainty == UncertaintyMetric.Entropy:
    entropy = uncertainty_calc.entropy(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                               perc_cut=perc_cut)
    np.save(folder_uncertainty / f'{UncertaintyMetric.Entropy.value}_{data_group.value}.npy', entropy)
    uncertainty_array = entropy
elif metric_uncertainty == UncertaintyMetric.Surprise:
    surprise = uncertainty_calc.surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                 perc_cut=perc_cut)
    np.save(folder_uncertainty / f'{UncertaintyMetric.Surprise.value}_{data_group.value}.npy', surprise)
    uncertainty_array = surprise
elif metric_uncertainty == UncertaintyMetric.WeightedSurprise:
    weighted_surprise = uncertainty_calc.weighted_surprise(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                                   perc_cut=perc_cut)
    np.save(folder_uncertainty / f'{UncertaintyMetric.WeightedSurprise.value}_{data_group.value}.npy', weighted_surprise)
    uncertainty_array = weighted_surprise
elif metric_uncertainty == UncertaintyMetric.ProbMean:
    prob_mean = uncertainty_calc.prob_mean(min_target_scale=min_target_scale, max_target_scale=max_target_scale,
                                   perc_cut=perc_cut)
    np.save(folder_uncertainty / f'{UncertaintyMetric.ProbMean.value}_{data_group.value}.npy', prob_mean)
    uncertainty_array = prob_mean

# %% Load reference mosaics and check if exists ignored index

# Load reference mosaics
labels_paths = [str(path) for path in label_tiles_dir.iterdir() 
                     if path.suffix=='.tiff' or path.suffix=='.tif']
labels_paths.sort()
y_mosaics = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
y_mosaics = stack_uneven(y_mosaics)[..., np.newaxis]

ignore_in_y_mosaics = ignore_index in y_mosaics    


# %% Export mosaics and set ignored index (if exists)  	

if export_mosaics:
    mosaics = MosaicGenerator(test_array=uncertainty_array, 
                              info_tiles=info_tiles_test, 
                              tiles_dir=label_tiles_dir,
                              output_dir=folder_uncertainty)
    mosaics.build_mosaics()
    if ignore_in_y_mosaics:
        mosaics.mosaics[y_mosaics == ignore_index] = ignore_index   
    mosaics.export_mosaics(prefix=prefix)

# %% Create new x dir and insert data

x_dir_uncer = XDirUncertain(in_x_folder=in_x_dir, y_folder=y_dir, 
                            out_x_folder=out_x_dir, 
                            model_dirs=model_dirs,
                            metric=metric_uncertainty, 
                            min_scale_uncertainty=min_target_scale, 
                            max_scale_uncertainty=max_target_scale,
                            perc_cut=perc_cut)

x_dir_uncer.create()

x_dir_uncer.insert_data() 