# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:24:58 2025

@author: Marcel
"""

# Replace values in tiff files in a y directory
# Used to replace 255 with 1 in train, valid and test directories

# %% Import Libraries

from package_doc.extracao.replace_value import replace_value_y_files
import argparse

# %% Add arguments

parser = argparse.ArgumentParser(description='Replaces the specified y value with another value in an '
                                             'Y image directory')

parser.add_argument('y_dir', type=str, help='Enter the Y directory path of raster masks')
parser.add_argument('-v', '--value', metavar='y_value', type=int, required=True, help='Enter the y value to be replaced')
parser.add_argument('-s', '--substitute', metavar='y_substitute', required=True, type=int, help='Enter the new y value')

args = parser.parse_args()
# args = parser.parse_args(['y_dir', 'dataset_massachusetts_mnih_exp/train/maps',
#                           '-v', '255',
#                           '-s', '1'])

y_dir = args.y_dir
y_value = args.value
y_substitute = args.substitute

# %% Replace value

replace_value_y_files(y_dir=y_dir, old_value=y_value, new_value=y_substitute)