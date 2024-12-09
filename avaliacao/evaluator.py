# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:45:51 2024

@author: Marcel
"""
# %% Imports general

from osgeo import gdal
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
import json

# %% Import from own modules

from ..treinamento.arquiteturas.unetr_2d import Patches
from ..treinamento.arquiteturas.segformer_tf_k2.models.modules import MixVisionTransformer
from ..treinamento.arquiteturas.segformer_tf_k2.models.Head import SegFormerHead
from ..treinamento.arquiteturas.segformer_tf_k2.models.utils import ResizeLayer

from ..avaliacao.compute_functions import RelaxedMetricCalculator, stack_uneven
from ..avaliacao.mosaic_functions import MosaicGenerator

# %% Class definition


# TODO
# To average precision is prob array and not pred array
class ModelEvaluator:
    def __init__(self, x_dir, y_dir, output_dir, label_tiles_dir=None):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.output_dir = output_dir
        self.label_tiles_dir = label_tiles_dir
        
        self.model = load_model(output_dir + 'best_model' + '.keras', compile=False, custom_objects={"Patches": Patches, 
                                                                                                     "MixVisionTransformer": MixVisionTransformer,
                                                                                                     "SegFormerHead": SegFormerHead,
                                                                                                     "ResizeLayer": ResizeLayer})
        
        self._set_numpy_arrays()
        
    def _set_numpy_arrays(self):
        # Load X and Y Train
        self.x_train = np.load(self.x_dir + 'x_train.npy')
        self.y_train = np.load(self.y_dir + 'y_train.npy')        
        
        # Load X and Y Valid
        self.x_valid = np.load(self.x_dir + 'x_valid.npy')
        self.y_valid = np.load(self.y_dir + 'y_valid.npy')
        
        # Load X and Y Test
        self.x_test = np.load(self.x_dir + 'x_test.npy')
        self.y_test = np.load(self.y_dir + 'y_test.npy')
        
    def model_predict(self, x_array, batch_step=2):
        prob = self.model.predict(x_array, batch_size=batch_step, verbose=1) # Calculate Probabilities
        
        pred = np.argmax(prob, axis=-1)[..., np.newaxis] # Calculate Predictions
        
        prob = prob[..., 1:2] # Only road probabilites are selected (Class 1)

        return prob, pred
    
    def _evaluate_train(self, buffers_px, include_avg_precision):
        # Predict Train
        if not ( ( Path(self.output_dir) / 'prob_train.npy').exists() and ( Path(self.output_dir) / 'pred_train.npy').exists() ):
            prob_train, pred_train = self.model_predict(x_array=self.x_train)
        
            # Cast to type to occupy less space para tipos que ocupam menos espaço
            prob_train = prob_train.astype(np.float16)
            pred_train = pred_train.astype(np.uint8)
            
            np.save(self.output_dir + 'prob_train.npy', prob_train)
            np.save(self.output_dir + 'pred_train.npy', pred_train)
            
        # Load probabilities and prediction
        if include_avg_precision:
            prob_train = np.load(self.output_dir + 'prob_train.npy')
        else:
            prob_train = None
            
        pred_train = np.load(self.output_dir + 'pred_train.npy')
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=self.y_train, pred_array=pred_train, buffer_px=buffer_px, prob_array=prob_train)
            metric_calculator.calculate_metrics(include_avg_precision=include_avg_precision)
            metric_calculator.export_results(output_dir=self.output_dir, group='train') 
            
    def _evaluate_valid(self, buffers_px, include_avg_precision):
        # Predict Valid
        if not ( ( Path(self.output_dir) / 'prob_valid.npy').exists() and ( Path(self.output_dir) / 'pred_valid.npy').exists() ) :
            prob_valid, pred_valid = self.model_predict(x_array=self.x_valid)
        
            # Cast to type to occupy less space para tipos que ocupam menos espaço
            prob_valid = prob_valid.astype(np.float16)
            pred_valid = pred_valid.astype(np.uint8)
            
            np.save(self.output_dir + 'prob_valid.npy', prob_valid)
            np.save(self.output_dir + 'pred_valid.npy', pred_valid)
            
        # Load probabilities and prediction
        if include_avg_precision:
            prob_valid = np.load(self.output_dir + 'prob_valid.npy')
        else:
            prob_valid = None
        
        pred_valid = np.load(self.output_dir + 'pred_valid.npy')
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=self.y_valid, pred_array=pred_valid, buffer_px=buffer_px, prob_array=prob_valid)
            metric_calculator.calculate_metrics(include_avg_precision=include_avg_precision)
            metric_calculator.export_results(output_dir=self.output_dir, group='valid')
            
    def _evaluate_test(self, buffers_px, include_avg_precision):
        # Predict Test
        if not ( ( Path(self.output_dir) / 'prob_test.npy').exists() and ( Path(self.output_dir) / 'pred_test.npy').exists() ):
            prob_test, pred_test = self.model_predict(x_array=self.x_test)
        
            # Cast to type to occupy less space para tipos que ocupam menos espaço
            prob_test = prob_test.astype(np.float16)
            pred_test = pred_test.astype(np.uint8)
            
            np.save(self.output_dir + 'prob_test.npy', prob_test)
            np.save(self.output_dir + 'pred_test.npy', pred_test)
            
        # Load probabilities and prediction
        if include_avg_precision:
            prob_test = np.load(self.output_dir + 'prob_test.npy')
        else:
            prob_test = None
        
        pred_test = np.load(self.output_dir + 'pred_test.npy')
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=self.y_test, pred_array=pred_test, buffer_px=buffer_px, prob_array=prob_test)
            metric_calculator.calculate_metrics(include_avg_precision=include_avg_precision)
            metric_calculator.export_results(output_dir=self.output_dir, group='test')  
        
    def evaluate_model(self, evaluate_train=False, evaluate_valid=True, evaluate_test=True, buffers_px=[3],
                       include_avg_precision=False):
        # Evaluate train
        if evaluate_train:
            self._evaluate_train(buffers_px=buffers_px, include_avg_precision=include_avg_precision)
        
        # Evaluate valid
        if evaluate_valid:
            self._evaluate_valid(buffers_px=buffers_px, include_avg_precision=include_avg_precision)    
            
        # Evaluate test
        if evaluate_test:
            self._evaluate_test(buffers_px=buffers_px, include_avg_precision=include_avg_precision) 
            
    def build_test_mosaics(self, prefix='outmosaic'):
        if self.label_tiles_dir is None:
            raise Exception("Couldn't build mosaics because the label tiles directory was not informed")
            
        print('Mosaics will be builded with pred_test saved in output directory')
        # Load probabilities and prediction
        prob_test = np.load(self.output_dir + 'prob_test.npy')
        pred_test = np.load(self.output_dir + 'pred_test.npy')
        
        # Load information
        with open(Path(self.y_dir) / 'info_tiles_test.json') as fp:   
            info_tiles_test = json.load(fp)
        
        # Build, save and export mosaics for pred
        mosaics = MosaicGenerator(test_array=pred_test, info_tiles=info_tiles_test, tiles_dir=self.label_tiles_dir,
                                  output_dir=self.output_dir)
        mosaics.build_mosaics()
        mosaics.save_mosaics(prefix='pred')
        mosaics.export_mosaics(prefix=prefix+'_pred_')
        
        # Build, save and export mosaics for prob
        mosaics = MosaicGenerator(test_array=prob_test, info_tiles=info_tiles_test, tiles_dir=self.label_tiles_dir,
                                  output_dir=self.output_dir)
        mosaics.build_mosaics()
        mosaics.save_mosaics(prefix='prob')
        mosaics.export_mosaics(prefix=prefix+'_prob_')
        
    def evaluate_mosaics(self, buffers_px=[3], include_avg_precision=False):
        # Load probabilities and prediction for mosaics
        if include_avg_precision:
            prob_mosaics = np.load(self.output_dir + 'prob_mosaics.npy')
        else:
            prob_mosaics = None
            
        pred_mosaics = np.load(self.output_dir + 'pred_mosaics.npy')
        
        # Load reference mosaics
        labels_paths = [str(path) for path in Path(self.label_tiles_dir).iterdir() 
                             if path.suffix=='.tiff' or path.suffix=='.tif']
        labels_paths.sort()
        y_mosaics = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
        y_mosaics = stack_uneven(y_mosaics)[..., np.newaxis]
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=y_mosaics, pred_array=pred_mosaics, buffer_px=buffer_px, prob_array=prob_mosaics)
            metric_calculator.calculate_metrics(include_avg_precision=include_avg_precision)
            metric_calculator.export_results(output_dir=self.output_dir, group='mosaics')  