# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:54:55 2024

@author: Marcel
"""

# %% Load Libraries

from osgeo import gdal
from pathlib import Path
import numpy as np
import os, pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

# %% Desabilita GPU

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% Limita cores usados (Limita CPU) 

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Imprime 0 gpus (usa cpu)


# %% Load own libraries 
from package_doc.treinamento.arquiteturas.unetr_2d import Patches
from package_doc.treinamento.arquiteturas.segformer_tf_k2.models.modules import MixVisionTransformer
from package_doc.treinamento.arquiteturas.segformer_tf_k2.models.Head import SegFormerHead
from package_doc.treinamento.arquiteturas.segformer_tf_k2.models.utils import ResizeLayer

from package_doc.avaliacao.compute_functions import RelaxedMetricCalculator, stack_uneven
from package_doc.avaliacao.mosaic_functions import MosaicGenerator

# %% Directories

x_dir = 'teste_x/'
y_dir = 'teste_y/'
output_dir = 'saida_resunet_loop_2x_2b/'

# %% Load Arrays

x_train = np.load(x_dir + 'x_train.npy')
y_train = np.load(y_dir + 'y_train.npy')

x_valid = np.load(x_dir + 'x_valid.npy')
y_valid = np.load(y_dir + 'y_valid.npy')

x_test = np.load(x_dir + 'x_test.npy')
y_test = np.load(y_dir + 'y_test.npy')

# %% 

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
    
    def _evaluate_train(self, buffers_px=[3]):
        # Predict Train
        if not ( ( Path(output_dir) / 'prob_train.npy').exists() and ( Path(output_dir) / 'pred_train.npy').exists() ):
            prob_train, pred_train = self.model_predict(x_array=self.x_train)
        
            # Cast to type to occupy less space para tipos que ocupam menos espaço
            prob_train = prob_train.astype(np.float16)
            pred_train = pred_train.astype(np.uint8)
            
            np.save(self.output_dir + 'prob_train.npy', prob_train)
            np.save(self.output_dir + 'pred_train.npy', pred_train)
            
        # Load prediction
        pred_train = np.load(output_dir + 'pred_train.npy')
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=self.y_train, pred_array=pred_train, buffer_px=buffer_px)
            metric_calculator.calculate_metrics()
            metric_calculator.export_results(output_dir=self.output_dir, group='train') 
            
    def _evaluate_valid(self, buffers_px=[3]):
        # Predict Valid
        if not ( ( Path(output_dir) / 'prob_valid.npy').exists() and ( Path(output_dir) / 'pred_valid.npy').exists() ) :
            prob_valid, pred_valid = self.model_predict(x_array=self.x_valid)
        
            # Cast to type to occupy less space para tipos que ocupam menos espaço
            prob_valid = prob_valid.astype(np.float16)
            pred_valid = pred_valid.astype(np.uint8)
            
            np.save(self.output_dir + 'prob_valid.npy', prob_valid)
            np.save(self.output_dir + 'pred_valid.npy', pred_valid)
            
        # Load prediction
        pred_valid = np.load(output_dir + 'pred_valid.npy')
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=self.y_valid, pred_array=pred_valid, buffer_px=buffer_px)
            metric_calculator.calculate_metrics()
            metric_calculator.export_results(output_dir=self.output_dir, group='valid')
            
    def _evaluate_test(self, buffers_px=[3]):
        # Predict Test
        if not ( ( Path(output_dir) / 'prob_test.npy').exists() and ( Path(output_dir) / 'pred_test.npy').exists() ):
            prob_test, pred_test = self.model_predict(x_array=self.x_test)
        
            # Cast to type to occupy less space para tipos que ocupam menos espaço
            prob_test = prob_test.astype(np.float16)
            pred_test = pred_test.astype(np.uint8)
            
            np.save(self.output_dir + 'prob_test.npy', prob_test)
            np.save(self.output_dir + 'pred_test.npy', pred_test)
            
        # Load prediction
        pred_test = np.load(output_dir + 'pred_test.npy')
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=self.y_test, pred_array=pred_test, buffer_px=buffer_px)
            metric_calculator.calculate_metrics()
            metric_calculator.export_results(output_dir=self.output_dir, group='test')  
        
    def evaluate_model(self, evaluate_train=False, evaluate_valid=True, evaluate_test=True, buffers_px=[3]):
        # Evaluate train
        if evaluate_train:
            self._evaluate_train(buffers_px=buffers_px)
        
        # Evaluate valid
        if evaluate_valid:
            self._evaluate_valid(buffers_px=buffers_px)    
            
        # Evaluate test
        if evaluate_test:
            self._evaluate_test(buffers_px=buffers_px) 
            
    def build_test_mosaics(self, prefix='outmosaic'):
        if self.label_tiles_dir is None:
            raise Exception("Couldn't build mosaics because the label tiles directory was not informed")
            
        print('Mosaics will be builded with pred_test saved in output directory')
        # Load prediction
        pred_test = np.load(self.output_dir + 'pred_test.npy')
        
        # Load information
        with open(self.y_dir + 'info_label_tiles.pickle', "rb") as fp:   
            info_label_tiles = pickle.load(fp)
        
        # Build, save and export mosaics
        mosaics = MosaicGenerator(test_array=pred_test, info_tiles=info_label_tiles, tiles_dir=self.label_tiles_dir,
                                  output_dir=self.output_dir)
        mosaics.build_mosaics()
        mosaics.save_mosaics()
        mosaics.export_mosaics(prefix=prefix)
        
    def evaluate_mosaics(self, buffers_px=[3]):
        # Load predicted mosaics
        pred_mosaics = np.load(self.output_dir + 'pred_mosaics.npy')
        
        # Load reference mosaics
        labels_paths = [str(path) for path in Path(self.label_tiles_dir).iterdir() 
                             if path.suffix=='.tiff' or path.suffix=='.tif']
        labels_paths.sort()
        y_mosaics = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
        y_mosaics = stack_uneven(y_mosaics)[..., np.newaxis]
        y_mosaics[y_mosaics==255] = 0 # Transform 255 (supposedly NODATA) into 0 
        
        # Evaluate for buffer distances
        for buffer_px in buffers_px:
            metric_calculator = RelaxedMetricCalculator(y_array=y_mosaics, pred_array=pred_mosaics, buffer_px=buffer_px)
            metric_calculator.calculate_metrics()
            metric_calculator.export_results(output_dir=self.output_dir, group='mosaic')  
        
        
        
# %% Testa Classe

evaluator = ModelEvaluator(x_dir=x_dir, y_dir=y_dir, output_dir=output_dir)
evaluator.evaluate_model()            


    