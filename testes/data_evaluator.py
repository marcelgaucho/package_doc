# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:19:42 2024

@author: Marcel
"""

import numpy as np
from package_doc.avaliacao.buffer_functions import buffer_patches_array
import json

# %% Directories

x_dir = 'teste_x/'
y_dir = 'teste_y/'
output_dir = 'teste_saida/'

# %% Load Arrays

x_test = np.load(x_dir + 'x_test.npy')
y_test = np.load(y_dir + 'y_test.npy')
pred_test = np.load(output_dir + 'pred_test.npy')


# %%


class RelaxedMetricCalculator:
    def __init__(self, y_array, pred_array, buffer_px):
        self.y_array = y_array
        self.pred_array = pred_array
        
        self.buffer_px = buffer_px
        
    def _set_buffers(self):
        self.buffer_y_array = buffer_patches_array(self.y_array, radius_px=self.buffer_px)
        self.buffer_pred_array = buffer_patches_array(self.pred_array, radius_px=self.buffer_px)
    
    def calculate_metrics(self):
        # Calculate buffers
        self._set_buffers()
        
        # Epsilon (to not divide by 0)
        epsilon = 1e-7
        
        # For relaxed precision (use buffer on Y to calculate true positives)
        true_positive_relaxed_precision = (self.buffer_y_array*self.pred_array).sum()
        predicted_positive = self.pred_array.sum()
        
        # For relaxed recall (use buffer on Prediction to calculate true positives)
        true_positive_relaxed_recall = (self.y_array*self.buffer_pred_array).sum() 
        actual_positive = self.y_array.sum()

        # Calculate relaxed precision and relaxed recall
        relaxed_precision = true_positive_relaxed_precision / (predicted_positive + epsilon)
        relaxed_recall = true_positive_relaxed_recall / (actual_positive + epsilon)
        
        # Calculate relaxed F1 from relaxed precision and relaxed recall
        relaxed_f1 = (2 * relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall + epsilon)
        
        # Dictionary of metrics
        self.metrics = {'relaxed_precision': relaxed_precision, 'relaxed_recall': relaxed_recall, 'relaxed_f1': relaxed_f1}
        
        return self.metrics
    


class TestRelaxedMetricCalculator(RelaxedMetricCalculator):
    def export_results(self, output_dir):
        if not hasattr(self, 'metrics'):
            raise Exception("Metrics weren't calculated. First is necessary to "
                            "calculate metrics with the method calculate_metrics")
        
        with open(output_dir + f'relaxed_metrics_test_{self.buffer_px}px.json', 'w') as f:
            json.dump(self.metrics, f)
            
            
class ValidRelaxedMetricCalculator(RelaxedMetricCalculator):
    def export_results(self, output_dir):
        if not hasattr(self, 'metrics'):
            raise Exception("Metrics weren't calculated. First is necessary to "
                            "calculate metrics with the method calculate_metrics")
        
        with open(output_dir + f'relaxed_metrics_valid_{self.buffer_px}px.json', 'w') as f:
            json.dump(self.metrics, f)
        
# %%   
    
test_metric = TestRelaxedMetricCalculator(y_array=y_test, pred_array=pred_test, buffer_px=3)
test_metric.calculate_metrics()
test_metric.export_results(output_dir)
        
