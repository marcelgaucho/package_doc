# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:40:04 2024

@author: Marcel
"""

# Class and functions used in computations of evaluation

# %% Imports

import numpy as np
from sklearn.utils.extmath import stable_cumsum

import json, pickle

from .buffer_function import buffer_patches_array

# %% Class used to calculate the relaxed metrics

class RelaxedMetricCalculator:
    def __init__(self, y_array, pred_array=None, buffer_px=3, prob_array=None):
        self.y_array = y_array
        self.pred_array = pred_array
        self.prob_array = prob_array
        
        self.buffer_px = buffer_px
        
        self.buffer_y_array = buffer_patches_array(self.y_array, radius_px=self.buffer_px)
        self.buffer_pred_array = None
        
        self.metrics = None
        
        self.ap_lists = {}
    
    def _calculate_buffer_pred(self, print_interval):
        self.buffer_pred_array = buffer_patches_array(self.pred_array, radius_px=self.buffer_px, print_interval=print_interval)
        
    def _calculate_prec_recall_f1(self, value_zero_division, print_interval):
        # Give error if prediction isn't set
        assert self.pred_array is not None, "Prediction must be set, please set self.pred_array = array"
        
        # Set buffer of prediction
        self._calculate_buffer_pred(print_interval=print_interval)
        
        # Epsilon (to not divide by 0)
        epsilon = 1e-7
        
        # For relaxed precision (use buffer on Y to calculate true positives)
        true_positive_relaxed_precision = (self.buffer_y_array*self.pred_array).sum()
        predicted_positive = self.pred_array.sum()
        # false_positive_relaxed_precision = predicted_positive - true_positive_relaxed_precision
        
        # For relaxed recall (use buffer on Prediction to calculate true positives)
        true_positive_relaxed_recall = (self.y_array*self.buffer_pred_array).sum() 
        actual_positive = self.y_array.sum()
        # false_negative_relaxed_recall = actual_positive - true_positive_relaxed_recall

        # Calculate relaxed precision and relaxed recall
        relaxed_precision = true_positive_relaxed_precision / (predicted_positive + epsilon)
        relaxed_recall = true_positive_relaxed_recall / (actual_positive + epsilon)
        
        # Special case
        # If there are no actual positives, recall is 1 because "all" the positives will be discovered
        if actual_positive == 0:
            relaxed_recall = 1
                
        # Set the marked value in case of zero division, if it is setted
        if value_zero_division:
            if predicted_positive == 0:
                relaxed_precision = value_zero_division
            if actual_positive == 0:
                relaxed_recall = value_zero_division        
        
        # Calculate relaxed F1 from relaxed precision and relaxed recall
        relaxed_f1 = (2 * relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall + epsilon)
        
        # Dictionary of metrics
        metrics = {'relaxed_precision': relaxed_precision, 'relaxed_recall': relaxed_recall, 'relaxed_f1': relaxed_f1}
        
        return metrics
    
    def _calculate_avg_precision(self, print_interval, interpolated):
        ''' Calculate Thresholds '''
        # Flatten prediction array and y_true buffer
        prob_flat = self.prob_array.flatten() # maintain original for use in recall calculus
        buffer_y_array_flat = self.buffer_y_array.flatten()
        
        # Sort predictions in descending order of probabilities
        desc_score_indices = np.argsort(prob_flat, kind="mergesort")[::-1]
        prob_flat = prob_flat[desc_score_indices]
        
        # Calculate Thresholds using threshold indexes, which are the change indexes plus the final element index
        # Change indexes are the diff indexes where probability difference with next probability> 0  
        diff_scores = np.diff(prob_flat)
        change_idxs = np.where(diff_scores)[0]
        threshold_idxs = np.r_[change_idxs, prob_flat.size - 1]
        thresholds = prob_flat[threshold_idxs]
        
        ''' Calculate Cumulative TP (for Precision) and Cumulative Predicted Positives for Precision '''
        # Cumulative sum of Trues Positives in the threshold indexes
        # Threshold used in index i is pred_scores[threshold_idxs[i]]
        buffer_y_array_flat = buffer_y_array_flat[desc_score_indices] # sort buffer of y
        cumtp_prec = stable_cumsum(buffer_y_array_flat)
        cumtp_thres_prec = cumtp_prec[threshold_idxs]
        
        # Cumulative (predicted) positives in thresholds
        # Index (1-based) is used as positives, as the predictions array is sorted
        cumpos_thres = threshold_idxs + 1
        
        ''' Calculate Cumulative TP (for Recall) and Actual Positives for Recall '''
        # We can't use the sorted y_true to calculate the TPs in relaxed recall
        # because the TPs depend on the prediction buffer
        
        # Cumulative TP (for thresholds) 
        cumtp_thres_recall = []     
    
        # Loop through thresholds
        for i, threshold in enumerate(thresholds):
            if print_interval:
                if i % print_interval == 0:
                    print(f'Calculating threshold {i:>6d}/{len(thresholds):>6d}')
            
            # Predictions for the current threshold
            pred = (self.prob_array >= threshold).astype(int)
            
            # Buffer for prediction
            pred_buffer = buffer_patches_array(pred, radius_px=self.buffer_px)
            
            # True Positive for Recall
            true_positive_relaxed_recall = (self.y_array * pred_buffer).sum()
            
            # Append value in array
            cumtp_thres_recall.append(true_positive_relaxed_recall)        
    
           
        # Transform in array
        cumtp_thres_recall = np.array(cumtp_thres_recall) 
        
        # Actual positives 
        actual_positive = self.y_array.sum()    
        
        ''' Calculate Precision and Recall '''
        # Initialize and calculate precision if posible (positives != 0)
        # If not posible, precision is set to 0
        precision = np.zeros_like(cumtp_thres_prec)
        np.divide(cumtp_thres_prec, cumpos_thres, out=precision, where=(cumpos_thres != 0))
        
        # Reverse order of precision (Increasing Precision) and append 1 to final
        # If average precision is interpolated, only the maximum precision 
        # up to current position is preserved. For the interpolation, 
        # the ascending ordered precision list is used
        precision = np.hstack((precision[::-1], 1))
        if interpolated:
            precision = np.maximum.accumulate(precision)            
            
        self.ap_lists['precision'] = precision # Store precision list
        
        # Calculate recall. Set recall to 1 if there are no positive label in y_true
        if actual_positive == 0:
            print(
                "No positive class found in y_true, "
                "recall is set to one for all thresholds."
            )
            recall = np.ones_like(precision)
        else:
            recall = cumtp_thres_recall / actual_positive
            
        # Reverse order of recall (Decreasing Recall) and append 0 to final
        recall = np.hstack((recall[::-1], 0))
        self.ap_lists['recall'] = recall # Store recall list
        
        # Reverse order of thresholds
        thresholds = thresholds[::-1] 
        self.ap_lists['thresholds'] = thresholds # Store thresholds list
    
        # Calculate average precision
        avg_precision = np.sum(-np.diff(recall) * precision[:-1]) 
    
        
        return avg_precision
    
    def calculate_metrics(self, value_zero_division=None, include_avg_precision=False,
                          prob_array=None, print_interval=200, interpolated_avg_precision=True):
        # Give error if probabilities isn't set when calculating average precision
        if include_avg_precision:
            assert self.prob_array is not None, ("Probabilities array must be set to calculate average precision, "
                                                 "please set self.prob_array = array")
        
        # Calculate relaxed precision, recall and f1
        metrics = self._calculate_prec_recall_f1(value_zero_division=value_zero_division, print_interval=print_interval)
        
        # Calculate average precision
        if include_avg_precision:
            metrics['relaxed_avg_precision'] = self._calculate_avg_precision(print_interval=print_interval,
                                                                             interpolated=interpolated_avg_precision)
        # Store metrics variable
        self.metrics = metrics        
            
        return metrics        
        
    def export_results(self, output_dir, group='test', export_ap_lists=True):
        # Check if group is correct
        assert group in ('train', 'valid', 'test', 'mosaics'), "Parameter group must be 'train', 'valid', 'test' or 'mosaics'"
        
        if not hasattr(self, 'metrics'):
            raise Exception("Metrics weren't calculated. First is necessary to "
                            "calculate metrics with the method calculate_metrics")
        
        with open(output_dir / f'relaxed_metrics_{group}_{self.buffer_px}px.json', 'w') as f:
            json.dump(self.metrics, f)
            
        if export_ap_lists:
            with open(output_dir / f'avg_precision_lists_{group}_{self.buffer_px}px.pickle', "wb") as fp: 
                pickle.dump(self.ap_lists, fp)
            
            
