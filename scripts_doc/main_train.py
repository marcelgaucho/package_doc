# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:05:35 2026

@author: Marcel
"""

# %% Import Libraries

from osgeo import gdal
from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict
from tensorflow.keras.optimizers import Adam
from package_doc.treinamento.custom_loss import (masked_weighted_cce, masked_cce, custom_entropy_loss, 
                                                 custom_add_entropy_loss,
                                                 custom_offset_entropy_loss)
from package_doc.entropy.uncertain import DataGroups
from package_doc.entropy.utils import UncertaintyMetric #, uncert_metric_method
from package_doc.geral.ensemble_manager import EnsembleManager
from package_doc.geral.utils import setup_hardware
from package_doc.geral.ensemble_config import EnsembleConfig
from package_doc.treinamento.metrics import MaskedPrecision, MaskedRecall, MaskedF1Score
from package_doc.treinamento.fine_tuning import LayerIndexStrategy



# %%

def main():
    # 0. Setup hardware
    setup_hardware(cpu_threads=1, gpu_memory_limit=20480)
    
    # 1. Build the model
    model_params = {
        'input_shape': (64, 64, 14),
        'n_classes': 2,
        'model_type': 'resunet',
        'config_dict': config_dict,
        'dropout_rate': 0
        }
    model = build_model(**model_params)
    
    # 2. Initialize the Orchestrator
    ensemble_params = {
            'x_dir': 'experimentos_deforestation/x_dir/',
            'y_dir': 'experimentos_deforestation/y_dir/',
            'base_output_dir': 'experimentos_deforestation/out_resunet_nobuffer_teste/',
            'base_model': model,
            'n_models': 2
            }
    manager = EnsembleManager(**ensemble_params)
    
    # 3. Execute Pipeline Steps cleanly
    # --- TRAIN ---
    manager.train_all(
        optimizer_class=Adam,
        train_kwargs={
            'epochs': 2000,
            'early_stopping_epochs': 1,
            'batch_size': 16,
            'metrics_train': [MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
            'metrics_val': [MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
            'learning_rate': 0.0001,
            'loss_fn': custom_offset_entropy_loss,
            'uncertainty_dir': 'experimentos_deforestation/out_resunet_nobuffer/uncertainty/',
            'uncertainty_metric': UncertaintyMetric.StdDev
            # ... other args
        }
    )
    
    # --- FINE TUNE ---
    strategy = LayerIndexStrategy(fine_tune_at=31, 
                                  learning_rate=1e-5)
    manager.fine_tune_all(
        optimizer_class=Adam,
        strategy=strategy,
        base_models_dir='experimentos_deforestation/out_resunet_nobuffer/',
        fine_tune_kwargs={
            'epochs': 2000,
            'early_stopping_epochs': 1,
            'batch_size': 16,
            'metrics_train': [MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
            'metrics_val': [MaskedF1Score(), MaskedPrecision(), MaskedRecall()],
            'loss_fn': masked_cce,
            'uncertainty_dir': None,
            'uncertainty_metric': UncertaintyMetric.StdDev
            # ... other args
        }
    )
    
    # --- EVALUATE ---
    manager.evaluate_all(
        label_tiles_dir='tiles_t2_preprocessed_nobuffer/test/',
        eval_kwargs={'splits': ['valid', 'test'],
                     'buffers_px': [0],
                     'include_avg_precision': False
                     },
        mosaic_kwargs={'export_pred_mosaics': False, 
                       'export_prob_mosaics': False,
                       'ignore_index': 255,
                       'min_area_px': 69
                       },
        eval_mosaic_kwargs={'buffers_px': [0],
                     'include_avg_precision': False,
                     'min_area_px': 69
                     }
    )

    # --- UNCERTAINTY ---
    manager.calculate_uncertainty(
        data_groups=[DataGroups.Train, DataGroups.Valid, DataGroups.Test],
        scale_result=True,
        metric=UncertaintyMetric.StdDev,
        metric_kwargs={
            'min_target_scale': 0,
            'max_target_scale': 1,
            'perc_cut': None
        },
        label_tiles_dir='tiles_t2_preprocessed_nobuffer/test/',
        info_tiles_name='info_tiles_test.json',
        ignore_index=255,
        export_mosaics=True,
        generate_plot=True
    )

# %%

if __name__ == "__main__":
    main()