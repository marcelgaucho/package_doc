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
    
    # 1. Load the merged configuration
    config = EnsembleConfig.from_yaml('package_doc/exp_config/experiment_01.yaml', 
                                      'package_doc/exp_config/base_config.yaml')
    
    # 2. Build the model (injecting the non-YAML python config_dict)
    model_params = config.model_params.copy()
    model_params['config_dict'] = config_dict
    model = build_model(**model_params)
    
    # 3. Initialize the Orchestrator
    manager = EnsembleManager(
        x_dir=config.x_dir,
        y_dir=config.y_dir,
        base_output_dir=str(config.base_output_dir),
        base_model=model,
        n_models=config.n_models
    )
    
    # Define metrics globally so we only instantiate them once
    eval_metrics = [MaskedF1Score(), MaskedPrecision(), MaskedRecall()]
    
    # --- TRAIN ---
    train_kwargs = config.train_kwargs.copy()
    train_kwargs.update({'metrics_train': eval_metrics, 'metrics_val': eval_metrics})
    manager.train_all(optimizer_class=Adam, train_kwargs=train_kwargs)
    
    # --- FINE TUNE ---
    fine_tune_kwargs = config.fine_tune_kwargs.copy()
    
    # Pop strategy variables out of the dictionary before passing it to train_loop
    strategy = LayerIndexStrategy(
        fine_tune_at=fine_tune_kwargs.pop('fine_tune_at', 31), 
        learning_rate=fine_tune_kwargs.pop('learning_rate', 1e-5)
    )
    fine_tune_kwargs.update({'metrics_train': eval_metrics, 'metrics_val': eval_metrics})
    
    manager.fine_tune_all(
        optimizer_class=Adam,
        strategy=strategy,
        base_models_dir=config.base_models_dir,
        fine_tune_kwargs=fine_tune_kwargs
    )
    
    # --- EVALUATE ---
    eval_kwargs = config.eval_kwargs.copy()
    # Pop the directory path out of kwargs since it is a direct parameter in evaluate_all
    label_tiles_dir = eval_kwargs.pop('label_tiles_dir') 
    
    manager.evaluate_all(
        label_tiles_dir=label_tiles_dir,
        eval_kwargs=eval_kwargs,
        mosaic_kwargs=config.mosaic_kwargs,
        eval_mosaic_kwargs=config.eval_mosaic_kwargs
    )

    # --- UNCERTAINTY ---
    manager.calculate_uncertainty(**config.uncertainty_kwargs)

# %%

if __name__ == "__main__":
    main()