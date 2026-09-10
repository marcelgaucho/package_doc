# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:05:35 2026

@author: Marcel
"""

# %% Import Libraries and setup hardware

from osgeo import gdal
from package_doc.geral.utils import setup_hardware
from pathlib import Path

# Setup hardware
setup_hardware(cpu_threads=8, gpu_memory_limit=19456) # Multiple of 1024 MiB (1 GiB)

from package_doc.treinamento.arquiteturas.models import build_model
from package_doc.treinamento.arquiteturas.unetr_2d_dict import config_dict
from tensorflow.keras.optimizers import Adam
from package_doc.geral.ensemble_manager import EnsembleManager
from package_doc.geral.ensemble_config import EnsembleConfig
from package_doc.treinamento.metrics import CustomF1Score
from package_doc.treinamento.fine_tuning import LayerIndexStrategy
from package_doc.avaliacao.cross_reporter import CrossExperimentReporter
from package_doc.exibicao.figure_generator import FigureGenerator

from tensorflow.keras.metrics import Precision, Recall

# %%

def main():
    # 1. Load the merged configuration
    config = EnsembleConfig.from_yaml('package_doc/exp_config/experiment_01_mnih.yaml', 
                                      'package_doc/exp_config/base_config_mnih.yaml')
    
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
    eval_metrics = [CustomF1Score(), Precision(class_id=1), Recall(class_id=1)]
    
    # --- TRAIN ---
    if config.run_training:
        train_kwargs = config.train_kwargs.copy()
        train_kwargs.update({'metrics_train': eval_metrics, 'metrics_val': eval_metrics})
        manager.train_all(optimizer_class=Adam, train_kwargs=train_kwargs)
    
    # --- FINE TUNE ---
    if config.run_finetune:
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
    if config.run_evaluation:
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
    if config.run_uncertainty:
        manager.calculate_uncertainty(**config.uncertainty_kwargs)
        
    # --- GLOBAL REPORTS ---
    if config.run_aggregate_report:
        # Get base dir and buffers list from config
        base_directory = config.base_exp_dir
        buffers_list = config.eval_mosaic_kwargs.get('buffers_px', [0])
                
        # Initialize the reporter pointing to the root experiments folder
        global_reporter = CrossExperimentReporter(base_exp_dir=base_directory)
        
        # Generate the master summary tables for the buffers list
        master_tables = global_reporter.generate_master_summaries(
            buffers_px=buffers_list, 
            export_csv=True,
            export_excel=True
        )   
        
        # Optionally display the first few rows of the first master table in the console
        try:
            first_master_table = next(iter(master_tables.values()))
        except StopIteration:
            raise ValueError("No master tables were generated in the dictionary.")
        
        if not first_master_table.empty:
            print("\nFirst Master Summary Preview:")
            print(first_master_table.head())
            
    # --- COMPARISON PLOT ---
    if config.save_comparison_plot:
        base_directory_path = Path(config.base_exp_dir)
        
        # Map plot display titles to experiment identifiers
        experiment_map = {
            'Standard CE': 'cross_standard',
            'U-CE': 'cross_uce'
        }
        
        figure_generator = FigureGenerator(
            base_exp_dir=config.base_exp_dir,
            master_csv_path=base_directory_path / 'master_experiment_summary_0px.csv',
            experiment_map=experiment_map
        )
        
        # Plot figure with a patch that highlights the benefits of uncertainty-aware losses
        TARGET_PATCH = 42
        figure_generator.generate_figure(patch_idx=TARGET_PATCH, 
                                         save_path=base_directory_path / "comparison_grid.pdf",
                                         strategy='median')
            

# %%

if __name__ == "__main__":
    main()