# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:20:03 2026

@author: Marcel
"""

# %% Import Libraries

from ..entropy.utils import UncertaintyMetric, DataGroups
from .utils import merge_dicts_with_preference

from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
import yaml
# 1. Define your Loss Registry mapping strings to actual functions
# Import your custom losses here
from package_doc.treinamento.custom_loss import masked_cce, custom_offset_entropy_loss

from tensorflow.keras.losses import CategoricalCrossentropy

# %%

LOSS_REGISTRY = {
    "masked_cce": masked_cce,
    "custom_offset_entropy_loss": custom_offset_entropy_loss,
    "default_cce": CategoricalCrossentropy()
}

# %%

@dataclass
class EnsembleConfig:
    x_dir: str
    y_dir: str
    base_exp_dir: str
    base_pretrained_dir: str
    experiment_name: str
    n_models: int
    
    run_training: bool
    run_finetune: bool
    run_evaluation: bool
    run_uncertainty: bool
    
    # All configurations are now loaded as dynamic dictionaries
    model_params: dict = field(default_factory=dict)
    train_kwargs: dict = field(default_factory=dict)
    fine_tune_kwargs: dict = field(default_factory=dict)
    eval_kwargs: dict = field(default_factory=dict)
    mosaic_kwargs: dict = field(default_factory=dict)
    eval_mosaic_kwargs: dict = field(default_factory=dict)
    uncertainty_kwargs: dict = field(default_factory=dict)

    @property
    def base_output_dir(self) -> Path:
        return Path(self.base_exp_dir) / f'out_{self.experiment_name}'

    @classmethod
    def from_yaml(cls, experiment_yaml_path: str, base_yaml_path: str = 'base_config.yaml'):
        with open(base_yaml_path, 'r') as file:
            base_data = yaml.safe_load(file)
            
        with open(experiment_yaml_path, 'r') as file:
            exp_data = yaml.safe_load(file)

        # 1. Merge the raw dictionaries deeply
        data = merge_dicts_with_preference(base_data, exp_data)

        # 2. Parse Strings into Python Objects/Enums
        for phase in ['train_kwargs', 'fine_tune_kwargs']:
            if phase in data:
                kw = data[phase]
                if 'loss_fn' in kw and isinstance(kw['loss_fn'], str):
                    kw['loss_fn'] = LOSS_REGISTRY[kw['loss_fn']]
                if 'uncertainty_metric' in kw and isinstance(kw['uncertainty_metric'], str):
                    kw['uncertainty_metric'] = UncertaintyMetric[kw['uncertainty_metric']]

        # Parse Uncertainty specific enums
        if 'uncertainty_kwargs' in data:
            ukw = data['uncertainty_kwargs']
            if 'metric' in ukw and isinstance(ukw['metric'], str):
                ukw['metric'] = UncertaintyMetric[ukw['metric']]
            if 'data_groups' in ukw:
                ukw['data_groups'] = [DataGroups[g] for g in ukw['data_groups']]

        return cls(
            x_dir=data.get('x_dir'),
            y_dir=data.get('y_dir'),
            base_exp_dir=data.get('base_exp_dir'),
            base_pretrained_dir=data.get('base_pretrained_dir'),
            experiment_name=data.get('experiment_name', 'default_experiment'),
            n_models=data.get('n_models', 5),
            run_training=data.get('run_training'),
            run_finetune=data.get('run_finetune'),
            run_evaluation=data.get('run_evaluation'),
            run_uncertainty=data.get('run_uncertainty'),
            model_params=data.get('model_params', {}),
            train_kwargs=data.get('train_kwargs', {}),
            fine_tune_kwargs=data.get('fine_tune_kwargs', {}),
            eval_kwargs=data.get('eval_kwargs', {}),
            mosaic_kwargs=data.get('mosaic_kwargs', {}),
            eval_mosaic_kwargs=data.get('eval_mosaic_kwargs', {}),
            uncertainty_kwargs=data.get('uncertainty_kwargs', {})
        )