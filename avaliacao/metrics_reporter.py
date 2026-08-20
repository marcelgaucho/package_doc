# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:19:40 2026

@author: Marcel
"""

# %% Import Libraries

import pandas as pd
import json
from pathlib import Path

# %%

class MetricsReporter:
    """Responsible for aggregating distributed metric JSONs into unified tabular reports."""
    
    def __init__(self, base_output_dir: str, n_models: int):
        self.base_output_dir = Path(base_output_dir)
        self.n_models = n_models
        
    def generate_mosaic_summary_table(self, buffer_px: int = 0, export_csv: bool = True) -> pd.DataFrame:
        """Aggregates mosaic metrics across all models into a single DataFrame."""
        print(f"\n--- Generating Mosaic Summary Table ({buffer_px}px buffer) ---")
        
        all_records = []
        
        for i in range(self.n_models):
            model_name = f'm_{i}'
            metrics_path = self.base_output_dir / model_name / f'relaxed_metrics_mosaics_{buffer_px}px.json'
            # Skip model if doesn't exist
            if not metrics_path.exists():
                print(f"  [!] Missing metrics for {model_name}. Skipping.")
                continue
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            # Prepend the model identifier to the record for the table
            record = {'Model': model_name}
            record.update(metrics)
            all_records.append(record)
            
        # Convert the list of dictionaries into a Pandas DataFrame
        df = pd.DataFrame(all_records)
        
        # Calculate average, median and standard deviation of the Ensemble metrics
        if not df.empty:
            numeric_df = df.drop(columns=['Model'])
            mean_row = pd.DataFrame([{'Model': 'MEAN', **numeric_df.mean().to_dict()}])
            median_row = pd.DataFrame([{'Model': 'MEDIAN', **numeric_df.median().to_dict()}])
            std_row = pd.DataFrame([{'Model': 'STD', **numeric_df.std().to_dict()}])
            
            # Append summary rows to the bottom of the table
            df = pd.concat([df, mean_row, median_row, std_row], ignore_index=True)
            
        if export_csv and not df.empty:
            output_path = self.base_output_dir / f'ensemble_mosaic_summary_{buffer_px}px.csv'
            df.to_csv(output_path, index=False)
            print(f"Summary table exported to: {output_path.name}")
            
        return df
            
            