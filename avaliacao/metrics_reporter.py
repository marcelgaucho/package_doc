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
    
    # Translation dictionary stored as a class attribute for easy editing
    COLUMN_TRANSLATIONS = {
        'Model': 'Membro',
        'relaxed_precision': 'Precisão',
        'relaxed_recall': 'Sensibilidade',
        'relaxed_f1': 'F1',
        'ece': 'ECE'
    }    
    
    def __init__(self, base_output_dir: str, n_models: int):
        self.base_output_dir = Path(base_output_dir)
        self.n_models = n_models
        
    @staticmethod
    def _format_metrics_to_percentage(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
        """Helper: Converts float metric columns to rounded percentage numbers (e.g., 0.85432 -> 85.43)."""
        df_formatted = df.copy()        
        
        # Select all floating-point metric columns (automatically ignores strings)
        float_cols = [
            col for col in df_formatted.select_dtypes(include=['float', 'float64']).columns    
        ]
        
        # Vectorized multiplication and rounding (retains float64 dtype)
        for col in float_cols:
            df_formatted[col] = (df_formatted[col] * 100).round(2)
            
        return df_formatted
        
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
            # 1. Compute summary stats on raw decimal floats
            numeric_df = df.drop(columns=['Model'])
            mean_row = pd.DataFrame([{'Model': 'MEAN', **numeric_df.mean().to_dict()}])
            median_row = pd.DataFrame([{'Model': 'MEDIAN', **numeric_df.median().to_dict()}])
            std_row = pd.DataFrame([{'Model': 'STD', **numeric_df.std().to_dict()}])
            
            # Append summary rows to the bottom of the table
            df = pd.concat([df, mean_row, median_row, std_row], ignore_index=True)
            
            # 2. Scale to percentage scale (0–100) and round to 2 decimal places (retains float dtype)
            df = self._format_metrics_to_percentage(df)
            
            # 3. Apply Presentation Layer translation right before export
            df = df.rename(columns=self.COLUMN_TRANSLATIONS)
            
        if export_csv and not df.empty:
            output_path = self.base_output_dir / f'ensemble_mosaic_summary_{buffer_px}px.csv'
            df.to_csv(output_path, index=False)
            print(f"Summary table exported to: {output_path.name}")
            
        return df
            
            