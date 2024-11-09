import pandas as pd
import numpy as np

def select_metric(df, metric = 'IC50 (nM)'):
    df[metric] = pd.to_numeric(df[metric], errors = 'coerce')
    return df.dropna(subset = metric)

def clean_na_columns(df, threshold = 0.5):
    threshold = len(df) * threshold

    for col in df.columns:
        try:
            # Attempt to convert the column to numeric (forcing np.float64)
            df[col] = pd.to_numeric(df[col], errors='raise').astype(np.float64)
        except (ValueError, TypeError):
            # If conversion fails, leave the column as it is
            pass
    
    return df.dropna(axis = 1, thresh = threshold)

