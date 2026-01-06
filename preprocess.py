import pandas as pd
import numpy as np

def process_dense_features(df, dense_cols):
    df[dense_cols] = df[dense_cols].fillna(0)
    for col in dense_cols:
        df[col] = np.log1p(df[col])
    return df