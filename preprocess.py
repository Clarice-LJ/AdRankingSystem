import pandas as pd
import numpy as np

def process_dense_features(df, dense_cols):
    df[dense_cols] = df[dense_cols].fillna(0)
    for col in dense_cols:
        df[col] = np.log1p(df[col])
    return df

def chunk_load(path, file, sample_ratio, seed, usecols=None, chunksize=None, encoding=None, sep=None, names=None, dtype_dict=None):
    data_chunk = pd.read_csv(f'{path}{file}', encoding=encoding, chunksize=chunksize, sep=sep, usecols=usecols, names=names, dtype=dtype_dict)
    data_temp = []
    

def preprocess_data(chunk_seed, col_name_train):
    # load data
    df_train = chunk_load