"""
Data preprocessing module for Criteo dataset
Handles loading, cleaning, and transforming the dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import os


class CriteoPreprocessor:
    def __init__(self, config):
        self.config = config
        self.dense_cols = [f'I{i}' for i in range(1, 14)]
        self.sparse_cols = [f'C{i}' for i in range(1, 27)]
        self.label_encoders = {}
        self.feature_dict = {}

    def load_data(self, file_path, nrows=None, sample_ratio=None):
        """Load Criteo dataset"""
        print(f"Loading data from {file_path}...")

        # Define column names
        col_names = ['label'] + self.dense_cols + self.sparse_cols

        # Load data in chunks if sample_ratio is provided
        if sample_ratio and sample_ratio < 1.0:
            chunk_iter = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=col_names,
                chunksize=100000
            )

            sampled_chunks = []
            for chunk in tqdm(chunk_iter, desc="Loading chunks"):
                sampled = chunk.sample(frac=sample_ratio, random_state=42)
                sampled_chunks.append(sampled)

            df = pd.concat(sampled_chunks, ignore_index=True)
        else:
            df = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=col_names,
                nrows=nrows
            )

        print(f"Loaded {len(df)} rows")
        return df

    def process_dense_features(self, df):
        """Process dense (numerical) features with log transformation"""
        print("Processing dense features...")

        # Fill missing values with 0
        df[self.dense_cols] = df[self.dense_cols].fillna(0)

        # Apply log transformation to handle long-tail distribution
        for col in self.dense_cols:
            df[col] = np.log1p(df[col])

        return df

    def process_sparse_features(self, df, is_train=True):
        """Process sparse (categorical) features with label encoding"""
        print("Processing sparse features...")

        # Fill missing values with a placeholder
        df[self.sparse_cols] = df[self.sparse_cols].fillna('missing')

        for col in tqdm(self.sparse_cols, desc="Encoding categorical features"):
            if is_train:
                # Fit label encoder on training data
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use existing label encoder
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )

        return df

    def build_feature_dict(self, df):
        """Build feature dictionary for embedding layers"""
        print("Building feature dictionary...")

        feature_index = 0
        for col in self.dense_cols:
            self.feature_dict[col] = feature_index
            feature_index += 1

        for col in self.sparse_cols:
            unique_values = df[col].nunique()
            self.feature_dict[col] = {
                'start_idx': feature_index,
                'vocab_size': unique_values
            }
            feature_index += unique_values

        return self.feature_dict

    def preprocess(self, df, is_train=True):
        """Main preprocessing pipeline"""
        # Process features
        df = self.process_dense_features(df)
        df = self.process_sparse_features(df, is_train=is_train)

        # Build feature dictionary for training data
        if is_train:
            self.build_feature_dict(df)

        return df

    def save_preprocessor(self, save_path):
        """Save label encoders and feature dictionary"""
        os.makedirs(save_path, exist_ok=True)

        with open(f'{save_path}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)

        with open(f'{save_path}/feature_dict.pkl', 'wb') as f:
            pickle.dump(self.feature_dict, f)

        print(f"Preprocessor saved to {save_path}")

    def load_preprocessor(self, load_path):
        """Load label encoders and feature dictionary"""
        with open(f'{load_path}/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)

        with open(f'{load_path}/feature_dict.pkl', 'rb') as f:
            self.feature_dict = pickle.load(f)

        print(f"Preprocessor loaded from {load_path}")


def split_data(df, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation, and test sets"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df
