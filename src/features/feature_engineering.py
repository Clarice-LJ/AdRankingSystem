"""
Feature engineering module
Creates additional features like cross features, statistical features, etc.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import pickle


class FeatureEngine:
    """Feature engineering for ad ranking"""

    def __init__(self, config):
        self.config = config
        self.create_cross_features = config.get('cross_features', True)
        self.create_statistical_features = config.get('statistical_features', True)
        self.scalers = {}
        self.cross_feature_cols = []
        self.stat_feature_cols = []

    def create_cross_feature_pairs(self, df, sparse_cols, max_pairs=5):
        """
        Create cross features from categorical columns
        Cross features capture feature interactions
        """
        if not self.create_cross_features:
            return df

        print("Creating cross features...")

        # Select important feature pairs (in practice, use feature importance)
        # For demonstration, we'll use the first few categorical features
        selected_cols = sparse_cols[:min(5, len(sparse_cols))]

        cross_features = []
        for col1, col2 in combinations(selected_cols, 2):
            cross_col_name = f'{col1}_{col2}_cross'
            df[cross_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

            # Hash to integer to keep dimensionality manageable
            df[cross_col_name] = df[cross_col_name].apply(
                lambda x: hash(x) % 10000
            )

            cross_features.append(cross_col_name)

            if len(cross_features) >= max_pairs:
                break

        self.cross_feature_cols = cross_features
        print(f"Created {len(cross_features)} cross features")

        return df

    def create_statistical_features(self, df, dense_cols, group_cols=None):
        """
        Create statistical features
        E.g., aggregations, ratios, differences
        """
        if not self.create_statistical_features:
            return df

        print("Creating statistical features...")

        stat_features = []

        # Create ratio features between numerical columns
        if len(dense_cols) >= 2:
            # Ratio of first two dense features
            df['dense_ratio_1_2'] = df[dense_cols[0]] / (df[dense_cols[1]] + 1e-8)
            stat_features.append('dense_ratio_1_2')

        # Create sum and mean features
        df['dense_sum'] = df[dense_cols].sum(axis=1)
        df['dense_mean'] = df[dense_cols].mean(axis=1)
        df['dense_std'] = df[dense_cols].std(axis=1)
        df['dense_max'] = df[dense_cols].max(axis=1)
        df['dense_min'] = df[dense_cols].min(axis=1)

        stat_features.extend(['dense_sum', 'dense_mean', 'dense_std', 'dense_max', 'dense_min'])

        self.stat_feature_cols = stat_features
        print(f"Created {len(stat_features)} statistical features")

        return df

    def create_frequency_features(self, df, sparse_cols, is_train=True):
        """
        Create frequency encoding for categorical features
        """
        print("Creating frequency features...")

        freq_features = []

        for col in sparse_cols:
            freq_col_name = f'{col}_freq'

            if is_train:
                # Calculate frequency on training data
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self.scalers[freq_col_name] = freq_map
            else:
                # Use frequency from training data
                freq_map = self.scalers.get(freq_col_name, {})

            # Map frequencies
            df[freq_col_name] = df[col].map(freq_map).fillna(0)
            freq_features.append(freq_col_name)

        print(f"Created {len(freq_features)} frequency features")

        return df

    def engineer_features(self, df, dense_cols, sparse_cols, is_train=True):
        """Main feature engineering pipeline"""

        # Create cross features
        if self.create_cross_features:
            df = self.create_cross_feature_pairs(df, sparse_cols)

        # Create statistical features
        if self.create_statistical_features:
            df = self.create_statistical_features(df, dense_cols)

        return df

    def get_all_feature_cols(self, dense_cols, sparse_cols):
        """Get all feature column names after engineering"""
        all_cols = list(dense_cols) + list(sparse_cols)

        if self.create_cross_features:
            all_cols.extend(self.cross_feature_cols)

        if self.create_statistical_features:
            all_cols.extend(self.stat_feature_cols)

        return all_cols

    def save(self, save_path):
        """Save feature engineering artifacts"""
        with open(f'{save_path}/feature_engine.pkl', 'wb') as f:
            pickle.dump({
                'cross_feature_cols': self.cross_feature_cols,
                'stat_feature_cols': self.stat_feature_cols,
                'scalers': self.scalers
            }, f)
        print(f"Feature engineering artifacts saved to {save_path}")

    def load(self, load_path):
        """Load feature engineering artifacts"""
        with open(f'{load_path}/feature_engine.pkl', 'rb') as f:
            data = pickle.load(f)
            self.cross_feature_cols = data['cross_feature_cols']
            self.stat_feature_cols = data['stat_feature_cols']
            self.scalers = data['scalers']
        print(f"Feature engineering artifacts loaded from {load_path}")
