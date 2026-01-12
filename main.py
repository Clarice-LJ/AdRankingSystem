"""
Main pipeline for Advertisement Ranking System
Orchestrates the entire flow: data loading -> preprocessing -> retrieval -> ranking -> auction -> display
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.preprocessing.data_processor import CriteoPreprocessor, split_data
from src.retrieval.retrieval import EmbeddingRetrieval
from src.features.feature_engineering import FeatureEngine
from src.ranking.models import RankingModel
from src.auction.auction import AuctionMechanism, BidGenerator
from src.display.display import ResultsDisplay


class AdDataset(Dataset):
    """PyTorch Dataset for ad data"""

    def __init__(self, df, sparse_cols, dense_cols):
        self.df = df
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get sparse features as dict
        sparse_features = {col: torch.tensor(row[col], dtype=torch.long)
                          for col in self.sparse_cols}

        # Get label
        label = torch.tensor(row['label'], dtype=torch.float32)

        return sparse_features, label


class AdRankingPipeline:
    """Main pipeline for ad ranking system"""

    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 80)
        print("AD RANKING SYSTEM INITIALIZED")
        print("=" * 80)
        print(f"Model Type: {self.config['ranking']['model_type']}")
        print(f"Auction Mechanism: {self.config['auction']['mechanism']}")
        print(f"Sample Ratio: {self.config['data']['sample_ratio']}")
        print("=" * 80 + "\n")

        # Initialize components
        self.preprocessor = CriteoPreprocessor(self.config['preprocessing'])
        self.feature_engine = FeatureEngine(self.config['features'])
        self.retrieval = EmbeddingRetrieval(self.config['retrieval'])
        self.auction = AuctionMechanism(self.config['auction'])
        self.display = ResultsDisplay(self.config['display'])

        # Data
        self.train_df = None
        self.val_df = None
        self.test_df = None

        # Model
        self.ranking_model = None

        # Feature columns
        self.dense_cols = [f'I{i}' for i in range(1, 14)]
        self.sparse_cols = [f'C{i}' for i in range(1, 27)]

    def load_and_preprocess_data(self, data_path, download=False):
        """Step 1: Load and preprocess Criteo dataset"""
        print("\n" + "=" * 80)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("=" * 80 + "\n")

        if download:
            # Download dataset using kagglehub
            import kagglehub as kh
            path = kh.dataset_download("mrkmakr/criteo-dataset")
            data_path = path + "/dac/train.txt"
            print(f"Dataset downloaded to: {data_path}")

        # Load data
        sample_ratio = self.config['data']['sample_ratio']
        df = self.preprocessor.load_data(data_path, sample_ratio=sample_ratio)

        # Preprocess
        df = self.preprocessor.preprocess(df, is_train=True)

        # Split data
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        self.train_df, self.val_df, self.test_df = split_data(df, train_ratio, val_ratio)

        # Save preprocessor
        os.makedirs('models', exist_ok=True)
        self.preprocessor.save_preprocessor('models')

        print(f"✓ Data preprocessing complete")
        print(f"  - Train set: {len(self.train_df)} samples")
        print(f"  - Val set: {len(self.val_df)} samples")
        print(f"  - Test set: {len(self.test_df)} samples\n")

    def build_retrieval_index(self):
        """Step 2: Build retrieval index"""
        print("\n" + "=" * 80)
        print("STEP 2: BUILDING RETRIEVAL INDEX")
        print("=" * 80 + "\n")

        # Build embeddings for training data
        embeddings, transformer = self.retrieval.build_embeddings(
            self.train_df,
            self.preprocessor.feature_dict,
            self.sparse_cols
        )

        # Index ads (using indices as ad IDs)
        ad_ids = np.arange(len(self.train_df))
        self.retrieval.index_ads(embeddings, ad_ids)

        # Save retrieval index
        self.retrieval.save('models')

        print(f"✓ Retrieval index built with {len(ad_ids)} ads\n")

    def engineer_features(self):
        """Step 3: Feature engineering"""
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 80 + "\n")

        # Apply feature engineering to all datasets
        self.train_df = self.feature_engine.engineer_features(
            self.train_df, self.dense_cols, self.sparse_cols, is_train=True
        )

        self.val_df = self.feature_engine.engineer_features(
            self.val_df, self.dense_cols, self.sparse_cols, is_train=False
        )

        self.test_df = self.feature_engine.engineer_features(
            self.test_df, self.dense_cols, self.sparse_cols, is_train=False
        )

        # Save feature engine
        self.feature_engine.save('models')

        print(f"✓ Feature engineering complete\n")

    def train_ranking_model(self):
        """Step 4: Train ranking model (CTR prediction)"""
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING RANKING MODEL")
        print("=" * 80 + "\n")

        model_type = self.config['ranking']['model_type']

        # Get feature dimensions for sparse features
        feature_dims = {}
        for col in self.sparse_cols:
            feature_dims[col] = self.train_df[col].max() + 1

        # Initialize model
        self.ranking_model = RankingModel(
            model_type=model_type,
            config=self.config['ranking'],
            feature_dims=feature_dims
        )

        if self.ranking_model.is_neural_network():
            # Create data loaders for neural network models
            train_dataset = AdDataset(self.train_df, self.sparse_cols, self.dense_cols)
            val_dataset = AdDataset(self.val_df, self.sparse_cols, self.dense_cols)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['ranking']['batch_size'],
                shuffle=True,
                num_workers=0
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['ranking']['batch_size'],
                shuffle=False,
                num_workers=0
            )

            # Train
            history = self.ranking_model.train(
                train_loader,
                val_loader,
                epochs=self.config['ranking']['epochs']
            )
        else:
            # For traditional ML models
            X_train = self.train_df[self.sparse_cols + self.dense_cols].values
            y_train = self.train_df['label'].values

            history = self.ranking_model.train((X_train, y_train))

        # Save model
        self.ranking_model.save('models')

        print(f"\n✓ Ranking model training complete\n")

    def predict_ctr(self, df):
        """Predict CTR for a dataframe"""
        if self.ranking_model.is_neural_network():
            # Create dataset and dataloader
            dataset = AdDataset(df, self.sparse_cols, self.dense_cols)
            loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)

            predictions = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.ranking_model.model.eval()

            with torch.no_grad():
                for batch in tqdm(loader, desc="Predicting CTR"):
                    features, _ = batch
                    features = {k: v.to(device) for k, v in features.items()}
                    outputs = self.ranking_model.model(features)
                    predictions.extend(outputs.cpu().numpy().flatten())

            return np.array(predictions)
        else:
            # For traditional ML models
            X = df[self.sparse_cols + self.dense_cols].values
            return self.ranking_model.predict(X)

    def run_auction_and_display(self, test_sample_size=1000):
        """Step 5 & 6: Run auction and display results"""
        print("\n" + "=" * 80)
        print("STEP 5: RUNNING AUCTION")
        print("=" * 80 + "\n")

        # Sample test data for demonstration
        test_sample = self.test_df.sample(n=min(test_sample_size, len(self.test_df)))

        # Predict CTR (quality scores)
        print("Predicting CTR for test samples...")
        quality_scores = self.predict_ctr(test_sample)

        # Generate bids (in practice, these come from advertisers)
        bid_generator = BidGenerator()
        bids = bid_generator.generate_bids_from_ctr(quality_scores, conversion_value=10.0)

        # Run auction
        ad_ids = test_sample.index.values
        auction_result = self.auction.run_auction(bids, quality_scores, ad_ids)

        print(f"✓ Auction complete with {len(auction_result['ad_ids'])} ads\n")

        # Display results
        print("\n" + "=" * 80)
        print("STEP 6: DISPLAYING RESULTS")
        print("=" * 80 + "\n")

        self.display.display_auction_results(auction_result)

        return auction_result, quality_scores

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80 + "\n")

        # Predict on test set
        print("Predicting on test set...")
        y_pred = self.predict_ctr(self.test_df)
        y_true = self.test_df['label'].values

        # Compute metrics
        metrics = self.display.compute_metrics(y_true, y_pred)

        # Create visualizations
        print("\nGenerating visualizations...")
        os.makedirs('results', exist_ok=True)

        self.display.plot_ctr_distribution(y_pred, y_true, save_path='results')
        self.display.plot_roc_curve(y_true, y_pred, save_path='results')
        self.display.plot_precision_recall_curve(y_true, y_pred, save_path='results')

        return metrics

    def run_full_pipeline(self, data_path, download=False):
        """Run the complete pipeline"""
        print("\n" + "=" * 80)
        print("STARTING FULL AD RANKING PIPELINE")
        print("=" * 80 + "\n")

        # Step 1: Load and preprocess data
        self.load_and_preprocess_data(data_path, download=download)

        # Step 2: Build retrieval index
        self.build_retrieval_index()

        # Step 3: Feature engineering
        self.engineer_features()

        # Step 4: Train ranking model
        self.train_ranking_model()

        # Step 5: Evaluate model
        self.evaluate_model()

        # Step 6 & 7: Run auction and display
        auction_result, quality_scores = self.run_auction_and_display()

        # Generate auction analysis plots
        print("\nGenerating auction analysis...")
        self.display.plot_auction_analysis(auction_result, save_path='results')

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print("Results saved to 'results/' directory")
        print("Models saved to 'models/' directory")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    pipeline = AdRankingPipeline('config/config.yaml')

    # Run full pipeline
    # Note: Set download=True to download the dataset from Kaggle
    # Make sure you have kagglehub installed and authenticated
    pipeline.run_full_pipeline(
        data_path='data/train.txt',  # or use download=True
        download=False
    )
