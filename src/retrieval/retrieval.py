"""
Retrieval module for candidate ad generation
Uses embedding-based similarity search to retrieve relevant ads
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.random_projection import GaussianRandomProjection
import pickle


class EmbeddingRetrieval:
    """Simple embedding-based retrieval system"""

    def __init__(self, config):
        self.config = config
        self.top_k = config.get('top_k_candidates', 100)
        self.embedding_dim = config.get('embedding_dim', 16)
        self.ad_embeddings = None
        self.user_embeddings = None

    def build_embeddings(self, df, feature_dict, sparse_cols):
        """
        Build simple embeddings for ads and users
        In production, these would be learned from user-ad interactions
        """
        print("Building embeddings for retrieval...")

        # For simplicity, we'll use the hash of categorical features
        # In production, this would be learned embeddings from a two-tower model
        n_samples = len(df)

        # Use random projection for dimensionality reduction
        # This simulates learned embeddings
        n_features = len(sparse_cols)
        feature_matrix = df[sparse_cols].values

        # Create random projection
        transformer = GaussianRandomProjection(
            n_components=self.embedding_dim,
            random_state=42
        )

        embeddings = transformer.fit_transform(feature_matrix)

        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings, transformer

    def index_ads(self, ad_embeddings, ad_ids):
        """Index ad embeddings for fast retrieval"""
        self.ad_embeddings = ad_embeddings
        self.ad_ids = ad_ids
        print(f"Indexed {len(ad_ids)} ads")

    def retrieve(self, query_embeddings, top_k=None):
        """
        Retrieve top-k similar ads for each query
        Uses cosine similarity
        """
        if top_k is None:
            top_k = self.top_k

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(query_embeddings, self.ad_embeddings.T)

        # Get top-k indices for each query
        top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]

        return top_k_indices

    def save(self, save_path):
        """Save retrieval index"""
        with open(f'{save_path}/retrieval_index.pkl', 'wb') as f:
            pickle.dump({
                'ad_embeddings': self.ad_embeddings,
                'ad_ids': self.ad_ids
            }, f)
        print(f"Retrieval index saved to {save_path}")

    def load(self, load_path):
        """Load retrieval index"""
        with open(f'{load_path}/retrieval_index.pkl', 'rb') as f:
            data = pickle.load(f)
            self.ad_embeddings = data['ad_embeddings']
            self.ad_ids = data['ad_ids']
        print(f"Retrieval index loaded from {load_path}")


class TwoTowerRetrieval(nn.Module):
    """
    Two-tower model for retrieval (user tower and ad tower)
    This is a more sophisticated approach used in production systems
    """

    def __init__(self, user_feature_dims, ad_feature_dims, embedding_dim=64):
        super(TwoTowerRetrieval, self).__init__()

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(sum(user_feature_dims.values()), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Ad tower
        self.ad_tower = nn.Sequential(
            nn.Linear(sum(ad_feature_dims.values()), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, user_features, ad_features):
        user_emb = self.user_tower(user_features)
        ad_emb = self.ad_tower(ad_features)

        # Normalize embeddings
        user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=1)
        ad_emb = torch.nn.functional.normalize(ad_emb, p=2, dim=1)

        return user_emb, ad_emb

    def compute_similarity(self, user_emb, ad_emb):
        """Compute cosine similarity"""
        return torch.matmul(user_emb, ad_emb.t())
