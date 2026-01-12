"""
Ranking models for CTR (Click-Through Rate) prediction
Implements multiple models: LR, FM, DeepFM, XGBoost, LightGBM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb


class DeepFM(nn.Module):
    """
    DeepFM: Factorization-Machine based Neural Network for CTR Prediction
    Combines the power of factorization machines and deep neural networks
    """

    def __init__(self, feature_dims, embedding_dim=8, hidden_dims=[256, 128, 64], dropout=0.2):
        super(DeepFM, self).__init__()

        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(feature_dims)

        # Embedding layers for sparse features
        self.embeddings = nn.ModuleDict()
        self.linear_weights = nn.ModuleDict()

        total_linear_dim = 0
        total_embedding_dim = 0

        for field_name, field_dim in feature_dims.items():
            # Embedding for FM and DNN
            self.embeddings[field_name] = nn.Embedding(field_dim, embedding_dim)
            total_embedding_dim += embedding_dim

            # Linear weights for FM first-order
            self.linear_weights[field_name] = nn.Embedding(field_dim, 1)
            total_linear_dim += 1

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

        # Deep component
        layers = []
        input_dim = total_embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.dnn = nn.Sequential(*layers)

    def fm_layer(self, embeddings):
        """
        Factorization Machine layer
        Computes second-order feature interactions
        """
        # Sum of squares
        square_of_sum = torch.sum(embeddings, dim=1) ** 2

        # Square of sum
        sum_of_square = torch.sum(embeddings ** 2, dim=1)

        # FM interaction
        fm_output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return fm_output

    def forward(self, x):
        """
        x: dict of {field_name: indices}
        """
        # Linear part (first-order)
        linear_output = self.bias

        # Embeddings for FM and DNN
        embedding_list = []

        for field_name, indices in x.items():
            # Linear
            linear_output += self.linear_weights[field_name](indices).squeeze(1)

            # Embedding
            emb = self.embeddings[field_name](indices)
            embedding_list.append(emb)

        # Stack embeddings
        embeddings = torch.stack(embedding_list, dim=1)  # [batch_size, num_fields, embedding_dim]

        # FM second-order interactions
        fm_output = self.fm_layer(embeddings)

        # DNN part
        dnn_input = embeddings.view(embeddings.size(0), -1)  # Flatten
        dnn_output = self.dnn(dnn_input)

        # Combine all parts
        output = linear_output + fm_output + dnn_output

        return torch.sigmoid(output)


class WideAndDeep(nn.Module):
    """
    Wide & Deep model
    Combines memorization (wide) and generalization (deep)
    """

    def __init__(self, wide_dim, deep_dims, embedding_dim=8, hidden_dims=[256, 128, 64], dropout=0.2):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model)
        self.wide = nn.Linear(wide_dim, 1)

        # Deep part
        layers = []
        input_dim = sum(deep_dims.values()) * embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        # Embeddings for deep part
        self.embeddings = nn.ModuleDict()
        for field_name, field_dim in deep_dims.items():
            self.embeddings[field_name] = nn.Embedding(field_dim, embedding_dim)

    def forward(self, wide_x, deep_x):
        """
        wide_x: features for wide part
        deep_x: dict of {field_name: indices} for deep part
        """
        # Wide part
        wide_output = self.wide(wide_x)

        # Deep part
        embedding_list = []
        for field_name, indices in deep_x.items():
            emb = self.embeddings[field_name](indices)
            embedding_list.append(emb)

        deep_input = torch.cat(embedding_list, dim=1)
        deep_output = self.deep(deep_input)

        # Combine
        output = wide_output + deep_output

        return torch.sigmoid(output)


class RankingModel:
    """
    Wrapper class for different ranking models
    """

    def __init__(self, model_type, config, feature_dims=None):
        self.model_type = model_type
        self.config = config
        self.model = None

        if model_type == "deepfm":
            self.model = DeepFM(
                feature_dims=feature_dims,
                embedding_dim=config.get('embedding_dim', 8),
                hidden_dims=config.get('hidden_dims', [256, 128, 64]),
                dropout=config.get('dropout', 0.2)
            )
        elif model_type == "lr":
            self.model = LogisticRegression(max_iter=100)
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                objective='binary:logistic'
            )
        elif model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                objective='binary'
            )

    def is_neural_network(self):
        """Check if model is a neural network"""
        return self.model_type in ["deepfm", "wide_deep"]

    def train(self, train_loader, val_loader=None, epochs=5):
        """Train the model"""
        if self.is_neural_network():
            return self._train_neural_network(train_loader, val_loader, epochs)
        else:
            return self._train_traditional_model(train_loader)

    def _train_neural_network(self, train_loader, val_loader, epochs):
        """Train neural network models"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                features, labels = batch
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion, device)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        return history

    def _train_traditional_model(self, train_data):
        """Train traditional ML models (LR, XGBoost, LightGBM)"""
        X_train, y_train = train_data
        self.model.fit(X_train, y_train)
        return {}

    def evaluate(self, data_loader, criterion, device):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                features, labels = batch
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device).float().unsqueeze(1)

                outputs = self.model(features)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict(self, X):
        """Make predictions"""
        if self.is_neural_network():
            self.model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            with torch.no_grad():
                if isinstance(X, dict):
                    X = {k: v.to(device) for k, v in X.items()}
                    outputs = self.model(X)
                else:
                    outputs = self.model(X.to(device))

                return outputs.cpu().numpy()
        else:
            return self.model.predict_proba(X)[:, 1]

    def save(self, save_path):
        """Save the model"""
        if self.is_neural_network():
            torch.save(self.model.state_dict(), f'{save_path}/{self.model_type}_model.pth')
        else:
            import pickle
            with open(f'{save_path}/{self.model_type}_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

        print(f"Model saved to {save_path}")

    def load(self, load_path):
        """Load the model"""
        if self.is_neural_network():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(f'{load_path}/{self.model_type}_model.pth', map_location=device))
        else:
            import pickle
            with open(f'{load_path}/{self.model_type}_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

        print(f"Model loaded from {load_path}")
