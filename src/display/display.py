"""
Display and visualization module
Shows ranking results, metrics, and visualizations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, roc_curve


class ResultsDisplay:
    """Display and visualize ad ranking results"""

    def __init__(self, config):
        self.config = config
        self.top_n = config.get('top_n', 10)
        self.show_metrics = config.get('show_metrics', True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)

    def display_auction_results(self, auction_result):
        """Display auction results in a formatted table"""
        print("\n" + "=" * 80)
        print("AUCTION RESULTS - TOP {} ADS".format(self.top_n))
        print("=" * 80)

        # Create DataFrame
        df = pd.DataFrame({
            'Rank': range(1, min(self.top_n + 1, len(auction_result['ad_ids']) + 1)),
            'Ad ID': auction_result['ad_ids'][:self.top_n],
            'Bid': auction_result['bids'][:self.top_n],
            'Price': auction_result['prices'][:self.top_n],
            'Quality Score': auction_result['quality_scores'][:self.top_n],
            'Rank Score': auction_result['rank_scores'][:self.top_n]
        })

        print(df.to_string(index=False))

        # Summary statistics
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80)
        print(f"Total Ads in Auction: {len(auction_result['ad_ids'])}")
        print(f"Total Revenue: ${np.sum(auction_result['prices']):.2f}")
        print(f"Average Price: ${np.mean(auction_result['prices']):.2f}")
        print(f"Average Quality Score: {np.mean(auction_result['quality_scores']):.4f}")
        print(f"Average Bid: ${np.mean(auction_result['bids']):.2f}")
        print("=" * 80 + "\n")

    def plot_ctr_distribution(self, predicted_ctrs, actual_clicks=None, save_path=None):
        """Plot CTR distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of predicted CTRs
        axes[0].hist(predicted_ctrs, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Predicted CTR')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Predicted CTR')
        axes[0].grid(True, alpha=0.3)

        # If actual clicks available, show calibration
        if actual_clicks is not None:
            # Bin by predicted CTR and compute actual CTR
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(predicted_ctrs, bins)

            bin_predicted = []
            bin_actual = []

            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_predicted.append(np.mean(predicted_ctrs[mask]))
                    bin_actual.append(np.mean(actual_clicks[mask]))

            axes[1].scatter(bin_predicted, bin_actual, s=100, alpha=0.6)
            axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            axes[1].set_xlabel('Predicted CTR')
            axes[1].set_ylabel('Actual CTR')
            axes[1].set_title('CTR Calibration Plot')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(f'{save_path}/ctr_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}/ctr_distribution.png")

        plt.show()

    def plot_roc_curve(self, y_true, y_pred, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(f'{save_path}/roc_curve.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}/roc_curve.png")

        plt.show()

    def plot_precision_recall_curve(self, y_true, y_pred, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(f'{save_path}/pr_curve.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}/pr_curve.png")

        plt.show()

    def plot_feature_importance(self, feature_names, importances, top_k=20, save_path=None):
        """Plot feature importance"""
        # Sort by importance
        indices = np.argsort(importances)[-top_k:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.tight_layout()

        if save_path:
            plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}/feature_importance.png")

        plt.show()

    def plot_auction_analysis(self, auction_result, save_path=None):
        """Plot auction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Bid vs Price
        axes[0, 0].scatter(auction_result['bids'], auction_result['prices'], alpha=0.6)
        axes[0, 0].plot([0, max(auction_result['bids'])], [0, max(auction_result['bids'])], 'r--', label='Bid = Price')
        axes[0, 0].set_xlabel('Bid')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].set_title('Bid vs Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Quality Score distribution
        axes[0, 1].hist(auction_result['quality_scores'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Quality Score (Predicted CTR)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Quality Score Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Rank Score vs Position
        positions = np.arange(1, len(auction_result['rank_scores']) + 1)
        axes[1, 0].plot(positions[:50], auction_result['rank_scores'][:50], marker='o')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Rank Score')
        axes[1, 0].set_title('Rank Score vs Position (Top 50)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Revenue by Position
        cumulative_revenue = np.cumsum(auction_result['prices'][:50])
        axes[1, 1].bar(positions[:50], auction_result['prices'][:50], alpha=0.6, label='Per Position')
        axes[1, 1].plot(positions[:50], cumulative_revenue, 'r-', marker='o', label='Cumulative')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Revenue')
        axes[1, 1].set_title('Revenue Analysis (Top 50)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(f'{save_path}/auction_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}/auction_analysis.png")

        plt.show()

    def compute_metrics(self, y_true, y_pred):
        """Compute and display evaluation metrics"""
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)

        print("\n" + "=" * 80)
        print("MODEL EVALUATION METRICS")
        print("=" * 80)
        print(f"AUC (Area Under ROC Curve): {auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print("=" * 80 + "\n")

        return {'auc': auc, 'log_loss': logloss}
