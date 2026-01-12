# Advertisement Ranking System

A complete advertisement ranking system using the Criteo dataset, implementing the full pipeline from data preprocessing to auction-based ad ranking and results visualization.

## Overview

This project implements a production-grade online advertising ranking system covering:

1. **Data Preprocessing** - Loading and transforming Criteo dataset with log transformation for long-tail features
2. **Retrieval** - Embedding-based candidate ad generation
3. **Feature Engineering** - Cross features, statistical features, and frequency encoding
4. **Ranking** - CTR prediction using DeepFM, Wide&Deep, XGBoost, or LightGBM
5. **Auction** - GSP (Generalized Second Price), VCG, or First-Price auction mechanisms
6. **Display** - Results visualization with metrics and analysis

## Project Structure

```
AdRankingSystem/
├── config/
│   └── config.yaml          # Configuration file
├── data/                    # Dataset directory
├── models/                  # Saved models and artifacts
├── results/                 # Output visualizations and results
├── src/
│   ├── preprocessing/       # Data preprocessing module
│   │   └── data_processor.py
│   ├── retrieval/          # Candidate generation module
│   │   └── retrieval.py
│   ├── features/           # Feature engineering module
│   │   └── feature_engineering.py
│   ├── ranking/            # CTR prediction models
│   │   └── models.py
│   ├── auction/            # Auction mechanisms
│   │   └── auction.py
│   └── display/            # Visualization module
│       └── display.py
├── main.py                 # Main pipeline script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AdRankingSystem
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the **Criteo Display Advertising Challenge** dataset:
- 13 numerical features (I1-I13)
- 26 categorical features (C1-C26)
- Binary label (click/no-click)

### Download Options:

**Option 1: Using kagglehub (Recommended)**
```python
import kagglehub as kh
path = kh.dataset_download("mrkmakr/criteo-dataset")
```

**Option 2: Manual download**
Download from Kaggle: https://www.kaggle.com/datasets/mrkmakr/criteo-dataset
Place the dataset in the `data/` directory.

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  sample_ratio: 0.1      # Use 10% of data for faster training

ranking:
  model_type: "deepfm"   # Options: lr, fm, deepfm, xgboost, lightgbm
  embedding_dim: 8
  hidden_dims: [256, 128, 64]
  learning_rate: 0.001
  batch_size: 2048
  epochs: 5

auction:
  mechanism: "gsp"       # Options: gsp, vcg, first_price
  reserve_price: 0.01
  quality_weight: 0.5

display:
  top_n: 10             # Number of ads to display
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

### Step-by-Step Usage

```python
from main import AdRankingPipeline

# Initialize pipeline
pipeline = AdRankingPipeline('config/config.yaml')

# Run full pipeline
pipeline.run_full_pipeline(
    data_path='data/train.txt',
    download=False  # Set to True to auto-download from Kaggle
)
```

### Custom Pipeline

```python
# Step 1: Load and preprocess data
pipeline.load_and_preprocess_data('data/train.txt')

# Step 2: Build retrieval index
pipeline.build_retrieval_index()

# Step 3: Feature engineering
pipeline.engineer_features()

# Step 4: Train ranking model
pipeline.train_ranking_model()

# Step 5: Evaluate model
metrics = pipeline.evaluate_model()

# Step 6: Run auction and display results
auction_result, quality_scores = pipeline.run_auction_and_display()
```

## Components

### 1. Data Preprocessing
- Handles missing values
- Log transformation for numerical features (long-tail distribution)
- Label encoding for categorical features
- Train/validation/test split

### 2. Retrieval
- Embedding-based similarity search
- Fast candidate generation using random projection
- Optional: Two-tower model for learned embeddings

### 3. Feature Engineering
- **Cross Features**: Interaction between categorical features
- **Statistical Features**: Sum, mean, std, max, min of numerical features
- **Frequency Encoding**: Frequency of categorical values

### 4. Ranking Models

**DeepFM** (Default)
- Combines Factorization Machines and Deep Neural Networks
- Captures both low and high-order feature interactions

**Wide & Deep**
- Memorization (wide) + Generalization (deep)

**XGBoost / LightGBM**
- Gradient boosting decision trees
- Fast training and high accuracy

**Logistic Regression**
- Simple baseline model

### 5. Auction Mechanisms

**GSP (Generalized Second Price)**
- Used by Google AdWords
- Each advertiser pays the minimum bid to maintain position

**VCG (Vickrey-Clarke-Groves)**
- Truthful mechanism
- Each advertiser pays their externality

**First Price**
- Each advertiser pays their own bid
- Simpler but not truthful

### 6. Display & Visualization

Generates:
- Auction results table with top N ads
- CTR distribution and calibration plots
- ROC curve and AUC metrics
- Precision-Recall curve
- Feature importance analysis
- Auction analysis (bid vs price, revenue by position)

## Output

The pipeline generates:

**Models** (saved to `models/`):
- Trained ranking model
- Preprocessor artifacts (label encoders, feature dict)
- Retrieval index
- Feature engineering artifacts

**Results** (saved to `results/`):
- `ctr_distribution.png` - CTR distribution and calibration
- `roc_curve.png` - ROC curve with AUC score
- `pr_curve.png` - Precision-Recall curve
- `auction_analysis.png` - Auction metrics visualization

**Console Output**:
- Auction results table
- Summary statistics (revenue, quality scores)
- Model evaluation metrics (AUC, Log Loss)

## Example Output

```
================================================================================
AUCTION RESULTS - TOP 10 ADS
================================================================================
Rank  Ad ID    Bid    Price  Quality Score  Rank Score
   1   5432   4.52    3.21          0.452       2.042
   2   1234   3.87    2.15          0.556       2.152
   3   8765   3.21    1.98          0.421       1.351
   ...

--------------------------------------------------------------------------------
SUMMARY STATISTICS
--------------------------------------------------------------------------------
Total Ads in Auction: 1000
Total Revenue: $1,234.56
Average Price: $1.23
Average Quality Score: 0.1234
Average Bid: $2.34
================================================================================

================================================================================
MODEL EVALUATION METRICS
================================================================================
AUC (Area Under ROC Curve): 0.7856
Log Loss: 0.4532
================================================================================
```

## Performance

On a sample of 10% Criteo data:
- **Training time**: ~5-10 minutes (DeepFM on CPU)
- **Inference**: ~100ms for 1000 ads
- **AUC**: 0.75-0.80 (depending on model)

## Technical Details

### CTR Prediction
- Uses click-through rate (CTR) as quality score
- DeepFM architecture with embedding layers
- Binary cross-entropy loss

### Ranking Score
```
rank_score = bid × quality_score^w
```
where `w` is the quality weight (default: 0.5)

### Auction Pricing (GSP)
```
price[i] = rank_score[i+1] / quality[i]^w
```

## Future Enhancements

- [ ] Add multi-task learning (CTR + CVR)
- [ ] Implement attention mechanisms
- [ ] Add user behavior sequence modeling
- [ ] Support distributed training
- [ ] Add A/B testing framework
- [ ] Implement real-time serving API

## References

- Criteo Dataset: https://www.kaggle.com/datasets/mrkmakr/criteo-dataset
- DeepFM Paper: https://arxiv.org/abs/1703.04247
- GSP Auction: Edelman, B., Ostrovsky, M., & Schwarz, M. (2007)
- Wide & Deep: https://arxiv.org/abs/1606.07792

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.