"""
Example usage scripts for the Advertisement Ranking System
"""
import yaml
from main import AdRankingPipeline


def example_1_quick_demo():
    """Example 1: Quick demo with small sample"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: QUICK DEMO (Small Sample)")
    print("=" * 80 + "\n")

    # Modify config for quick demo
    config = {
        'data': {
            'sample_ratio': 0.01,  # Use only 1% of data
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        },
        'preprocessing': {
            'num_dense_features': 13,
            'num_sparse_features': 26,
            'min_frequency': 10
        },
        'retrieval': {
            'top_k_candidates': 50,
            'embedding_dim': 16
        },
        'features': {
            'cross_features': True,
            'statistical_features': True
        },
        'ranking': {
            'model_type': 'deepfm',
            'embedding_dim': 8,
            'hidden_dims': [128, 64],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 1024,
            'epochs': 2  # Reduced epochs for quick demo
        },
        'auction': {
            'mechanism': 'gsp',
            'reserve_price': 0.01,
            'quality_weight': 0.5
        },
        'display': {
            'top_n': 10,
            'show_metrics': True
        }
    }

    # Save temporary config
    with open('config/config_demo.yaml', 'w') as f:
        yaml.dump(config, f)

    # Run pipeline
    pipeline = AdRankingPipeline('config/config_demo.yaml')
    # Note: Replace with actual data path or set download=True
    # pipeline.run_full_pipeline(data_path='data/train.txt', download=False)

    print("To run this example, provide a valid data path or set download=True")


def example_2_compare_models():
    """Example 2: Compare different ranking models"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: COMPARING DIFFERENT RANKING MODELS")
    print("=" * 80 + "\n")

    models = ['deepfm', 'xgboost', 'lightgbm']
    results = {}

    for model_type in models:
        print(f"\n--- Training {model_type.upper()} ---")

        # Modify config for this model
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        config['ranking']['model_type'] = model_type
        config['ranking']['epochs'] = 3

        # Save temporary config
        with open('config/config_temp.yaml', 'w') as f:
            yaml.dump(config, f)

        # Run pipeline
        pipeline = AdRankingPipeline('config/config_temp.yaml')
        # Note: Add actual pipeline execution here
        # pipeline.run_full_pipeline(data_path='data/train.txt')

        # Store results
        # results[model_type] = metrics

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    for model, metrics in results.items():
        print(f"{model}: {metrics}")


def example_3_compare_auctions():
    """Example 3: Compare different auction mechanisms"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: COMPARING AUCTION MECHANISMS")
    print("=" * 80 + "\n")

    mechanisms = ['gsp', 'vcg', 'first_price']
    results = {}

    for mechanism in mechanisms:
        print(f"\n--- Testing {mechanism.upper()} Auction ---")

        # Modify config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        config['auction']['mechanism'] = mechanism

        with open('config/config_temp.yaml', 'w') as f:
            yaml.dump(config, f)

        pipeline = AdRankingPipeline('config/config_temp.yaml')
        # Note: Add actual pipeline execution here

    print("\n" + "=" * 80)
    print("AUCTION MECHANISM COMPARISON")
    print("=" * 80)


def example_4_custom_workflow():
    """Example 4: Custom workflow with manual steps"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: CUSTOM WORKFLOW")
    print("=" * 80 + "\n")

    pipeline = AdRankingPipeline('config/config.yaml')

    # Step 1: Load and preprocess data
    print("Step 1: Loading data...")
    # pipeline.load_and_preprocess_data('data/train.txt')

    # Step 2: Build retrieval index
    print("Step 2: Building retrieval index...")
    # pipeline.build_retrieval_index()

    # Step 3: Feature engineering
    print("Step 3: Engineering features...")
    # pipeline.engineer_features()

    # Step 4: Train model
    print("Step 4: Training model...")
    # pipeline.train_ranking_model()

    # Step 5: Evaluate
    print("Step 5: Evaluating model...")
    # metrics = pipeline.evaluate_model()

    # Step 6: Run auction
    print("Step 6: Running auction...")
    # auction_result, quality_scores = pipeline.run_auction_and_display(test_sample_size=500)

    print("\nCustom workflow demonstration complete!")


def example_5_inference_only():
    """Example 5: Load trained model and run inference only"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: INFERENCE WITH PRETRAINED MODEL")
    print("=" * 80 + "\n")

    pipeline = AdRankingPipeline('config/config.yaml')

    # Load preprocessor
    print("Loading preprocessor...")
    pipeline.preprocessor.load_preprocessor('models')

    # Load feature engine
    print("Loading feature engine...")
    pipeline.feature_engine.load('models')

    # Load model
    print("Loading ranking model...")
    # pipeline.ranking_model.load('models')

    # Load test data (assume already preprocessed)
    print("Loading test data...")
    # test_df = pd.read_csv('data/test_processed.csv')

    # Run inference
    print("Running inference...")
    # predictions = pipeline.predict_ctr(test_df)

    # Run auction with predictions
    print("Running auction...")
    # auction_result = pipeline.run_auction_and_display()

    print("\nInference complete!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ADVERTISEMENT RANKING SYSTEM - EXAMPLES")
    print("=" * 80)

    print("\nAvailable examples:")
    print("1. Quick demo with small sample")
    print("2. Compare different ranking models")
    print("3. Compare auction mechanisms")
    print("4. Custom workflow with manual steps")
    print("5. Inference with pretrained model")

    print("\nTo run an example, uncomment the function call below:\n")

    # Uncomment to run examples:
    # example_1_quick_demo()
    # example_2_compare_models()
    # example_3_compare_auctions()
    # example_4_custom_workflow()
    # example_5_inference_only()

    print("\nNote: Make sure to provide a valid data path or set download=True")
    print("before running the examples.")
