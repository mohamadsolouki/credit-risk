"""
Setup Script for Credit Risk Assessment Platform

This script preprocesses data and trains all models, saving them for deployment.
Run this script locally before pushing to GitHub for Streamlit Community Cloud deployment.

Usage:
    python setup_models.py
    
This will:
1. Load and preprocess the credit risk dataset
2. Train all available ML models
3. Save preprocessed data and trained models to the models/ directory
4. These files will be loaded automatically when the app starts on Streamlit Cloud
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor, save_preprocessed_data
from src.models import CreditRiskModels
from config.settings import MODELS_DIR

def main():
    """Main function to setup models for deployment."""
    print("=" * 60)
    print("Credit Risk Assessment Platform - Setup Script")
    print("=" * 60)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n[1/4] Loading data...")
    loader = DataLoader()
    df = loader.load_data()
    print(f"      Loaded {len(df):,} records with {len(df.columns)} features")
    
    # Step 2: Preprocess data
    print("\n[2/4] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Handle outliers
    df_processed = preprocessor.handle_outliers(df, method="clip")
    
    # Impute missing values
    df_processed = preprocessor.impute_missing_values(df_processed)
    
    # Fit and transform
    X, y, feature_names = preprocessor.fit_transform(df_processed)
    print(f"      Transformed data shape: {X.shape}")
    print(f"      Features: {len(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    print(f"      Training samples: {len(X_train):,}")
    print(f"      Test samples: {len(X_test):,}")
    
    # Handle class imbalance with SMOTE
    X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(
        X_train, y_train, method="smote"
    )
    print(f"      Training samples after SMOTE: {len(X_train_balanced):,}")
    
    # Save preprocessed data
    data_path = save_preprocessed_data(
        X_train_balanced, X_test, y_train_balanced, y_test, feature_names
    )
    print(f"      Saved preprocessed data to: {data_path}")
    
    # Step 3: Train all models
    print("\n[3/4] Training models...")
    credit_models = CreditRiskModels(feature_names=feature_names)
    
    all_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "AdaBoost"
    ]
    
    for i, model_name in enumerate(all_models, 1):
        print(f"      [{i}/{len(all_models)}] Training {model_name}...", end=" ")
        result = credit_models.train_model(
            model_name,
            X_train_balanced,
            y_train_balanced,
            X_test,
            y_test
        )
        print(f"ROC-AUC: {result.roc_auc:.4f}")
    
    # Step 4: Save all model results
    print("\n[4/4] Saving models...")
    models_path = credit_models.save_all_results()
    print(f"      Saved all models to: {models_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    comparison_df = credit_models.get_comparison_dataframe()
    print("\nModel Performance Summary:")
    print("-" * 60)
    for _, row in comparison_df.iterrows():
        print(f"  {row['Rank']}. {row['Model']}: ROC-AUC = {row['ROC-AUC']:.4f}")
    
    print("\n" + "-" * 60)
    print(f"Best Model: {comparison_df.iloc[0]['Model']}")
    print(f"Best ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
    
    print("\nSaved files:")
    print(f"  - {data_path}")
    print(f"  - {models_path}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  1. Commit these changes to git")
    print("  2. Push to GitHub")
    print("  3. Deploy to Streamlit Community Cloud")
    print("=" * 60)


if __name__ == "__main__":
    main()
