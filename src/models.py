"""
Machine Learning Models Module for Credit Risk Assessment.

This module provides a comprehensive suite of ML models for credit risk
prediction, including model training, evaluation, and comparison utilities.

Model Selection Philosophy:
==========================
Credit risk modeling requires a balance between:
1. Predictive accuracy (minimize default losses)
2. Interpretability (regulatory requirements, fair lending)
3. Operational simplicity (production deployment)

We implement and compare multiple model families:
- Linear models (interpretable baseline)
- Tree-based models (good accuracy, feature importance)
- Ensemble methods (best accuracy, complex)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)

# Advanced ensemble imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RANDOM_STATE, CV_FOLDS, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Data class to store model training results."""
    model_name: str
    model: Any
    train_time: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    cv_scores: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None


class CreditRiskModels:
    """
    Credit Risk Model Training and Evaluation Class.
    
    This class provides methods for training, evaluating, and comparing
    multiple ML models for credit risk assessment.
    
    Attributes:
        models: Dictionary of model instances
        results: Dictionary of ModelResult objects
        feature_names: List of feature names for interpretation
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the CreditRiskModels class.
        
        Args:
            feature_names: List of feature names for feature importance
        """
        self.feature_names = feature_names
        self.models = self._initialize_models()
        self.results: Dict[str, ModelResult] = {}
        self.best_model: Optional[str] = None
    
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all ML models with default configurations.
        
        Returns:
            Dictionary mapping model names to model instances.
        """
        models = {
            # Linear Models
            "Logistic Regression": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            # Tree-based Models
            "Decision Tree": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=10,
                class_weight='balanced'
            ),
            
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                max_depth=15,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                max_depth=5,
                learning_rate=0.1
            ),
            
            # Advanced Ensemble Models
            "XGBoost": XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            
            "LightGBM": LGBMClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                max_depth=5,
                learning_rate=0.1,
                class_weight='balanced',
                verbose=-1
            ),
            
            # Other Models
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            
            "Naive Bayes": GaussianNB(),
            
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                learning_rate=0.1
            )
        }
        
        return models
    
    def train_model(
        self, 
        model_name: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> ModelResult:
        """
        Train a single model and evaluate its performance.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            ModelResult object with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name}...")
        
        model = self.models[model_name]
        
        # Train with timing
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = self._get_probabilities(model, X_test)
        
        # Calculate metrics
        result = ModelResult(
            model_name=model_name,
            model=model,
            train_time=train_time,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
            confusion_matrix=confusion_matrix(y_test, y_pred),
            feature_importance=self._get_feature_importance(model),
            predictions=y_pred,
            probabilities=y_proba
        )
        
        self.results[model_name] = result
        logger.info(f"{model_name} trained in {train_time:.2f}s - ROC-AUC: {result.roc_auc:.4f}")
        
        return result
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, ModelResult]:
        """
        Train all (or selected) models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            models_to_train: Optional list of specific models to train
            
        Returns:
            Dictionary of ModelResult objects
        """
        models_list = models_to_train or list(self.models.keys())
        
        for model_name in models_list:
            try:
                self.train_model(model_name, X_train, y_train, X_test, y_test)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Determine best model based on ROC-AUC
        if self.results:
            self.best_model = max(
                self.results.keys(), 
                key=lambda x: self.results[x].roc_auc
            )
        
        return self.results
    
    def cross_validate_model(
        self, 
        model_name: str, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = CV_FOLDS
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation for a specific model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with CV scores for different metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        
        cv_results = {
            'accuracy': cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy'),
            'precision': cross_val_score(model, X, y, cv=cv_strategy, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=cv_strategy, scoring='recall'),
            'f1': cross_val_score(model, X, y, cv=cv_strategy, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=cv_strategy, scoring='roc_auc')
        }
        
        return cv_results
    
    def _get_probabilities(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities if available."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            return model.decision_function(X)
        return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        if self.feature_names is None:
            return None
        
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        
        if importance is not None:
            # Ensure we have matching lengths
            n_features = min(len(importance), len(self.feature_names))
            return dict(zip(self.feature_names[:n_features], importance[:n_features]))
        
        return None
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get a comparison DataFrame of all trained models.
        
        Returns:
            DataFrame with model performance comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': round(result.accuracy, 4),
                'Precision': round(result.precision, 4),
                'Recall': round(result.recall, 4),
                'F1 Score': round(result.f1, 4),
                'ROC-AUC': round(result.roc_auc, 4),
                'Train Time (s)': round(result.train_time, 2)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_roc_curve_data(self, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Get ROC curve data for all trained models.
        
        Args:
            y_test: True labels
            
        Returns:
            Dictionary with FPR, TPR, and thresholds for each model
        """
        roc_data = {}
        
        for name, result in self.results.items():
            if result.probabilities is not None:
                fpr, tpr, thresholds = roc_curve(y_test, result.probabilities)
                roc_data[name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': result.roc_auc
                }
        
        return roc_data
    
    def get_precision_recall_curve_data(self, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Get Precision-Recall curve data for all trained models.
        
        Args:
            y_test: True labels
            
        Returns:
            Dictionary with precision, recall, and thresholds for each model
        """
        pr_data = {}
        
        for name, result in self.results.items():
            if result.probabilities is not None:
                precision, recall, thresholds = precision_recall_curve(
                    y_test, result.probabilities
                )
                pr_data[name] = {
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds,
                    'avg_precision': average_precision_score(y_test, result.probabilities)
                }
        
        return pr_data
    
    def save_model(self, model_name: str, filepath: Optional[Path] = None) -> Path:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Optional custom filepath
            
        Returns:
            Path where model was saved
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.joblib"
        
        joblib.dump(self.results[model_name].model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: Path) -> Any:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model object
        """
        return joblib.load(filepath)


def get_model_explanations() -> Dict[str, str]:
    """
    Get detailed explanations for each model type.
    
    Returns:
        Dictionary mapping model names to explanations
    """
    explanations = {
        "Logistic Regression": """
        **Logistic Regression**
        
        A linear model that estimates the probability of default using a logistic function.
        
        **How it works:**
        - Fits a linear combination of features to log-odds of default
        - Uses sigmoid function to convert to probability
        - Decision boundary is linear in feature space
        
        **Strengths:**
        - Highly interpretable (coefficient = feature impact)
        - Fast training and prediction
        - Provides well-calibrated probabilities
        - Required by many regulatory frameworks
        
        **Weaknesses:**
        - Assumes linear relationship (may underfit)
        - Sensitive to outliers without regularization
        - Cannot capture feature interactions without engineering
        
        **Credit Risk Context:**
        - Widely used in banking due to interpretability
        - Often used as regulatory baseline model
        - Coefficients directly show risk factors
        """,
        
        "Decision Tree": """
        **Decision Tree**
        
        A tree-structured model that makes decisions through a series of feature-based splits.
        
        **How it works:**
        - Recursively splits data based on feature thresholds
        - Each leaf node represents a class prediction
        - Splits chosen to maximize information gain
        
        **Strengths:**
        - Highly interpretable (visual tree structure)
        - Handles non-linear relationships
        - No scaling required
        - Captures feature interactions naturally
        
        **Weaknesses:**
        - Prone to overfitting (high variance)
        - Unstable (small data changes → different tree)
        - Can create biased trees with imbalanced data
        
        **Credit Risk Context:**
        - Useful for rule extraction
        - Easy to explain to stakeholders
        - Often limited depth for regulatory compliance
        """,
        
        "Random Forest": """
        **Random Forest**
        
        An ensemble of decision trees that reduces overfitting through averaging.
        
        **How it works:**
        - Trains multiple trees on bootstrap samples
        - Each tree sees random subset of features
        - Final prediction is majority vote
        
        **Strengths:**
        - Excellent accuracy without extensive tuning
        - Robust to outliers and noise
        - Provides feature importance scores
        - Parallelizable for fast training
        
        **Weaknesses:**
        - Less interpretable than single tree
        - Can be slow for very large datasets
        - May not extrapolate well beyond training data
        
        **Credit Risk Context:**
        - Industry standard for credit scoring
        - Good balance of accuracy and interpretability
        - Feature importance helps with variable selection
        """,
        
        "Gradient Boosting": """
        **Gradient Boosting**
        
        An ensemble method that builds trees sequentially, correcting previous errors.
        
        **How it works:**
        - Trees are trained on residuals of previous predictions
        - Each tree adds a small correction
        - Learning rate controls contribution of each tree
        
        **Strengths:**
        - Often achieves highest accuracy
        - Handles different types of features well
        - Flexible loss functions
        
        **Weaknesses:**
        - Slower training (sequential, not parallel)
        - More prone to overfitting without tuning
        - Many hyperparameters to tune
        
        **Credit Risk Context:**
        - Used for maximum predictive power
        - Popular in Kaggle competitions
        - Requires careful cross-validation
        """,
        
        "XGBoost": """
        **XGBoost (Extreme Gradient Boosting)**
        
        An optimized, regularized implementation of gradient boosting.
        
        **How it works:**
        - Gradient boosting with L1/L2 regularization
        - Uses second-order gradients for optimization
        - Implements tree pruning and column subsampling
        
        **Strengths:**
        - State-of-the-art accuracy on structured data
        - Built-in regularization prevents overfitting
        - Handles missing values natively
        - Highly optimized for speed
        
        **Weaknesses:**
        - Complex with many hyperparameters
        - Harder to interpret than simpler models
        - May require GPU for very large datasets
        
        **Credit Risk Context:**
        - Top performer in many credit scoring competitions
        - Used by Fintech companies for real-time scoring
        - SHAP values available for explainability
        """,
        
        "LightGBM": """
        **LightGBM (Light Gradient Boosting Machine)**
        
        An efficient gradient boosting implementation using histogram-based algorithms.
        
        **How it works:**
        - Bins continuous features into histograms
        - Uses leaf-wise tree growth (vs. level-wise)
        - Gradient-based one-side sampling (GOSS)
        
        **Strengths:**
        - Fastest training among boosting methods
        - Lower memory usage
        - Handles large datasets efficiently
        - Native categorical feature support
        
        **Weaknesses:**
        - May overfit on small datasets
        - Leaf-wise growth can create unbalanced trees
        
        **Credit Risk Context:**
        - Preferred for real-time credit decisions
        - Scales well to millions of records
        - Good for production deployment
        """,
        
        "K-Nearest Neighbors": """
        **K-Nearest Neighbors (KNN)**
        
        A non-parametric method that classifies based on similar cases.
        
        **How it works:**
        - Stores all training examples
        - For new case, finds K most similar cases
        - Predicts majority class among neighbors
        
        **Strengths:**
        - Simple and intuitive
        - No training phase (lazy learner)
        - Naturally handles multi-class problems
        
        **Weaknesses:**
        - Slow prediction (searches all data)
        - Sensitive to feature scaling
        - Suffers from curse of dimensionality
        
        **Credit Risk Context:**
        - Useful for case-based reasoning
        - Can find similar historical defaults
        - Not commonly used in production
        """,
        
        "Naive Bayes": """
        **Naive Bayes**
        
        A probabilistic classifier based on Bayes' theorem with independence assumption.
        
        **How it works:**
        - Assumes features are conditionally independent given class
        - Calculates P(default|features) using Bayes' rule
        - Predicts class with highest posterior probability
        
        **Strengths:**
        - Very fast training and prediction
        - Works well with high-dimensional data
        - Requires little training data
        
        **Weaknesses:**
        - Independence assumption often violated
        - Poor probability calibration
        - Cannot learn feature interactions
        
        **Credit Risk Context:**
        - Quick baseline model
        - Useful when features are truly independent
        - Often outperformed by other methods
        """,
        
        "AdaBoost": """
        **AdaBoost (Adaptive Boosting)**
        
        An ensemble method that focuses on misclassified examples.
        
        **How it works:**
        - Trains weak learners sequentially
        - Increases weight on misclassified examples
        - Combines weak learners with weighted voting
        
        **Strengths:**
        - Simple to implement
        - Often improves weak classifiers significantly
        - Less prone to overfitting than other boosters
        
        **Weaknesses:**
        - Sensitive to noisy data and outliers
        - Can be slow with many weak learners
        
        **Credit Risk Context:**
        - Historical importance in ensemble methods
        - Often outperformed by gradient boosting
        - Useful for understanding boosting concepts
        """
    }
    
    return explanations


def get_evaluation_metrics_guide() -> str:
    """
    Return a comprehensive guide on model evaluation metrics for credit risk.
    """
    guide = """
    # Model Evaluation Metrics for Credit Risk
    
    ## 1. Classification Metrics
    
    ### Accuracy
    - **Formula**: (TP + TN) / (TP + TN + FP + FN)
    - **Interpretation**: Overall correctness of predictions
    - **Limitation**: Misleading with imbalanced data (e.g., 95% accuracy by predicting all "no default")
    
    ### Precision (Positive Predictive Value)
    - **Formula**: TP / (TP + FP)
    - **Interpretation**: Of predicted defaults, how many actually defaulted?
    - **Use case**: When false positives are costly (rejecting good customers)
    
    ### Recall (Sensitivity/True Positive Rate)
    - **Formula**: TP / (TP + FN)
    - **Interpretation**: Of actual defaults, how many did we catch?
    - **Use case**: When false negatives are costly (approving bad loans)
    
    ### F1 Score
    - **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
    - **Interpretation**: Harmonic mean of precision and recall
    - **Use case**: When you need balance between precision and recall
    
    ## 2. Ranking Metrics
    
    ### ROC-AUC (Area Under ROC Curve)
    - **Range**: 0.5 (random) to 1.0 (perfect)
    - **Interpretation**: Probability that model ranks a random positive higher than random negative
    - **Industry benchmark**: 0.7+ acceptable, 0.8+ good, 0.9+ excellent
    
    ### Gini Coefficient
    - **Formula**: 2 × AUC - 1
    - **Range**: 0 to 1
    - **Use case**: Traditional banking metric
    
    ### KS Statistic (Kolmogorov-Smirnov)
    - **Definition**: Maximum separation between cumulative distributions
    - **Use case**: Measures discriminatory power at optimal threshold
    
    ## 3. Business Metrics
    
    ### Expected Loss
    - **Formula**: PD × LGD × EAD
    - **Components**: Probability of Default, Loss Given Default, Exposure at Default
    
    ### Profit/Loss Matrix
    - True Positive (catch default): Avoid loss of $X
    - False Positive (reject good): Lose profit of $Y
    - False Negative (miss default): Lose $X
    - True Negative (approve good): Earn $Y
    
    ## 4. Metric Selection for Credit Risk
    
    | Model Purpose | Primary Metric | Secondary Metrics |
    |--------------|----------------|-------------------|
    | Approval/Rejection | Recall (catch defaults) | Precision |
    | Risk Ranking | ROC-AUC, Gini | KS Statistic |
    | Probability Estimation | Brier Score | Calibration curves |
    | Business Optimization | Expected Profit | Precision at threshold |
    """
    
    return guide
