"""
Data Preprocessing Module for Credit Risk Assessment Application.

This module provides comprehensive data preprocessing capabilities including:
- Missing value imputation
- Outlier detection and handling
- Feature encoding (categorical to numerical)
- Feature scaling and normalization
- Feature engineering
- Data splitting for model training

Preprocessing Pipeline Philosophy:
=================================
Credit risk data requires careful preprocessing to ensure:
1. Model stability and reliability
2. Fair and unbiased predictions
3. Interpretable feature transformations
4. Reproducibility in production

The preprocessing pipeline follows scikit-learn conventions,
enabling easy integration with ML pipelines and serialization.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for credit risk data.
    
    This class handles all preprocessing steps including:
    - Missing value imputation
    - Outlier detection
    - Feature encoding
    - Feature scaling
    - Class imbalance handling
    
    Attributes:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        preprocessor: Fitted sklearn ColumnTransformer
        is_fitted: Whether the preprocessor has been fitted
    """
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        handle_imbalance: bool = True,
        imbalance_strategy: str = "smote"
    ):
        """
        Initialize the DataPreprocessor.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            handle_imbalance: Whether to handle class imbalance
            imbalance_strategy: Strategy for handling imbalance ('smote', 'undersample', 'none')
        """
        self.numerical_features = numerical_features or NUMERICAL_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.handle_imbalance = handle_imbalance
        self.imbalance_strategy = imbalance_strategy
        
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted: bool = False
        self._feature_names: Optional[List[str]] = None
        self._preprocessing_log: List[Dict] = []
    
    def _log_step(self, step: str, details: Dict) -> None:
        """Log a preprocessing step for documentation."""
        self._preprocessing_log.append({
            "step": step,
            "details": details
        })
        logger.info(f"Preprocessing: {step}")
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value analysis
        """
        missing_analysis = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                missing_analysis.append({
                    "Feature": col,
                    "Missing Count": missing_count,
                    "Missing %": round(missing_pct, 2),
                    "Data Type": str(df[col].dtype),
                    "Imputation Strategy": self._suggest_imputation(col, df[col])
                })
        
        return pd.DataFrame(missing_analysis) if missing_analysis else pd.DataFrame()
    
    def _suggest_imputation(self, col_name: str, series: pd.Series) -> str:
        """Suggest imputation strategy based on feature characteristics."""
        if col_name in self.numerical_features:
            # Check for skewness
            if series.dropna().skew() > 1:
                return "Median (skewed distribution)"
            return "Mean (normal distribution)"
        else:
            return "Mode (most frequent value)"
    
    def analyze_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """
        Analyze outliers in numerical features.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outlier analysis
        """
        outlier_analysis = []
        
        for col in self.numerical_features:
            if col not in df.columns:
                continue
                
            values = df[col].dropna()
            
            if method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            else:  # z-score
                z_scores = np.abs((values - values.mean()) / values.std())
                outliers = (z_scores > 3).sum()
            
            outlier_pct = (outliers / len(values)) * 100
            
            outlier_analysis.append({
                "Feature": col,
                "Outlier Count": outliers,
                "Outlier %": round(outlier_pct, 2),
                "Min": round(values.min(), 2),
                "Max": round(values.max(), 2),
                "Mean": round(values.mean(), 2),
                "Recommendation": "Cap/Floor" if outlier_pct > 5 else "Keep"
            })
        
        return pd.DataFrame(outlier_analysis)
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = "clip",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical features.
        
        Args:
            df: Input DataFrame
            method: 'clip' (cap at bounds), 'remove', or 'none'
            threshold: IQR multiplier for outlier detection
            
        Returns:
            DataFrame with handled outliers
        """
        df = df.copy()
        
        if method == "none":
            return df
        
        for col in self.numerical_features:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            if method == "clip":
                df[col] = df[col].clip(lower_bound, upper_bound)
                self._log_step(f"Clipped outliers in {col}", {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                })
            elif method == "remove":
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df = df[mask]
        
        return df
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using appropriate strategies.
        
        Numerical features: Median imputation (robust to outliers)
        Categorical features: Mode imputation (most frequent value)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Impute numerical features with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self._log_step(f"Imputed {col} with median", {"value": median_val})
        
        # Impute categorical features with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                self._log_step(f"Imputed {col} with mode", {"value": mode_val})
        
        return df
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            method: 'onehot' or 'label'
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if method == "label":
            for col in self.categorical_features:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    self._log_step(f"Label encoded {col}", {
                        "classes": list(le.classes_)
                    })
        else:  # onehot
            # Will be handled in the pipeline
            pass
        
        return df
    
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build a sklearn ColumnTransformer for preprocessing.
        
        This creates a reproducible preprocessing pipeline that can be
        used in production for consistent data transformation.
        
        Returns:
            Fitted ColumnTransformer
        """
        # Numerical preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop'
        )
        
        self._log_step("Built preprocessing pipeline", {
            "numerical_steps": ["median_imputation", "standard_scaling"],
            "categorical_steps": ["mode_imputation", "onehot_encoding"]
        })
        
        return preprocessor
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (transformed features, target, feature names)
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Build and fit preprocessor
        self.preprocessor = self.build_preprocessor()
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Generate feature names
        self._feature_names = self._get_feature_names()
        
        self.is_fitted = True
        
        self._log_step("Fitted and transformed data", {
            "original_shape": df.shape,
            "transformed_shape": X_transformed.shape
        })
        
        return X_transformed, y, self._feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.preprocessor.transform(df)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        feature_names = []
        
        # Numerical features keep their names
        feature_names.extend(self.numerical_features)
        
        # Categorical features get one-hot encoded names
        if hasattr(self.preprocessor, 'named_transformers_'):
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            if hasattr(cat_encoder, 'get_feature_names_out'):
                cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_features)
        
        return feature_names
    
    def get_preprocessing_log(self) -> List[Dict]:
        """Return the preprocessing log for documentation."""
        return self._preprocessing_log
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = TEST_SIZE,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            stratify: Whether to stratify split by target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=RANDOM_STATE,
            stratify=stratify_param
        )
        
        self._log_step("Split data", {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "stratified": stratify
        })
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        method: str = "smote"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: 'smote', 'undersample', or 'none'
            
        Returns:
            Tuple of (resampled X, resampled y)
        """
        if method == "none":
            return X, y
        
        original_counts = pd.Series(y).value_counts().to_dict()
        
        if method == "smote":
            sampler = SMOTE(random_state=RANDOM_STATE)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        new_counts = pd.Series(y_resampled).value_counts().to_dict()
        
        self._log_step(f"Applied {method}", {
            "original_distribution": original_counts,
            "new_distribution": new_counts
        })
        
        return X_resampled, y_resampled


def get_preprocessing_guide() -> str:
    """
    Return a comprehensive guide on data preprocessing for credit risk.
    
    This provides documentation on preprocessing best practices.
    """
    guide = """
    # Data Preprocessing Guide for Credit Risk Analysis
    
    ## 1. Missing Value Treatment
    
    ### Strategy Selection:
    
    | Data Type | Condition | Strategy |
    |-----------|-----------|----------|
    | Numerical | Normally distributed | Mean imputation |
    | Numerical | Skewed distribution | Median imputation |
    | Numerical | High importance | Model-based imputation (KNN, RF) |
    | Categorical | Low cardinality | Mode imputation |
    | Categorical | High cardinality | Create 'Unknown' category |
    
    ### Best Practices:
    - Never impute target variable - remove or flag those rows
    - Document imputation statistics for production use
    - Consider creating 'is_missing' flags for important features
    - Impute AFTER train/test split to prevent data leakage
    
    ## 2. Outlier Detection and Treatment
    
    ### Detection Methods:
    
    1. **IQR Method** (for skewed distributions):
       - Lower bound: Q1 - 1.5 × IQR
       - Upper bound: Q3 + 1.5 × IQR
    
    2. **Z-Score Method** (for normal distributions):
       - Flag values with |z-score| > 3
    
    3. **Domain Knowledge**:
       - Age: 18-100 years (logical bounds)
       - Income: Check against industry standards
       - Loan amount: Check against policy limits
    
    ### Treatment Options:
    - **Winsorization**: Cap at percentile bounds
    - **Transformation**: Log, square root for skewed data
    - **Removal**: Only for clearly erroneous data
    - **Flagging**: Create indicator for model to learn
    
    ## 3. Feature Encoding
    
    ### Categorical Encoding:
    
    | Feature Type | Encoding Method | Reason |
    |--------------|-----------------|--------|
    | Binary (Y/N) | Binary (0/1) | Simple, interpretable |
    | Nominal (no order) | One-Hot | Prevents false ordinality |
    | Ordinal (A-G grades) | Ordinal/Target | Preserves order |
    | High cardinality | Target encoding | Reduces dimensionality |
    
    ### One-Hot Encoding Considerations:
    - Drop one category to avoid multicollinearity
    - Handle unknown categories in production
    - Consider combining rare categories
    
    ## 4. Feature Scaling
    
    ### Methods:
    
    1. **StandardScaler** (z-score normalization):
       - Use when: Features have different scales
       - Required for: Linear models, SVM, Neural Networks
       - Formula: (x - μ) / σ
    
    2. **MinMaxScaler**:
       - Use when: Bounded range needed (0-1)
       - Sensitive to outliers
    
    3. **RobustScaler**:
       - Use when: Data has outliers
       - Uses median and IQR
    
    ### When Scaling is NOT Needed:
    - Tree-based models (RF, XGBoost, LightGBM)
    - Models with built-in normalization
    
    ## 5. Class Imbalance Handling
    
    ### Credit Risk Reality:
    - Default rates typically 2-15%
    - Creates highly imbalanced datasets
    - Models may predict majority class only
    
    ### Strategies:
    
    1. **Resampling**:
       - SMOTE: Synthetic minority oversampling
       - Undersampling: Reduce majority class
       - Combined: SMOTE + Tomek links
    
    2. **Cost-Sensitive Learning**:
       - Assign higher penalty to minority class errors
       - Use `class_weight='balanced'` parameter
    
    3. **Threshold Tuning**:
       - Adjust classification threshold post-training
       - Optimize for business metrics (profit, F1)
    
    ## 6. Feature Engineering
    
    ### Derived Features for Credit Risk:
    
    | New Feature | Formula | Rationale |
    |-------------|---------|-----------|
    | Debt-to-Income | loan_amnt / income | Repayment capacity |
    | Age at First Credit | age - credit_history | Financial maturity |
    | Income per Employment Year | income / emp_length | Career growth |
    
    ## 7. Production Considerations
    
    ### Pipeline Serialization:
    - Save fitted preprocessors with joblib
    - Version control preprocessing parameters
    - Test pipeline with edge cases
    
    ### Monitoring:
    - Track feature distributions for drift
    - Alert on out-of-bound values
    - Log preprocessing decisions
    """
    
    return guide
