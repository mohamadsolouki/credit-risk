"""
Data Loading Module for Credit Risk Assessment Application.

This module handles all data loading operations, provides data validation,
and offers utilities for data exploration and summary statistics.

Data Pipeline Architecture:
==========================
In a production environment, credit risk data typically flows through:

1. DATA SOURCES (Extraction)
   ├── Core Banking System (customer demographics, account info)
   ├── Loan Origination System (loan applications, amounts, terms)
   ├── Credit Bureau APIs (credit scores, history, defaults)
   ├── KYC/AML Systems (identity verification, risk flags)
   └── External Data Providers (market data, economic indicators)

2. DATA INGESTION LAYER
   ├── Real-time Streaming (Apache Kafka/AWS Kinesis)
   │   └── For immediate credit decisions
   └── Batch Processing (Apache Spark/AWS Glue)
       └── For historical analysis and model training

3. DATA WAREHOUSE (Storage & Transformation)
   ├── Raw Zone: Immutable source data
   ├── Staging Zone: Cleaned and validated data
   ├── Curated Zone: Business-ready datasets
   └── Analytics Zone: Aggregated metrics and features

4. FEATURE STORE
   ├── Historical Features (batch computed)
   └── Real-time Features (streaming computed)

5. MODEL SERVING
   ├── Batch Predictions (for portfolio analysis)
   └── Real-time Predictions (for credit decisions)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    DATASET_PATH, TARGET_COLUMN, NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES, FEATURE_DESCRIPTIONS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data Loader class for credit risk dataset.
    
    This class provides methods for loading, validating, and exploring
    the credit risk dataset with comprehensive documentation.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Optional path to the dataset. Defaults to configured path.
        """
        self.data_path = data_path or DATASET_PATH
        self._data: Optional[pd.DataFrame] = None
        self._load_timestamp: Optional[str] = None
    
    def load_data(self, validate: bool = True) -> pd.DataFrame:
        """
        Load the credit risk dataset from CSV file.
        
        This method loads data and optionally validates it against expected schema.
        
        Args:
            validate: Whether to perform schema validation.
            
        Returns:
            DataFrame containing the credit risk data.
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            ValueError: If validation fails.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        
        # Load the dataset
        self._data = pd.read_csv(self.data_path)
        self._load_timestamp = pd.Timestamp.now().isoformat()
        
        logger.info(f"Loaded {len(self._data)} records with {len(self._data.columns)} features")
        
        if validate:
            self._validate_schema()
        
        return self._data.copy()
    
    def _validate_schema(self) -> None:
        """Validate the dataset schema against expected configuration."""
        expected_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        missing_columns = set(expected_columns) - set(self._data.columns)
        
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset.
        
        Returns:
            Dictionary containing dataset statistics and metadata.
        """
        if self._data is None:
            self.load_data()
        
        df = self._data
        
        summary = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "numerical_features": len(NUMERICAL_FEATURES),
            "categorical_features": len(CATEGORICAL_FEATURES),
            "target_column": TARGET_COLUMN,
            "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "load_timestamp": self._load_timestamp
        }
        
        return summary
    
    def get_feature_info(self) -> pd.DataFrame:
        """
        Get detailed information about each feature.
        
        Returns:
            DataFrame with feature descriptions, types, and statistics.
        """
        if self._data is None:
            self.load_data()
        
        feature_info = []
        
        for col in self._data.columns:
            info = FEATURE_DESCRIPTIONS.get(col, {})
            
            row = {
                "Feature": col,
                "Description": info.get("description", "N/A"),
                "Type": info.get("type", self._data[col].dtype),
                "Data Source": info.get("source", "N/A"),
                "Business Relevance": info.get("business_relevance", "N/A"),
                "Missing %": (self._data[col].isnull().sum() / len(self._data) * 100),
                "Unique Values": self._data[col].nunique()
            }
            feature_info.append(row)
        
        return pd.DataFrame(feature_info)
    
    def get_numerical_stats(self) -> pd.DataFrame:
        """Get descriptive statistics for numerical features."""
        if self._data is None:
            self.load_data()
        
        return self._data[NUMERICAL_FEATURES].describe()
    
    def get_categorical_stats(self) -> Dict[str, pd.Series]:
        """Get value counts for categorical features."""
        if self._data is None:
            self.load_data()
        
        return {col: self._data[col].value_counts() for col in CATEGORICAL_FEATURES}
    
    def get_target_analysis(self) -> Dict[str, Any]:
        """
        Analyze the target variable distribution and class imbalance.
        
        Returns:
            Dictionary with target variable analysis.
        """
        if self._data is None:
            self.load_data()
        
        target = self._data[TARGET_COLUMN]
        value_counts = target.value_counts()
        
        analysis = {
            "distribution": value_counts.to_dict(),
            "distribution_percent": (value_counts / len(target) * 100).to_dict(),
            "imbalance_ratio": value_counts.min() / value_counts.max(),
            "is_imbalanced": value_counts.min() / value_counts.max() < 0.5,
            "total_samples": len(target),
            "class_labels": {0: "No Default (Good)", 1: "Default (Bad)"}
        }
        
        return analysis


def get_data_collection_guide() -> str:
    """
    Return a comprehensive guide on data collection for credit risk assessment.
    
    This provides documentation on how similar data can be gathered,
    processed, and loaded into a data warehouse.
    """
    guide = """
    # Credit Risk Data Collection and Processing Guide
    
    ## 1. Data Sources and Collection Methods
    
    ### Primary Data Sources:
    
    #### A. Customer Application Data
    - **Collection Point**: Online/offline loan application forms
    - **Data Elements**: Personal info, income, employment, loan purpose
    - **Validation**: KYC verification, document scanning, OCR
    - **Storage**: Application Management System
    
    #### B. Credit Bureau Data
    - **Collection Point**: API integration with credit bureaus (Equifax, Experian, TransUnion)
    - **Data Elements**: Credit score, credit history, defaults, inquiries
    - **Frequency**: Real-time pull during application, periodic batch refresh
    - **Storage**: Credit Data Repository
    
    #### C. Internal Banking Data
    - **Collection Point**: Core Banking System (CBS)
    - **Data Elements**: Account history, transaction patterns, existing relationships
    - **Frequency**: Daily batch or real-time streaming
    - **Storage**: Customer Data Platform (CDP)
    
    #### D. Alternative Data (Optional)
    - **Collection Point**: Third-party APIs with customer consent
    - **Data Elements**: Social media activity, utility payments, e-commerce behavior
    - **Frequency**: Periodic batch processing
    - **Storage**: Alternative Data Lake
    
    ## 2. Data Pipeline Architecture
    
    ### ETL/ELT Process:
    
    ```
    [Source Systems] → [Ingestion Layer] → [Raw Zone] → [Staging Zone] → [Curated Zone]
                              ↓                                               ↓
                        [Data Quality]                              [Feature Store]
                         [Validation]                                      ↓
                                                                   [ML Models]
    ```
    
    ### Technology Stack Options:
    
    | Layer | Open Source | Cloud (AWS) | Cloud (Azure) |
    |-------|-------------|-------------|---------------|
    | Ingestion | Apache Kafka | Kinesis | Event Hubs |
    | Processing | Spark | Glue/EMR | Databricks |
    | Storage | HDFS | S3 | ADLS |
    | Warehouse | PostgreSQL | Redshift | Synapse |
    | Orchestration | Airflow | Step Functions | Data Factory |
    
    ## 3. Data Warehouse Schema Design
    
    ### Star Schema for Credit Risk:
    
    #### Fact Table: FACT_LOAN_APPLICATIONS
    - application_id (PK)
    - customer_id (FK)
    - loan_amount
    - interest_rate
    - loan_status (Default/No Default)
    - application_date
    - decision_date
    
    #### Dimension Tables:
    - DIM_CUSTOMER (demographics, income)
    - DIM_CREDIT_PROFILE (credit bureau data)
    - DIM_LOAN_PRODUCT (loan types, terms)
    - DIM_DATE (calendar dimension)
    - DIM_GEOGRAPHY (location hierarchy)
    
    ## 4. Data Quality Framework
    
    ### Quality Dimensions:
    1. **Completeness**: % of non-null values
    2. **Accuracy**: Validation against source systems
    3. **Consistency**: Cross-field validation rules
    4. **Timeliness**: Data freshness metrics
    5. **Uniqueness**: Duplicate detection
    
    ### Quality Rules Examples:
    - Age must be between 18 and 100
    - Income must be positive
    - Loan amount cannot exceed 10x annual income
    - Credit history length cannot exceed age - 18
    
    ## 5. Compliance and Governance
    
    ### Regulatory Requirements:
    - **Fair Lending**: Ensure no discrimination in credit decisions
    - **GDPR/CCPA**: Data privacy and consent management
    - **Basel III**: Risk-weighted asset calculations
    - **Model Risk Management**: SR 11-7 guidelines
    
    ### Data Lineage:
    - Track data from source to consumption
    - Document transformations and business rules
    - Enable audit trails for regulatory review
    """
    
    return guide


# Convenience function for quick data loading
def load_credit_data() -> pd.DataFrame:
    """Quick function to load credit risk data."""
    loader = DataLoader()
    return loader.load_data()
