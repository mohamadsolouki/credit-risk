"""
Configuration settings for the Credit Risk Assessment Application.
This module centralizes all configuration parameters for easy maintenance.
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATIONS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA CONFIGURATIONS
# ============================================================================
DATASET_FILENAME = "credit_risk_dataset.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Target variable
TARGET_COLUMN = "loan_status"

# Feature categories
NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    "person_age": {
        "description": "Age of the loan applicant in years",
        "type": "Numerical",
        "source": "Customer application form / KYC documents",
        "business_relevance": "Indicates financial maturity and life stage"
    },
    "person_income": {
        "description": "Annual income of the applicant in USD",
        "type": "Numerical",
        "source": "Salary slips, tax returns, bank statements",
        "business_relevance": "Key indicator of repayment capacity"
    },
    "person_home_ownership": {
        "description": "Home ownership status (RENT, OWN, MORTGAGE, OTHER)",
        "type": "Categorical",
        "source": "Customer application form / Address verification",
        "business_relevance": "Indicates financial stability and asset ownership"
    },
    "person_emp_length": {
        "description": "Employment length in years",
        "type": "Numerical",
        "source": "Employment verification / HR documents",
        "business_relevance": "Job stability indicator affecting repayment reliability"
    },
    "loan_intent": {
        "description": "Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, DEBTCONSOLIDATION, HOMEIMPROVEMENT)",
        "type": "Categorical",
        "source": "Loan application form",
        "business_relevance": "Risk varies by loan purpose; venture loans typically higher risk"
    },
    "loan_grade": {
        "description": "Risk grade assigned to the loan (A-G, A being lowest risk)",
        "type": "Categorical (Ordinal)",
        "source": "Internal credit scoring system",
        "business_relevance": "Pre-calculated risk tier based on initial assessment"
    },
    "loan_amnt": {
        "description": "Requested loan amount in USD",
        "type": "Numerical",
        "source": "Loan application form",
        "business_relevance": "Higher amounts increase exposure and default impact"
    },
    "loan_int_rate": {
        "description": "Interest rate on the loan (%)",
        "type": "Numerical",
        "source": "Loan pricing system based on risk assessment",
        "business_relevance": "Higher rates often assigned to higher risk borrowers"
    },
    "loan_status": {
        "description": "Loan default status (0 = No Default, 1 = Default)",
        "type": "Binary Target",
        "source": "Loan performance tracking system",
        "business_relevance": "Target variable - indicates whether borrower defaulted"
    },
    "loan_percent_income": {
        "description": "Loan amount as percentage of annual income",
        "type": "Numerical",
        "source": "Calculated: loan_amnt / person_income",
        "business_relevance": "Debt-to-income ratio; higher values indicate higher risk"
    },
    "cb_person_default_on_file": {
        "description": "Historical default on credit bureau file (Y/N)",
        "type": "Categorical (Binary)",
        "source": "Credit Bureau report",
        "business_relevance": "Strong predictor - past behavior predicts future behavior"
    },
    "cb_person_cred_hist_length": {
        "description": "Length of credit history in years",
        "type": "Numerical",
        "source": "Credit Bureau report",
        "business_relevance": "Longer history provides more data for risk assessment"
    }
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model hyperparameter grids for tuning
MODEL_CONFIGS = {
    "Logistic Regression": {
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
            "max_iter": [1000]
        }
    },
    "Random Forest": {
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "XGBoost": {
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }
    },
    "LightGBM": {
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, -1],
            "learning_rate": [0.01, 0.1],
            "num_leaves": [31, 50]
        }
    }
}

# ============================================================================
# VISUALIZATION CONFIGURATIONS
# ============================================================================
PLOT_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f"
}

PLOT_TEMPLATE = "plotly_white"

# ============================================================================
# APP CONFIGURATIONS
# ============================================================================
APP_TITLE = "Credit Risk Assessment Platform"
APP_ICON = "🏦"
PAGE_LAYOUT = "wide"
