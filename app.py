"""
Credit Risk Assessment Platform - Main Streamlit Application

This is an advanced, fully-integrated Streamlit application for credit risk
assessment that provides comprehensive analysis, model comparison, and
educational content about the data analytics process.

Author: Credit Risk Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom modules
from src.data_loader import DataLoader, get_data_collection_guide
from src.preprocessing import (
    DataPreprocessor, get_preprocessing_guide,
    save_preprocessed_data, load_preprocessed_data, preprocessed_data_exists
)
from src.models import CreditRiskModels, get_model_explanations, get_evaluation_metrics_guide
from src.visualizations import CreditRiskVisualizer
from config.settings import (
    APP_TITLE, APP_ICON, PAGE_LAYOUT, TARGET_COLUMN,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, FEATURE_DESCRIPTIONS,
    MODELS_DIR
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize session state variables and load saved data if available."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    
    # Try to load saved preprocessed data
    if st.session_state.X_train is None and preprocessed_data_exists():
        saved_data = load_preprocessed_data()
        if saved_data is not None:
            st.session_state.X_train = saved_data['X_train']
            st.session_state.X_test = saved_data['X_test']
            st.session_state.y_train = saved_data['y_train']
            st.session_state.y_test = saved_data['y_test']
            st.session_state.feature_names = saved_data['feature_names']
    
    # Try to load saved model results
    if st.session_state.model_results is None and CreditRiskModels.results_exist():
        credit_models = CreditRiskModels()
        if credit_models.load_all_results():
            st.session_state.model_results = credit_models
            st.session_state.models_trained = True
            if st.session_state.feature_names is None:
                st.session_state.feature_names = credit_models.feature_names


initialize_session_state()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            [
                "🏠 Home",
                "📊 Data Overview",
                "🔧 Preprocessing",
                "🤖 Model Training",
                "📈 Model Comparison",
                "💡 Feature Importance",
                "📚 Data Pipeline Guide",
                "📖 Analytics Process"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Data loading section
        st.subheader("Data Status")
        if st.session_state.data_loaded:
            st.success(f"✅ Data loaded: {len(st.session_state.df):,} records")
        else:
            st.warning("⚠️ Data not loaded")
        
        if st.session_state.models_trained:
            st.success("✅ Models trained")
        else:
            st.info("ℹ️ Models not trained")
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("🔄 Reload Data", use_container_width=True):
            load_data()
        
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        return page


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data
def load_data():
    """Load and cache the credit risk dataset."""
    loader = DataLoader()
    df = loader.load_data()
    return df


def ensure_data_loaded():
    """Ensure data is loaded before proceeding."""
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    return st.session_state.df


# ============================================================================
# PAGE RENDERERS
# ============================================================================
def render_home_page():
    """Render the home page."""
    st.markdown('<p class="main-header">🏦 Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning for Credit Risk Analysis</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Data Analysis
        - Comprehensive EDA
        - Feature distributions
        - Correlation analysis
        - Missing value analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 ML Models
        - 9+ algorithms compared
        - Automated training
        - Performance metrics
        - Feature importance
        """)
    
    with col3:
        st.markdown("""
        ### 📚 Documentation
        - Data pipeline guide
        - Preprocessing steps
        - Model explanations
        - Best practices
        """)
    
    st.divider()
    
    # Dataset overview
    st.subheader("📁 Dataset Overview")
    
    df = ensure_data_loaded()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        default_rate = (df[TARGET_COLUMN].sum() / len(df) * 100)
        st.metric("Default Rate", f"{default_rate:.1f}%")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing Data", f"{missing_pct:.2f}%")
    
    st.divider()
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    <div class="info-box">
    <strong>How to use this application:</strong>
    <ol>
        <li><strong>Data Overview</strong>: Explore the dataset, understand features, and analyze distributions</li>
        <li><strong>Preprocessing</strong>: Review data cleaning steps, handle missing values, and prepare features</li>
        <li><strong>Model Training</strong>: Train multiple ML models with one click</li>
        <li><strong>Model Comparison</strong>: Compare model performance using various metrics</li>
        <li><strong>Feature Importance</strong>: Understand which features drive predictions</li>
        <li><strong>Guides</strong>: Learn about data pipelines and analytics processes</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # About the dataset
    st.subheader("📋 About the Dataset")
    
    st.markdown("""
    This credit risk dataset contains information about loan applicants and their loan outcomes.
    The goal is to predict whether a borrower will **default** (fail to repay) on their loan.
    
    **Key Features:**
    - **Applicant Information**: Age, income, employment length, home ownership
    - **Loan Details**: Amount, interest rate, purpose/intent
    - **Credit History**: Credit bureau data, previous defaults, credit history length
    - **Target Variable**: Loan status (0 = No Default, 1 = Default)
    """)


def render_data_overview_page():
    """Render the data overview page."""
    st.header("📊 Data Overview & Exploration")
    
    df = ensure_data_loaded()
    viz = CreditRiskVisualizer()
    
    tabs = st.tabs([
        "📋 Data Sample", 
        "📈 Feature Details",
        "🎯 Target Analysis",
        "📊 Distributions",
        "🔗 Correlations"
    ])
    
    # Tab 1: Data Sample
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
        
        with col2:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Type': df.dtypes.astype(str).value_counts().index,
                'Count': df.dtypes.astype(str).value_counts().values
            })
            st.dataframe(dtype_df, use_container_width=True)
    
    # Tab 2: Feature Details
    with tabs[1]:
        st.subheader("Feature Documentation")
        
        feature_data = []
        for col in df.columns:
            info = FEATURE_DESCRIPTIONS.get(col, {})
            feature_data.append({
                "Feature": col,
                "Description": info.get("description", "N/A"),
                "Type": info.get("type", str(df[col].dtype)),
                "Data Source": info.get("source", "N/A"),
                "Business Relevance": info.get("business_relevance", "N/A")
            })
        
        feature_df = pd.DataFrame(feature_data)
        st.dataframe(feature_df, use_container_width=True, height=500)
        
        # Numerical statistics
        st.subheader("Numerical Feature Statistics")
        st.dataframe(df[NUMERICAL_FEATURES].describe(), use_container_width=True)
    
    # Tab 3: Target Analysis
    with tabs[2]:
        st.subheader("Target Variable Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = viz.plot_target_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            target_stats = df[TARGET_COLUMN].value_counts()
            st.markdown("""
            ### Class Distribution
            
            | Class | Count | Percentage |
            |-------|-------|------------|
            | No Default (0) | {:,} | {:.1f}% |
            | Default (1) | {:,} | {:.1f}% |
            """.format(
                target_stats.get(0, 0),
                target_stats.get(0, 0) / len(df) * 100,
                target_stats.get(1, 0),
                target_stats.get(1, 0) / len(df) * 100
            ))
            
            imbalance_ratio = target_stats.min() / target_stats.max()
            if imbalance_ratio < 0.5:
                st.warning(f"""
                ⚠️ **Class Imbalance Detected**
                
                The minority class ratio is {imbalance_ratio:.2f}. 
                We will apply techniques like SMOTE during preprocessing 
                to address this imbalance.
                """)
    
    # Tab 4: Distributions
    with tabs[3]:
        st.subheader("Feature Distributions")
        
        sub_tabs = st.tabs(["Numerical Features", "Categorical Features", "Outlier Analysis"])
        
        with sub_tabs[0]:
            fig = viz.plot_numerical_distributions(df, NUMERICAL_FEATURES)
            st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[1]:
            fig = viz.plot_categorical_distributions(df, CATEGORICAL_FEATURES)
            st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[2]:
            fig = viz.plot_outlier_analysis(df, NUMERICAL_FEATURES)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Correlations
    with tabs[4]:
        st.subheader("Feature Correlations")
        
        fig = viz.plot_correlation_matrix(df, NUMERICAL_FEATURES + [TARGET_COLUMN])
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values
        st.subheader("Missing Values Analysis")
        fig = viz.plot_missing_values(df)
        st.plotly_chart(fig, use_container_width=True)


def render_preprocessing_page():
    """Render the preprocessing page."""
    st.header("🔧 Data Preprocessing")
    
    df = ensure_data_loaded()
    
    tabs = st.tabs([
        "📋 Overview",
        "❓ Missing Values",
        "📊 Outliers",
        "⚙️ Apply Preprocessing",
        "📖 Preprocessing Guide"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Preprocessing Pipeline Overview")
        
        st.markdown("""
        <div class="info-box">
        <strong>Our preprocessing pipeline includes:</strong>
        <ol>
            <li><strong>Missing Value Imputation</strong>: Median for numerical, mode for categorical</li>
            <li><strong>Outlier Handling</strong>: Optional clipping using IQR method</li>
            <li><strong>Feature Encoding</strong>: One-hot encoding for categorical variables</li>
            <li><strong>Feature Scaling</strong>: StandardScaler (z-score normalization)</li>
            <li><strong>Class Imbalance</strong>: SMOTE oversampling (optional)</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        preprocessor = DataPreprocessor()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numerical Features")
            st.write(", ".join(NUMERICAL_FEATURES))
        
        with col2:
            st.subheader("Categorical Features")
            st.write(", ".join(CATEGORICAL_FEATURES))
    
    # Tab 2: Missing Values
    with tabs[1]:
        st.subheader("Missing Value Analysis")
        
        preprocessor = DataPreprocessor()
        missing_df = preprocessor.analyze_missing_values(df)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <strong>Missing Value Treatment Strategy:</strong>
            <ul>
                <li>Numerical features: Impute with <strong>median</strong> (robust to outliers)</li>
                <li>Categorical features: Impute with <strong>mode</strong> (most frequent value)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No missing values detected in the dataset!")
    
    # Tab 3: Outliers
    with tabs[2]:
        st.subheader("Outlier Analysis")
        
        preprocessor = DataPreprocessor()
        outlier_df = preprocessor.analyze_outliers(df)
        
        st.dataframe(outlier_df, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Outlier Detection Method: IQR (Interquartile Range)</strong>
        <ul>
            <li>Lower Bound = Q1 - 1.5 × IQR</li>
            <li>Upper Bound = Q3 + 1.5 × IQR</li>
            <li>Values outside these bounds are flagged as outliers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab 4: Apply Preprocessing
    with tabs[3]:
        st.subheader("Apply Preprocessing to Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_outliers = st.selectbox(
                "Outlier Treatment",
                ["clip", "none"],
                help="'clip' will cap outliers at IQR bounds"
            )
        
        with col2:
            handle_imbalance = st.selectbox(
                "Class Imbalance Treatment",
                ["smote", "undersample", "none"],
                help="SMOTE creates synthetic minority samples"
            )
        
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
        if st.button("🚀 Apply Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                # Initialize preprocessor
                preprocessor = DataPreprocessor()
                
                # Handle outliers
                df_processed = preprocessor.handle_outliers(df, method=handle_outliers)
                
                # Impute missing values
                df_processed = preprocessor.impute_missing_values(df_processed)
                
                # Fit and transform
                X, y, feature_names = preprocessor.fit_transform(df_processed)
                
                # Split data
                X_train, X_test, y_train, y_test = preprocessor.split_data(
                    X, y, test_size=test_size
                )
                
                # Handle class imbalance on training data only
                X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(
                    X_train, y_train, method=handle_imbalance
                )
                
                # Store in session state
                st.session_state.preprocessor = preprocessor
                st.session_state.X_train = X_train_balanced
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train_balanced
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                
                # Save preprocessed data for deployment
                save_path = save_preprocessed_data(
                    X_train_balanced, X_test, y_train_balanced, y_test, feature_names
                )
                
                st.success("✅ Preprocessing completed and saved!")
                
                # Show preprocessing summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", f"{len(X_train_balanced):,}")
                with col2:
                    st.metric("Test Samples", f"{len(X_test):,}")
                with col3:
                    st.metric("Features", len(feature_names))
                
                st.info(f"💾 Data saved to: `{save_path.name}`")
                
                # Show preprocessing log
                st.subheader("Preprocessing Log")
                log = preprocessor.get_preprocessing_log()
                for entry in log:
                    st.write(f"✓ **{entry['step']}**: {entry['details']}")
    
    # Tab 5: Guide
    with tabs[4]:
        st.subheader("Data Preprocessing Guide")
        st.markdown(get_preprocessing_guide())


def render_model_training_page():
    """Render the model training page."""
    st.header("🤖 Model Training")
    
    # Check if data is preprocessed
    if st.session_state.X_train is None:
        st.warning("⚠️ Please preprocess the data first (go to Preprocessing page)")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>Available Models:</strong>
    <ul>
        <li><strong>Linear Models</strong>: Logistic Regression</li>
        <li><strong>Tree-based</strong>: Decision Tree, Random Forest, Gradient Boosting</li>
        <li><strong>Advanced Ensemble</strong>: XGBoost, LightGBM, AdaBoost</li>
        <li><strong>Other</strong>: K-Nearest Neighbors, Naive Bayes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
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
    
    selected_models = st.multiselect(
        "Select Models to Train",
        all_models,
        default=all_models,
        help="Select one or more models to train and compare"
    )
    
    if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model")
            return
        
        # Initialize models
        credit_models = CreditRiskModels(feature_names=st.session_state.feature_names)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train each model
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            credit_models.train_model(
                model_name,
                st.session_state.X_train,
                st.session_state.y_train,
                st.session_state.X_test,
                st.session_state.y_test
            )
            progress_bar.progress((i + 1) / len(selected_models))
        
        status_text.text("Training completed!")
        progress_bar.progress(1.0)
        
        # Store results
        st.session_state.model_results = credit_models
        st.session_state.models_trained = True
        
        # Save models for deployment
        save_path = credit_models.save_all_results()
        
        st.success(f"✅ Successfully trained {len(selected_models)} models!")
        st.info(f"💾 Models saved to: `{save_path.name}`")
        
        # Show quick summary
        st.subheader("Training Summary")
        comparison_df = credit_models.get_comparison_dataframe()
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model highlight
        best_model = comparison_df.iloc[0]['Model']
        best_auc = comparison_df.iloc[0]['ROC-AUC']
        
        st.markdown(f"""
        <div class="success-box">
        <strong>🏆 Best Model: {best_model}</strong><br>
        ROC-AUC Score: {best_auc:.4f}
        </div>
        """, unsafe_allow_html=True)
    
    # Model explanations
    st.divider()
    st.subheader("📖 Model Explanations")
    
    explanations = get_model_explanations()
    
    model_to_explain = st.selectbox(
        "Select a model to learn more",
        list(explanations.keys())
    )
    
    if model_to_explain:
        st.markdown(explanations[model_to_explain])


def render_model_comparison_page():
    """Render the model comparison page."""
    st.header("📈 Model Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first (go to Model Training page)")
        return
    
    credit_models = st.session_state.model_results
    viz = CreditRiskVisualizer()
    
    tabs = st.tabs([
        "📊 Performance Metrics",
        "📈 ROC Curves",
        "🎯 Confusion Matrices",
        "⏱️ Training Time",
        "🎪 Radar Chart"
    ])
    
    # Get comparison data
    comparison_df = credit_models.get_comparison_dataframe()
    
    # Tab 1: Performance Metrics
    with tabs[0]:
        st.subheader("Model Performance Comparison")
        
        # Styled dataframe
        st.dataframe(
            comparison_df.style.background_gradient(
                subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                cmap='RdYlGn'
            ),
            use_container_width=True
        )
        
        # Bar chart comparison
        fig = viz.plot_model_comparison(comparison_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics explanation
        st.markdown(get_evaluation_metrics_guide())
    
    # Tab 2: ROC Curves
    with tabs[1]:
        st.subheader("ROC Curves Comparison")
        
        roc_data = credit_models.get_roc_curve_data(st.session_state.y_test)
        fig = viz.plot_roc_curves(roc_data)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Interpreting ROC Curves:</strong>
        <ul>
            <li>The <strong>diagonal line</strong> represents a random classifier (AUC = 0.5)</li>
            <li>Curves closer to the <strong>top-left corner</strong> indicate better performance</li>
            <li><strong>AUC (Area Under Curve)</strong> summarizes overall discrimination ability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Precision-Recall curves
        st.subheader("Precision-Recall Curves")
        pr_data = credit_models.get_precision_recall_curve_data(st.session_state.y_test)
        fig = viz.plot_precision_recall_curves(pr_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Confusion Matrices
    with tabs[2]:
        st.subheader("Confusion Matrices")
        
        model_name = st.selectbox(
            "Select Model",
            list(credit_models.results.keys())
        )
        
        if model_name:
            result = credit_models.results[model_name]
            fig = viz.plot_confusion_matrix(result.confusion_matrix, model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            cm = result.confusion_matrix
            tn, fp, fn, tp = cm.ravel()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Negatives", f"{tn:,}", help="Correctly predicted no default")
            with col2:
                st.metric("False Positives", f"{fp:,}", help="Incorrectly predicted default")
            with col3:
                st.metric("False Negatives", f"{fn:,}", help="Missed defaults (costly!)")
            with col4:
                st.metric("True Positives", f"{tp:,}", help="Correctly predicted default")
    
    # Tab 4: Training Time
    with tabs[3]:
        st.subheader("Training Time Comparison")
        
        fig = viz.plot_training_time_comparison(comparison_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Training Time Considerations:</strong>
        <ul>
            <li>Faster models are preferred for <strong>real-time scoring</strong></li>
            <li>Complex models may require <strong>more computational resources</strong></li>
            <li>Consider trade-off between <strong>speed and accuracy</strong></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab 5: Radar Chart
    with tabs[4]:
        st.subheader("Multi-Metric Radar Comparison")
        
        fig = viz.plot_model_radar(comparison_df, top_n=5)
        st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_page():
    """Render the feature importance page."""
    st.header("💡 Feature Importance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first (go to Model Training page)")
        return
    
    credit_models = st.session_state.model_results
    viz = CreditRiskVisualizer()
    
    # Get models with feature importance
    models_with_importance = [
        name for name, result in credit_models.results.items()
        if result.feature_importance is not None
    ]
    
    if not models_with_importance:
        st.warning("No models with feature importance available")
        return
    
    st.subheader("Select Model for Feature Importance")
    
    selected_model = st.selectbox(
        "Model",
        models_with_importance
    )
    
    top_n = st.slider("Number of top features to display", 5, 25, 15)
    
    if selected_model:
        result = credit_models.results[selected_model]
        
        fig = viz.plot_feature_importance(
            result.feature_importance,
            selected_model,
            top_n=top_n
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("Feature Importance Ranking")
        
        importance_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ])
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        importance_df['Cumulative %'] = (
            importance_df['Importance'].cumsum() / 
            importance_df['Importance'].sum() * 100
        ).round(2)
        
        st.dataframe(importance_df, use_container_width=True)
        
        # Insights
        st.markdown("""
        <div class="info-box">
        <strong>Understanding Feature Importance:</strong>
        <ul>
            <li><strong>Tree-based models</strong>: Importance based on how much each feature reduces impurity</li>
            <li><strong>Linear models</strong>: Importance based on absolute coefficient values</li>
            <li>Features with <strong>higher importance</strong> have more influence on predictions</li>
            <li>Consider <strong>cumulative importance</strong> for feature selection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def render_data_pipeline_guide():
    """Render the data pipeline guide page."""
    st.header("📚 Data Pipeline Guide")
    
    viz = CreditRiskVisualizer()
    
    tabs = st.tabs([
        "🔄 Pipeline Overview",
        "📥 Data Collection",
        "🏗️ Data Warehouse",
        "📐 Architecture"
    ])
    
    # Tab 1: Pipeline Overview
    with tabs[0]:
        st.subheader("Credit Risk Data Pipeline Overview")
        
        fig = viz.plot_data_pipeline_diagram()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Pipeline Stages
        
        #### 1. Data Sources
        Multiple sources feed into the credit risk data pipeline:
        - **Core Banking System**: Customer accounts, transactions, existing relationships
        - **Loan Origination System**: New loan applications, amounts, terms
        - **Credit Bureau APIs**: External credit scores, history, defaults
        - **KYC/AML Systems**: Identity verification, risk flags
        
        #### 2. Data Ingestion
        Data is ingested through:
        - **Batch Processing**: Historical data, overnight loads
        - **Real-time Streaming**: Application decisions, fraud detection
        
        #### 3. Data Storage (Raw Zone)
        - Store raw data in immutable format
        - Enable data lineage and audit trails
        - Support for schema evolution
        
        #### 4. Data Processing
        - Data cleaning and validation
        - Business rule application
        - Feature engineering
        
        #### 5. Feature Store
        - Centralized feature repository
        - Version control for features
        - Real-time and batch serving
        
        #### 6. Model Training & Deployment
        - Train on historical data
        - Validate on holdout sets
        - Deploy for real-time scoring
        """)
    
    # Tab 2: Data Collection
    with tabs[1]:
        st.subheader("Data Collection Methods")
        st.markdown(get_data_collection_guide())
    
    # Tab 3: Data Warehouse
    with tabs[2]:
        st.subheader("Data Warehouse Design")
        
        st.markdown("""
        ### Star Schema for Credit Risk
        
        The data warehouse follows a star schema design optimized for analytical queries:
        
        ```
                                    ┌─────────────────┐
                                    │  DIM_CUSTOMER   │
                                    ├─────────────────┤
                                    │ customer_id (PK)│
                                    │ age             │
                                    │ income          │
                                    │ employment_len  │
                                    │ home_ownership  │
                                    └────────┬────────┘
                                             │
        ┌─────────────────┐         ┌────────▼────────┐         ┌─────────────────┐
        │ DIM_LOAN_PRODUCT│         │ FACT_LOAN_APP   │         │ DIM_CREDIT_PROF │
        ├─────────────────┤         ├─────────────────┤         ├─────────────────┤
        │ product_id (PK) │◄────────┤ application_id  │────────►│ profile_id (PK) │
        │ loan_type       │         │ customer_id (FK)│         │ credit_score    │
        │ min_amount      │         │ product_id (FK) │         │ credit_history  │
        │ max_amount      │         │ profile_id (FK) │         │ prev_defaults   │
        │ interest_range  │         │ date_id (FK)    │         │ inquiries       │
        └─────────────────┘         │ loan_amount     │         └─────────────────┘
                                    │ interest_rate   │
                                    │ loan_status     │
                                    │ default_flag    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │    DIM_DATE     │
                                    ├─────────────────┤
                                    │ date_id (PK)    │
                                    │ full_date       │
                                    │ year            │
                                    │ quarter         │
                                    │ month           │
                                    └─────────────────┘
        ```
        
        ### Key Design Principles
        
        1. **Fact Table**: Contains measurable events (loan applications, outcomes)
        2. **Dimension Tables**: Descriptive attributes for analysis
        3. **Surrogate Keys**: Use system-generated keys for dimension tables
        4. **Slowly Changing Dimensions**: Track historical changes in customer attributes
        5. **Aggregate Tables**: Pre-computed summaries for common queries
        """)
    
    # Tab 4: Architecture
    with tabs[3]:
        st.subheader("Technology Stack Options")
        
        architecture_df = pd.DataFrame({
            'Component': ['Ingestion', 'Processing', 'Storage', 'Warehouse', 'Orchestration', 'ML Platform'],
            'Open Source': ['Apache Kafka', 'Apache Spark', 'HDFS/MinIO', 'PostgreSQL', 'Apache Airflow', 'MLflow'],
            'AWS': ['Kinesis/MSK', 'Glue/EMR', 'S3', 'Redshift', 'Step Functions', 'SageMaker'],
            'Azure': ['Event Hubs', 'Databricks', 'ADLS', 'Synapse', 'Data Factory', 'Azure ML'],
            'GCP': ['Pub/Sub', 'Dataflow', 'GCS', 'BigQuery', 'Composer', 'Vertex AI']
        })
        
        st.dataframe(architecture_df, use_container_width=True)
        
        st.markdown("""
        ### Considerations for Technology Selection
        
        | Factor | Key Questions |
        |--------|---------------|
        | **Scale** | How much data? How many predictions/day? |
        | **Latency** | Batch processing sufficient, or real-time needed? |
        | **Budget** | Open source vs managed services trade-off |
        | **Skills** | Team expertise and training requirements |
        | **Compliance** | Data residency, security, audit requirements |
        """)


def render_analytics_process_page():
    """Render the analytics process overview page."""
    st.header("📖 Data Analytics Process Overview")
    
    st.markdown("""
    ## Credit Risk Analytics: End-to-End Process
    
    This page provides a comprehensive overview of the data analytics process
    for credit risk assessment, from business understanding to model deployment.
    """)
    
    tabs = st.tabs([
        "1️⃣ Business Understanding",
        "2️⃣ Data Understanding",
        "3️⃣ Data Preparation",
        "4️⃣ Modeling",
        "5️⃣ Evaluation",
        "6️⃣ Deployment"
    ])
    
    # Tab 1: Business Understanding
    with tabs[0]:
        st.subheader("Phase 1: Business Understanding")
        
        st.markdown("""
        ### Objectives
        
        The primary business objective is to **minimize credit losses** while 
        **maximizing loan approvals** (revenue). This requires:
        
        1. **Accurate default prediction**: Identify borrowers likely to default
        2. **Fair lending**: Ensure non-discriminatory credit decisions
        3. **Regulatory compliance**: Meet Basel III, SR 11-7 requirements
        4. **Explainability**: Provide reasons for adverse actions
        
        ### Key Stakeholders
        
        | Stakeholder | Interest | Success Metric |
        |-------------|----------|----------------|
        | Risk Management | Minimize losses | Default rate, Expected Loss |
        | Business | Maximize approvals | Approval rate, Revenue |
        | Compliance | Fair lending | Disparate impact metrics |
        | Operations | Efficient process | Processing time, Cost |
        
        ### Business Questions to Answer
        
        - What is the current default rate?
        - Which customer segments are highest risk?
        - What factors drive loan defaults?
        - How can we reduce defaults without reducing revenue?
        """)
    
    # Tab 2: Data Understanding
    with tabs[1]:
        st.subheader("Phase 2: Data Understanding")
        
        st.markdown("""
        ### Data Sources Inventory
        
        For credit risk, we typically collect data from:
        
        | Source | Data Elements | Update Frequency |
        |--------|---------------|------------------|
        | Application | Demographics, loan request | Real-time |
        | Credit Bureau | Scores, history, defaults | Daily |
        | Banking | Transactions, balances | Real-time |
        | Employment | Verification, income | On-demand |
        
        ### Data Quality Assessment
        
        Key quality dimensions to evaluate:
        
        1. **Completeness**: % of records with all required fields
        2. **Accuracy**: Verification against source systems
        3. **Consistency**: Cross-field validation rules
        4. **Timeliness**: Data freshness for decision-making
        5. **Uniqueness**: Duplicate detection and resolution
        
        ### Exploratory Data Analysis (EDA)
        
        Our EDA process includes:
        
        - **Univariate analysis**: Distribution of each feature
        - **Bivariate analysis**: Feature relationships with target
        - **Multivariate analysis**: Correlations and interactions
        - **Outlier detection**: Identify anomalous values
        - **Missing value patterns**: Analyze missingness mechanisms
        
        *See the "Data Overview" page for detailed EDA of our dataset.*
        """)
    
    # Tab 3: Data Preparation
    with tabs[2]:
        st.subheader("Phase 3: Data Preparation")
        
        st.markdown("""
        ### Preprocessing Steps
        
        #### 1. Data Cleaning
        
        ```
        Raw Data → Remove duplicates → Handle missing values → Fix data types → Clean Data
        ```
        
        #### 2. Feature Engineering
        
        Creating meaningful features from raw data:
        
        | New Feature | Formula | Rationale |
        |-------------|---------|-----------|
        | Debt-to-Income | loan_amnt / income | Repayment capacity |
        | Credit Utilization | Used / Limit | Credit behavior |
        | Payment History | On-time / Total | Reliability indicator |
        
        #### 3. Data Transformation
        
        - **Encoding**: Convert categorical to numerical
        - **Scaling**: Normalize numerical ranges
        - **Binning**: Create categorical from numerical (age groups)
        
        #### 4. Feature Selection
        
        Methods used:
        - Correlation analysis (remove highly correlated)
        - Feature importance from tree models
        - Recursive feature elimination
        - Domain expertise
        
        #### 5. Class Imbalance Handling
        
        For imbalanced credit data (~5-15% default rate):
        - **SMOTE**: Synthetic minority oversampling
        - **Class weights**: Penalize majority class errors
        - **Threshold tuning**: Adjust decision boundary
        
        *See the "Preprocessing" page for our specific implementation.*
        """)
    
    # Tab 4: Modeling
    with tabs[3]:
        st.subheader("Phase 4: Modeling")
        
        st.markdown("""
        ### Model Selection Strategy
        
        We evaluate multiple model families:
        
        #### Linear Models
        - **Logistic Regression**: Interpretable baseline, regulatory preferred
        - Provides clear coefficient interpretation
        
        #### Tree-based Models
        - **Decision Tree**: Single tree, highly interpretable
        - **Random Forest**: Ensemble of trees, improved accuracy
        - **Gradient Boosting**: Sequential tree building
        
        #### Advanced Ensembles
        - **XGBoost**: Regularized gradient boosting, state-of-the-art
        - **LightGBM**: Fast, memory-efficient boosting
        
        ### Training Process
        
        ```
        ┌───────────┐     ┌─────────────┐     ┌──────────────┐
        │ Train Set │────►│ Model Train │────►│ Trained Model│
        └───────────┘     └─────────────┘     └──────────────┘
              │                 │                    │
              │           ┌─────▼─────┐              │
              │           │ Cross-Val │              │
              │           └───────────┘              │
              │                                      ▼
        ┌─────▼─────┐                         ┌──────────────┐
        │ Hold-out  │◄────────────────────────│  Evaluation  │
        └───────────┘                         └──────────────┘
        ```
        
        ### Hyperparameter Tuning
        
        Methods used:
        - **Grid Search**: Exhaustive search over parameter grid
        - **Random Search**: Random sampling of parameter space
        - **Bayesian Optimization**: Intelligent search using Optuna
        
        *See the "Model Training" page to train models.*
        """)
    
    # Tab 5: Evaluation
    with tabs[4]:
        st.subheader("Phase 5: Evaluation")
        
        st.markdown("""
        ### Evaluation Metrics
        
        #### Classification Metrics
        
        | Metric | Formula | Use Case |
        |--------|---------|----------|
        | Accuracy | (TP+TN)/(All) | Overall correctness |
        | Precision | TP/(TP+FP) | Minimize false alarms |
        | Recall | TP/(TP+FN) | Catch all defaults |
        | F1 | 2×P×R/(P+R) | Balance P and R |
        
        #### Ranking Metrics
        
        | Metric | Range | Interpretation |
        |--------|-------|----------------|
        | ROC-AUC | 0.5-1.0 | Discrimination ability |
        | Gini | 0-1 | 2×AUC - 1 |
        | KS | 0-1 | Maximum separation |
        
        ### Business Metrics
        
        Beyond statistical metrics, evaluate:
        
        - **Expected Loss**: PD × LGD × EAD
        - **Profit/Loss**: Revenue from goods - Loss from bads
        - **Approval Rate**: % of applications approved
        
        ### Model Fairness
        
        Ensure fair lending by checking:
        
        - Equal opportunity across protected groups
        - Similar precision/recall for different demographics
        - No disparate impact in approval rates
        
        *See the "Model Comparison" page for our model evaluation.*
        """)
    
    # Tab 6: Deployment
    with tabs[5]:
        st.subheader("Phase 6: Deployment")
        
        st.markdown("""
        ### Deployment Options
        
        #### 1. Batch Scoring
        
        - Score entire portfolios overnight
        - Update risk ratings periodically
        - Generate management reports
        
        ```
        Daily Data → Feature Pipeline → Model Scoring → Risk Database
        ```
        
        #### 2. Real-time Scoring
        
        - Instant credit decisions
        - API-based model serving
        - Sub-second response times
        
        ```
        Application → API Gateway → Model Server → Decision
        ```
        
        ### MLOps Best Practices
        
        #### Model Versioning
        - Track model versions with metadata
        - Enable rollback to previous versions
        - Document changes and performance
        
        #### Monitoring
        - **Data Drift**: Feature distribution changes
        - **Concept Drift**: Relationship changes
        - **Performance Decay**: Accuracy degradation
        
        #### Retraining Triggers
        - Scheduled (quarterly, yearly)
        - Performance-based (AUC drops below threshold)
        - Data-based (significant drift detected)
        
        ### Champion-Challenger Framework
        
        ```
        ┌────────────────┐    ┌────────────────┐
        │   Champion     │    │   Challenger   │
        │  (Production)  │    │   (Testing)    │
        │    90%         │    │     10%        │
        └───────┬────────┘    └───────┬────────┘
                │                     │
                └──────────┬──────────┘
                           │
                     ┌─────▼─────┐
                     │  Compare  │
                     │ Results   │
                     └───────────┘
        ```
        
        ### Regulatory Considerations
        
        - **Model Documentation**: SR 11-7 compliant documentation
        - **Validation**: Independent model validation
        - **Audit Trail**: Log all predictions and reasons
        - **Adverse Action**: Provide reason codes for denials
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Data Overview":
        render_data_overview_page()
    elif page == "🔧 Preprocessing":
        render_preprocessing_page()
    elif page == "🤖 Model Training":
        render_model_training_page()
    elif page == "📈 Model Comparison":
        render_model_comparison_page()
    elif page == "💡 Feature Importance":
        render_feature_importance_page()
    elif page == "📚 Data Pipeline Guide":
        render_data_pipeline_guide()
    elif page == "📖 Analytics Process":
        render_analytics_process_page()


if __name__ == "__main__":
    main()
