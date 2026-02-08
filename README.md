# Credit Risk Assessment Platform

An advanced, fully-integrated Streamlit application for credit risk assessment using machine learning. This platform provides comprehensive data analysis, model comparison, and educational content about the data analytics process.

## 🚀 Features

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive visualization of data distributions, correlations, and patterns
- **Feature Documentation**: Detailed descriptions of all features with business context
- **Missing Value Analysis**: Automated detection and visualization of missing data
- **Outlier Detection**: IQR-based outlier analysis with visual summaries

### Machine Learning
- **9+ ML Models**: Compare Logistic Regression, Random Forest, XGBoost, LightGBM, and more
- **Automated Training**: One-click model training with progress tracking
- **Performance Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC comparison
- **Feature Importance**: Understand which features drive predictions

### Preprocessing Pipeline
- **Missing Value Imputation**: Median for numerical, mode for categorical
- **Outlier Handling**: Optional IQR-based clipping
- **Feature Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler normalization
- **Class Imbalance**: SMOTE oversampling support

### Educational Content
- **Data Pipeline Guide**: Learn how to build production data pipelines
- **Analytics Process**: Comprehensive CRISP-DM methodology walkthrough
- **Model Explanations**: Detailed descriptions of each ML algorithm
- **Best Practices**: Industry-standard approaches for credit risk

## 📁 Project Structure

```
credit-risk/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration parameters
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and validation
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── models.py               # ML model training and evaluation
│   └── visualizations.py       # Plotly visualization utilities
│
├── data/
│   └── credit_risk_dataset.csv # Credit risk dataset
│
├── models/                     # Saved model files
│   └── .gitkeep
│
├── logs/                       # Log files
│   └── .gitkeep
│
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/mohamadsolouki/credit-risk.git
cd credit-risk
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

## 📊 Dataset

The dataset contains credit risk information with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numerical | Age of the loan applicant |
| `person_income` | Numerical | Annual income in USD |
| `person_home_ownership` | Categorical | Home ownership status (RENT, OWN, MORTGAGE, OTHER) |
| `person_emp_length` | Numerical | Employment length in years |
| `loan_intent` | Categorical | Purpose of the loan |
| `loan_grade` | Categorical | Risk grade (A-G) |
| `loan_amnt` | Numerical | Loan amount requested |
| `loan_int_rate` | Numerical | Interest rate (%) |
| `loan_status` | Binary | Target: 0 = No Default, 1 = Default |
| `loan_percent_income` | Numerical | Loan as % of income |
| `cb_person_default_on_file` | Binary | Historical default (Y/N) |
| `cb_person_cred_hist_length` | Numerical | Credit history length in years |

## 🤖 Available Models

| Model | Type | Description |
|-------|------|-------------|
| Logistic Regression | Linear | Interpretable baseline, regulatory preferred |
| Decision Tree | Tree | Single tree, highly interpretable |
| Random Forest | Ensemble | Multiple trees, improved accuracy |
| Gradient Boosting | Ensemble | Sequential error correction |
| XGBoost | Advanced Ensemble | State-of-the-art gradient boosting |
| LightGBM | Advanced Ensemble | Fast, memory-efficient boosting |
| K-Nearest Neighbors | Instance-based | Similar case matching |
| Naive Bayes | Probabilistic | Fast baseline classifier |
| AdaBoost | Ensemble | Adaptive boosting |

## 📈 Application Pages

### 1. Home
Overview of the platform with quick statistics and getting started guide.

### 2. Data Overview
- Dataset preview
- Feature documentation
- Target variable analysis
- Distribution visualizations
- Correlation analysis

### 3. Preprocessing
- Missing value analysis
- Outlier detection
- Apply preprocessing with customizable options
- View preprocessing log

### 4. Model Training
- Select models to train
- One-click training with progress tracking
- View training summary
- Model explanations

### 5. Model Comparison
- Performance metrics comparison
- ROC curves
- Confusion matrices
- Training time analysis
- Radar chart visualization

### 6. Feature Importance
- Feature importance by model
- Ranking table
- Cumulative importance analysis

### 7. Data Pipeline Guide
- Pipeline architecture overview
- Data collection methods
- Data warehouse design
- Technology stack options

### 8. Analytics Process
- CRISP-DM methodology
- Business understanding
- Data preparation
- Modeling and evaluation
- Deployment considerations

## 🔧 Configuration

### Application Settings
Edit `config/settings.py` to customize:
- Feature configurations
- Model hyperparameters
- Visualization settings
- Random state and test size

### Streamlit Theme
Edit `.streamlit/config.toml` to customize the UI theme.

## 📚 Data Analytics Process

This application follows the **CRISP-DM** methodology:

1. **Business Understanding**: Define credit risk objectives
2. **Data Understanding**: Explore and analyze the dataset
3. **Data Preparation**: Clean, transform, and engineer features
4. **Modeling**: Train and compare multiple ML models
5. **Evaluation**: Assess model performance with multiple metrics
6. **Deployment**: Guidelines for production deployment

## 🏗️ Data Pipeline Architecture

The application includes documentation on building production data pipelines:

```
[Data Sources] → [Ingestion] → [Raw Zone] → [Processing] → [Feature Store] → [ML Models]
```

### Covered Topics:
- Data sources (Core Banking, Credit Bureau, KYC systems)
- Batch vs real-time ingestion
- Data warehouse schema design (Star Schema)
- Feature store implementation
- Model serving architecture
- MLOps best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: Credit risk dataset for educational purposes
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [LightGBM](https://lightgbm.readthedocs.io/)
- Visualizations powered by [Plotly](https://plotly.com/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for credit risk analysis**