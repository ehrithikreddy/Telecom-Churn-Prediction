Telecom Customer Churn Prediction System

A comprehensive machine learning system for predicting customer churn in the telecommunications industry. This project implements advanced data preprocessing, feature engineering, multiple classification algorithms, and provides both batch processing and real-time prediction capabilities through a REST API.

🌟 Features
Core Functionality
Comprehensive Data Analysis: Detailed exploratory data analysis with statistical summaries and visualizations
Advanced Preprocessing: Multiple preprocessing techniques including KNN imputation, feature engineering, and encoding strategies
Feature Selection: Multi-method feature selection using univariate selection, mutual information, tree-based importance, and L1 regularization
Multiple ML Models: Implementation of 8+ classification algorithms including ensemble methods
Hyperparameter Tuning: GridSearchCV with cross-validation for optimal model performance
Model Evaluation: Comprehensive metrics including ROC-AUC, precision, recall, F1-score, and confusion matrices
Advanced Capabilities
Database Integration: Full PostgreSQL integration for customer data and prediction storage
REST API: Production-ready Flask API for real-time churn predictions
Batch Processing: Automated batch prediction pipeline for large customer datasets
Visualization Suite: Comprehensive plotting and chart generation for analysis insights
Model Persistence: Automated model saving and loading for production deployment
Jupyter Integration: Automatic Jupyter notebook generation for interactive analysis
Business Intelligence
Risk Scoring: Automated customer risk level classification (Low/Medium/High)
Business Recommendations: Actionable insights based on prediction results
Customer Lifecycle Analysis: Detailed customer segmentation and behavior patterns
Retention Strategy Support: Targeted recommendations for customer retention campaigns
📋 Table of Contents
Installation
Quick Start
Database Setup
Usage Examples
API Documentation
Model Performance
File Structure
Configuration
Contributing
License
🚀 Installation
Prerequisites
Python 3.8+
PostgreSQL 12+ (optional, for database features)
4GB+ RAM recommended for model training
Core Dependencies
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
pip install flask sqlalchemy psycopg2-binary joblib pathlib
Complete Installation
bash
# Clone the repository
git clone <repository-url>
cd telecom-churn-prediction

# Install all dependencies
pip install -r requirements.txt

# Alternative: Install individually
pip install pandas==1.5.3 numpy==1.24.3 matplotlib==3.7.1 seaborn==0.12.2
pip install scikit-learn==1.3.0 xgboost==1.7.6 lightgbm==4.0.0
pip install flask==2.3.2 sqlalchemy==2.0.19 psycopg2-binary==2.9.7
pip install joblib==1.3.1 pathlib warnings logging textwrap
Data Requirements
Download the Telco Customer Dataset (Telco-Dataset.csv)
Place in the project root directory or update the path in the configuration
⚡ Quick Start
Basic Analysis
python
from proj_3 import TelecomChurnPredictor

# Initialize the predictor
analyzer = TelecomChurnPredictor('Telco-Dataset.csv')

# Run complete analysis pipeline
analyzer.run_complete_analysis()
Single Customer Prediction
python
from proj_3 import TelecomChurnPredictor
import joblib

# Load trained model
model = joblib.load('models/best_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Customer data
customer_data = {
    "customerID": "NEW-001",
    "tenure": 12,
    "MonthlyCharges": 75.50,
    "TotalCharges": "906.00",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    # ... additional features
}

# Make prediction (see full example in code)
REST API Server
python
from proj_3 import create_flask_api

# Create and run API
app = create_flask_api()
app.run(debug=True, host='0.0.0.0', port=5000)
🗄️ Database Setup
PostgreSQL Configuration
sql
-- Create database
CREATE DATABASE churn_db;

-- Connect to database and create user
CREATE USER churn_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;
Python Database Setup
python
from proj_3 import DatabaseManager

# Initialize database manager
db = DatabaseManager(
    host='localhost',
    database='churn_db',
    user='churn_user',
    password='your_password',
    port=5432
)

# Create tables
db.create_tables()

# Load customer data
db.save_customer_data(customer_dataframe)
Database Schema
The system creates three main tables:
customers: Customer profile information
churn_predictions: Prediction results and probabilities
model_performance: Model metrics tracking
📊 Usage Examples
Complete Pipeline Execution
python
# Full analysis pipeline
analyzer = TelecomChurnPredictor('Telco-Dataset.csv')

# Step 1: Load and validate data
analyzer.load_and_validate_data()

# Step 2: Exploratory data analysis
analyzer.comprehensive_eda()

# Step 3: Visualization and correlation analysis
analyzer.advanced_visualization()

# Step 4: Data preprocessing and feature engineering
analyzer.advanced_preprocessing()

# Step 5: Feature selection and model training
analyzer.feature_selection()

# Step 6: Generate comprehensive reports
analyzer.generate_comprehensive_report()
Batch Prediction Pipeline
python
from proj_3 import batch_prediction_pipeline

# Run batch predictions on all customers in database
results = batch_prediction_pipeline()
print(f"Processed {len(results)} customers")
Jupyter Notebook Generation
python
from proj_3 import create_jupyter_notebook

# Generate interactive Jupyter notebook
create_jupyter_notebook()
# Output: Telecom_Churn_Analysis.ipynb
🌐 API Documentation
Base URL
text
http://localhost:5000
Endpoints
Health Check
text
GET /health
Response:
json
{
    "status": "API is running",
    "model_loaded": true,
    "model_name": "Random Forest",
    "model_performance": {...}
}
Predict Churn
text
POST /predict_churn
Content-Type: application/json
Request Body:
json
{
    "customerID": "TEST-001",
    "tenure": 12,
    "MonthlyCharges": 50.0,
    "TotalCharges": "600.0",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes"
}
Response:
json
{
    "customer_id": "TEST-001",
    "churn_prediction": 1,
    "churn_probability": 0.7845,
    "risk_level": "High",
    "model_used": "Random Forest",
    "status": "success"
}
Model Information
text
GET /model_info
Response:
json
{
    "best_model_name": "Random Forest",
    "performance_metrics": {
        "Accuracy": 0.8234,
        "Precision": 0.7891,
        "Recall": 0.7456,
        "F1-Score": 0.7668,
        "ROC-AUC": 0.8567
    },
    "training_date": "2025-07-30T06:00:00",
    "feature_count": 47
}
cURL Examples
bash
# Health check
curl -X GET http://localhost:5000/health

# Make prediction
curl -X POST http://localhost:5000/predict_churn \
     -H "Content-Type: application/json" \
     -d '{"customerID":"TEST-001","tenure":12,"MonthlyCharges":50.0,...}'

# Get model info
curl -X GET http://localhost:5000/model_info
📈 Model Performance
Implemented Algorithms
Logistic Regression - Linear baseline with regularization
K-Nearest Neighbors - Instance-based learning
Decision Tree - Interpretable rule-based classifier
Random Forest - Ensemble bagging method
XGBoost - Gradient boosting optimization
Gradient Boosting - Sequential ensemble learning
Extra Trees - Extremely randomized trees
Voting Ensemble - Meta-ensemble combining multiple models
LightGBM - High-performance gradient boosting
Feature Engineering
Tenure Grouping: Customer lifecycle categorization
Service Adoption Metrics: Customer engagement scoring
Value Metrics: Average charges and service efficiency
Risk Indicators: Contract and payment method risk factors
Customer Lifecycle: Behavioral stage classification
Model Selection Criteria
Primary Metric: ROC-AUC score for imbalanced datasets
Cross-Validation: 5-fold stratified cross-validation
Hyperparameter Tuning: GridSearchCV with extensive parameter grids
Business Impact: Precision-recall balance for cost-effective targeting
Typical Performance Ranges
ROC-AUC: 0.82-0.89
Precision: 0.74-0.85
Recall: 0.71-0.82
F1-Score: 0.73-0.83
📁 File Structure
text
telecom-churn-prediction/
├── proj_3.py                          # Main application file
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── Telco-Dataset.csv                  # Input dataset
├── models/                            # Trained models directory
│   ├── best_churn_model.pkl          # Best performing model
│   ├── scaler.pkl                    # Feature scaler
│   ├── feature_columns.pkl           # Feature list
│   └── model_info.pkl                # Model metadata
├── plots/                            # Generated visualizations
│   ├── churn_analysis_categorical_*.png
│   ├── churn_analysis_numerical_*.png
│   ├── correlation_heatmap_*.png
│   ├── top_correlations_*.png
│   └── model_evaluation_suite_*.png
├── Telecom_Churn_Analysis.ipynb     # Generated Jupyter notebook
└── logs/                             # Application logs
    └── churn_prediction.log
⚙️ Configuration
Environment Variables
bash
# Database configuration
export DB_HOST=localhost
export DB_NAME=churn_db
export DB_USER=churn_user
export DB_PASSWORD=your_password
export DB_PORT=5432

# API configuration
export API_HOST=0.0.0.0
export API_PORT=5000
export API_DEBUG=True

# Model configuration
export MODEL_RANDOM_STATE=42
export CV_FOLDS=5
export TEST_SIZE=0.2
Custom Configuration
python
# Database settings
db_config = {
    'host': 'localhost',
    'database': 'churn_db',
    'user': 'postgres',
    'password': 'your_password',
    'port': 5432
}

# Model settings
model_config = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'scoring_metric': 'roc_auc'
}
🎯 Business Applications
Customer Retention
Proactive Intervention: Identify high-risk customers before they churn
Targeted Campaigns: Focus retention efforts on customers most likely to respond
Resource Optimization: Allocate retention budget efficiently based on churn probability
Strategic Planning
Customer Segmentation: Understand different customer behavior patterns
Product Development: Identify services that reduce churn risk
Pricing Strategy: Optimize pricing models based on churn sensitivity
Operational Excellence
Automated Alerts: Real-time notifications for high-risk customers
Performance Tracking: Monitor model performance and business impact
A/B Testing: Test retention strategies with predicted churn segments
🔧 Advanced Features
Feature Selection Methods
Univariate Selection: Statistical significance testing
Mutual Information: Information-theoretic feature ranking
Tree-based Importance: Random Forest feature importance
L1 Regularization: Lasso-based feature selection
Voting System: Combined selection across multiple methods
Preprocessing Pipeline
Missing Value Handling: KNN imputation for numerical features
Feature Engineering: Domain-specific feature creation
Encoding Strategies: Binary and one-hot encoding for categoricals
Scaling: Robust scaling for outlier resistance
Feature Selection: Multi-method voting system
Model Training Pipeline
Data Splitting: Stratified train-test split
Cross-Validation: 5-fold stratified cross-validation
Hyperparameter Tuning: GridSearchCV optimization
Model Ensemble: Voting classifier combination
Performance Evaluation: Comprehensive metrics suite
🚨 Troubleshooting
Common Issues
Dataset Not Found
python
# Solution: Update dataset path
analyzer = TelecomChurnPredictor('path/to/your/Telco-Dataset.csv')

# Or place dataset in project root directory
Database Connection Failed
python
# Check PostgreSQL service
sudo service postgresql start

# Verify connection parameters
db = DatabaseManager(host='localhost', database='churn_db', ...)

# Test connection
psql -h localhost -U churn_user -d churn_db
Model Files Not Found
bash
# Ensure models directory exists and contains required files
ls -la models/
# Should contain: best_churn_model.pkl, scaler.pkl, feature_columns.pkl, model_info.pkl

# If missing, run training pipeline first
python -c "from proj_3 import TelecomChurnPredictor; analyzer = TelecomChurnPredictor('Telco-Dataset.csv'); analyzer.run_complete_analysis()"
Memory Issues
python
# Reduce model complexity for limited memory environments
models_config = {
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=50),  # Reduced from 100-300
        'params': {'max_depth': [5, 10]}  # Reduced parameter grid
    }
}
Performance Optimization
Data Sampling: Use subset for initial development and testing
Feature Reduction: Implement more aggressive feature selection
Model Selection: Focus on faster algorithms for time-sensitive applications
Parallel Processing: Utilize n_jobs=-1 in sklearn algorithms
📊 Monitoring and Maintenance
Model Performance Monitoring
python
# Track model performance over time
def monitor_model_performance(predictions, actual_labels):
    current_accuracy = accuracy_score(actual_labels, predictions)
    
    # Compare with baseline performance
    baseline_accuracy = 0.82  # From training
    
    if current_accuracy < baseline_accuracy * 0.95:
        print("⚠️ Model performance degraded. Consider retraining.")
Data Drift Detection
python
# Monitor feature distributions
def check_data_drift(new_data, reference_data):
    # Compare feature distributions
    for column in new_data.columns:
        # Statistical tests for drift detection
        pass
Automated Retraining
python
# Schedule periodic model retraining
def schedule_retraining():
    # Load new data
    # Retrain models
    # Evaluate performance
    # Deploy if improved
    pass
🤝 Contributing
Development Setup
bash
# Clone repository
git clone <repository-url>
cd telecom-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt
Code Standards
PEP 8: Follow Python style guide
Type Hints: Use type annotations where possible
Documentation: Comprehensive docstrings for all functions
Testing: Unit tests for critical functions
Logging: Appropriate logging levels and messages
Contribution Process
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
Testing
bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Check code coverage
python -m pytest --cov=proj_3 tests/
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments
Scikit-learn: Machine learning library
XGBoost: Gradient boosting framework
LightGBM: High-performance gradient boosting
Flask: Web framework for API development
PostgreSQL: Database system
Matplotlib/Seaborn: Visualization libraries
📞 Support
Documentation
Code Comments: Extensive inline documentation
Function Docstrings: Detailed parameter and return value descriptions
Type Hints: Clear function signatures
Contact Information
Issues: Use GitHub Issues for bug reports and feature requests
Discussions: Use GitHub Discussions for questions and community support
Email: [Your contact email]
FAQ
Q: How accurate is the churn prediction?
A: The model typically achieves 82-89% ROC-AUC score, with precision around 74-85% and recall 71-82%.
Q: Can I use my own dataset?
A: Yes, but ensure your dataset has similar structure to the Telco dataset. You may need to modify preprocessing steps.
Q: How often should I retrain the model?
A: Recommended monthly or quarterly retraining, depending on data drift and business changes.
Q: Can this work with other industries?
A: The framework is adaptable, but feature engineering and business logic would need industry-specific modifications.
Last Updated: July 30, 2025
Version: 1.0.0
Python Version: 3.8+
License: MIT
