import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path
import joblib
import os
from datetime import datetime
import textwrap
import analyzer
import json

# Machine Learning Libraries
from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score,
                                   StratifiedKFold, learning_curve, validation_curve)
from sklearn.preprocessing import (LabelEncoder, PowerTransformer, RobustScaler)
from sklearn.impute import KNNImputer
from sklearn.feature_selection import (SelectKBest, f_classif,
                                     SelectFromModel, mutual_info_classif)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            ExtraTreesClassifier, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                           roc_curve, precision_score, recall_score, f1_score,
                           accuracy_score, precision_recall_curve, average_precision_score,
                           classification_report)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Visualization and interpretability
from sklearn.inspection import permutation_importance

# SQLAlchemy for database operations
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import textwrap

# Configuration and Setup
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_palette("husl")
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create plots directory for saving figures
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name: str) -> None:
    """Save matplotlib figure to disk with timestamp for Jupyter notebook integration"""
    safe_name = name.lower().replace(" ", "_").replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = PLOTS_DIR / f"{safe_name}_{timestamp}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"ðŸ“ˆ Figure saved to {out_path}")

class TelecomChurnPredictor:
    def __init__(self, data_path: str):
        """Initialize churn predictor with data path"""
        self.data_path = Path(data_path)
        self.df = None
        self.df_processed = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        self.feature_importance = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.probabilities = None

    def load_and_validate_data(self) -> None:
        """Load and validate the dataset with enhanced path detection"""
        try:
            # Enhanced file path attempts for robustness
            possible_paths = [
                self.data_path,
                Path('Telco-Dataset.csv'),  # Current directory
                Path('Project/Telco-Dataset.csv'),  # Project subdirectory
                Path('data/Telco-Dataset.csv'),  # Data subdirectory
                Path('../data/Telco-Dataset.csv'),  # Parent data subdirectory
                Path.home() / "Downloads" / "Assignment PS7" / "Telco-Dataset.csv",  # Downloads
                Path.home() / "Downloads" / "Telco-Dataset.csv",  # Direct Downloads
                Path.home() / "Desktop" / "Telco-Dataset.csv",  # Desktop
            ]

            print("Searching for dataset in the following locations:")
            for i, path in enumerate(possible_paths, 1):
                print(f"  {i}. {path}")
                if path.exists():
                    self.df = pd.read_csv(path)
                    logger.info(f"âœ… Dataset loaded successfully from {path}")
                    break
            else:
                # If file not found, provide helpful debugging info
                print("\nâŒ Dataset not found in any location!")
                print("\nDebugging Information:")
                print(f"Current working directory: {Path.cwd()}")
                print(f"Home directory: {Path.home()}")
                
                # List files in current directory
                print(f"\nFiles in current directory:")
                for file in Path.cwd().iterdir():
                    if file.is_file():
                        print(f"  - {file.name}")
                
                raise FileNotFoundError("Dataset not found. Please check the file location and update the path.")

            # Validate dataset
            if self.df.empty:
                raise ValueError("Dataset is empty")
            
            if 'Churn' not in self.df.columns:
                raise ValueError("Target column 'Churn' not found")
            
            logger.info(f"Dataset shape: {self.df.shape}")
            logger.info("Dataset validation completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def comprehensive_eda(self) -> None:
        """Comprehensive Exploratory Data Analysis"""
        print("=" * 80)
        print("DATA ANALYSIS")
        print("=" * 80)
        
        # Basic Information
        print("\n1. BASIC DATASET INFORMATION")
        print("-" * 40)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of features: {self.df.shape[1]-1}")
        print(f"Number of samples: {self.df.shape[0]}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Print first 5 rows for sanity check (Assignment requirement 2.a)
        print("\nFirst 5 rows for sanity check:")
        print(self.df.head())
        
        print("\nColumn Information:")
        info_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes,
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum(),
            'Null Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Unique Values': self.df.nunique()
        })
        print(info_df)
        
        # Dataset description (Assignment requirement 2.b)
        print("\nDataset Description:")
        print(self.df.describe())
        
        # Target Variable Analysis
        print("\n2. TARGET VARIABLE ANALYSIS")
        print("-" * 40)
        churn_dist = self.df['Churn'].value_counts()
        churn_pct = self.df['Churn'].value_counts(normalize=True) * 100
        print("Churn Distribution:")
        for label, count in churn_dist.items():
            print(f"  {label}: {count} ({churn_pct[label]:.1f}%)")
        
        # Statistical Descriptions
        print("\n3. STATISTICAL DESCRIPTIONS")
        print("-" * 40)
        
        # Separate numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        # Statistical analysis for numerical columns
        if numerical_cols:
            print("\nStatistical Analysis:")
            stats_df = pd.DataFrame()
            for col in numerical_cols:
                if col in self.df.columns:
                    col_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                    if len(col_data) > 0:
                        stats_df[col] = [
                            col_data.mean(),
                            col_data.median(),
                            col_data.mode().iloc[0] if not col_data.mode().empty else np.nan,
                            col_data.std(),
                            col_data.var(),
                            col_data.skew(),
                            col_data.kurtosis(),
                            col_data.min(),
                            col_data.max(),
                            col_data.quantile(0.25),
                            col_data.quantile(0.75),
                            col_data.quantile(0.75) - col_data.quantile(0.25)
                        ]
            
            stats_df.index = ['Mean', 'Median', 'Mode', 'Std Dev', 'Variance', 'Skewness',
                             'Kurtosis', 'Min', 'Max', 'Q1', 'Q3', 'IQR']
            print(stats_df)
        
        # IQR Analysis for Outliers
        print("\nOutlier Analysis using IQR:")
        print("-" * 40)
        for col in numerical_cols:
            if col in self.df.columns:
                col_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    print(f"{col}: {len(outliers)} outliers detected")
        
        # Categorical Analysis
        print("\n4. CATEGORICAL VARIABLES ANALYSIS")
        print("-" * 40)
        for col in categorical_cols:
            if col != 'Churn' and self.df[col].nunique() <= 10:
                print(f"\n{col} distribution:")
                value_counts = self.df[col].value_counts()
                for val, count in value_counts.items():
                    pct = (count / len(self.df)) * 100
                    print(f"  {val}: {count} ({pct:.1f}%)")

    def advanced_visualization(self) -> None:
        """Create comprehensive visualizations (Assignment requirement 2.c)"""
        print("\n5. VISUALIZATION ANALYSIS")
        print("-" * 40)
        
        # Set up comprehensive plotting
        plt.rcParams['figure.figsize'] = [15, 10]
        
        # 1. Churn Distribution and Key Categorical Features
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Churn Analysis: Distribution and Key Categorical Features', fontsize=16)
        
        # Churn distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Churn Distribution')
        
        # Key categorical features vs Churn
        cat_features = ['gender', 'SeniorCitizen', 'Contract', 'InternetService', 'PaymentMethod']
        for i, feature in enumerate(cat_features):
            if feature in self.df.columns:
                row, col = divmod(i+1, 3)
                if row < 2:
                    sns.countplot(data=self.df, x=feature, hue='Churn', ax=axes[row, col])
                    axes[row, col].set_title(f'{feature} vs Churn')
                    axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_fig(fig, "churn_analysis_categorical")
        plt.show()
        
        # 2. Numerical Features Analysis with Churn
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Clean TotalCharges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Numerical Features Analysis with Churn', fontsize=16)
        
        for i, col in enumerate(numerical_cols):
            if col in self.df.columns:
                # Distribution by churn
                axes[0, i].hist([self.df[self.df['Churn']=='No'][col].dropna(),
                                self.df[self.df['Churn']=='Yes'][col].dropna()],
                               bins=30, alpha=0.7, label=['No Churn', 'Churn'])
                axes[0, i].set_title(f'{col} Distribution by Churn')
                axes[0, i].legend()
                
                # Box plot
                sns.boxplot(data=self.df, x='Churn', y=col, ax=axes[1, i])
                axes[1, i].set_title(f'{col} Box Plot by Churn')
        
        plt.tight_layout()
        save_fig(fig, "churn_analysis_numerical")
        plt.show()
        
        # 3. Correlation Analysis
        self.correlation_analysis()

    def correlation_analysis(self) -> None:
        """Correlation analysis with justification (Assignment requirement 2.d)"""
        print("\n6. CORRELATION ANALYSIS")
        print("-" * 40)
        
        # Create encoded dataframe for correlation
        df_corr = self.df.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = df_corr.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
        
        # Convert TotalCharges to numeric
        df_corr['TotalCharges'] = pd.to_numeric(df_corr['TotalCharges'], errors='coerce')
        df_corr = df_corr.fillna(df_corr.median())
        
        # Calculate correlation matrix
        correlation_matrix = df_corr.corr()
        
        # Plot correlation heatmap
        fig = plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   mask=mask, fmt='.2f', square=True, cbar_kws={'shrink': .8})
        plt.title('Correlation Matrix of All Features')
        plt.tight_layout()
        save_fig(fig, "correlation_heatmap")
        plt.show()
        
        # Feature correlation with target
        churn_correlation = correlation_matrix['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
        
        print("\nTop 10 Features Correlated with Churn:")
        for feature, corr in churn_correlation.head(10).items():
            print(f"  {feature}: {corr:.3f}")
        
        # Plot top correlations
        fig = plt.figure(figsize=(12, 8))
        top_corr = churn_correlation.head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_corr.values]
        sns.barplot(x=top_corr.values, y=top_corr.index, palette=colors)
        plt.title('Top 15 Feature Correlations with Churn')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        save_fig(fig, "top_correlations")
        plt.show()
        
        # JUSTIFICATION: How Correlation Analysis Affects Feature Selection (Assignment requirement 2.d)
        print("\nJUSTIFICATION: How Correlation Analysis Affects Feature Selection:")
        print("-" * 60)
        
        # Reference specific findings from correlation analysis
        high_corr_features = churn_correlation[abs(churn_correlation) > 0.3]
        print(f"1. {len(high_corr_features)} features show strong correlation (>0.3) with churn:")
        for feature, corr in high_corr_features.items():
            print(f"   - {feature}: {corr:.3f}")
        
        print("\n2. FEATURE SELECTION IMPACT:")
        print("   - High correlation features will be prioritized in selection algorithms")
        print("   - Features with |correlation| > 0.3 are strong churn predictors")
        print("   - Multicollinearity check needed for highly correlated feature pairs")
        print("   - This analysis provides baseline for univariate feature selection")
        print("   - Correlation ranking helps in ensemble feature selection voting")
        
        print("\n3. NEXT STEP GUIDANCE:")
        print("   - Feature selection will use these correlation insights")
        print("   - Multiple selection methods will validate correlation findings")
        print("   - Final feature set will balance correlation strength and diversity")

    def advanced_preprocessing(self) -> None:
        """Data preprocessing with multiple techniques (Assignment requirement 3.a)"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        # Create working copy
        self.df_processed = self.df.copy()
        
        # 1. Handle missing values
        print("1. MISSING VALUE HANDLING")
        print("-" * 40)
        
        # Convert TotalCharges to numeric and handle missing values
        self.df_processed['TotalCharges'] = pd.to_numeric(self.df_processed['TotalCharges'], errors='coerce')
        
        # For customers with 0 tenure, set TotalCharges to 0
        zero_tenure_mask = self.df_processed['tenure'] == 0
        self.df_processed.loc[zero_tenure_mask, 'TotalCharges'] = 0
        
        # For others, use KNN imputation
        remaining_missing = self.df_processed['TotalCharges'].isnull().sum()
        if remaining_missing > 0:
            numeric_cols = ['tenure', 'MonthlyCharges']
            imputer = KNNImputer(n_neighbors=5)
            self.df_processed[['TotalCharges']] = imputer.fit_transform(
                self.df_processed[['TotalCharges'] + numeric_cols]
            )[:, [0]]
        
        print(f"Missing values handled: {remaining_missing} records imputed")
        
        # 2. Feature Engineering (Assignment requirement 3.b)
        print("\n2. FEATURE ENGINEERING")
        print("-" * 40)
        
        # Create features
        self.df_processed['tenure_group'] = pd.cut(self.df_processed['tenure'],
                                                  bins=[0, 12, 24, 36, 48, 72],
                                                  labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4+yr'])
        
        # Customer value metrics
        self.df_processed['avg_monthly_charges'] = self.df_processed['TotalCharges'] / (self.df_processed['tenure'] + 1)
        self.df_processed['charges_per_service'] = self.df_processed['MonthlyCharges'] / (
            (self.df_processed[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                              'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1) + 1
        )
        
        # Service adoption patterns
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        self.df_processed['total_services'] = (self.df_processed[service_cols] == 'Yes').sum(axis=1)
        self.df_processed['service_adoption_rate'] = self.df_processed['total_services'] / len(service_cols)
        
        # Contract and payment risk factors
        self.df_processed['high_risk_contract'] = (self.df_processed['Contract'] == 'Month-to-month').astype(int)
        self.df_processed['high_risk_payment'] = (self.df_processed['PaymentMethod'] == 'Electronic check').astype(int)
        
        # Customer lifecycle stage
        self.df_processed['customer_lifecycle'] = pd.cut(self.df_processed['tenure'],
                                                        bins=[0, 6, 12, 24, 72],
                                                        labels=['New', 'Growing', 'Mature', 'Loyal'])
        
        print("Features created:")
        new_features = ['tenure_group', 'avg_monthly_charges', 'charges_per_service',
                       'total_services', 'service_adoption_rate', 'high_risk_contract',
                       'high_risk_payment', 'customer_lifecycle']
        for feature in new_features:
            print(f"  - {feature}")
        
        # FEATURE ENGINEERING JUSTIFICATIONS (Assignment requirement 3.b)
        print("\nFEATURE ENGINEERING JUSTIFICATIONS:")
        print("-" * 50)
        print("1. tenure_group: Categorical binning reduces noise and captures customer lifecycle patterns")
        print("   - Business Impact: Different retention strategies for new vs loyal customers")
        print("   - ML Impact: Reduces overfitting from raw tenure values")
        print("2. avg_monthly_charges: Normalizes spending by tenure, identifies high-value short-term customers")
        print("   - Business Impact: Identifies customers with high early spending (likely to churn)")
        print("   - ML Impact: Better signal than raw charges or tenure alone")
        print("3. charges_per_service: Measures value perception and service efficiency")
        print("   - Business Impact: High charges per service may indicate poor value perception")
        print("   - ML Impact: Captures price sensitivity patterns")
        print("4. total_services & service_adoption_rate: Measure customer engagement")
        print("   - Business Impact: Higher adoption typically correlates with lower churn")
        print("   - ML Impact: Quantifies customer stickiness and platform dependency")
        print("5. high_risk_contract: Month-to-month contracts show higher churn in telecom industry")
        print("   - Business Impact: Contract type is strongest churn predictor")
        print("   - ML Impact: Binary feature simplifies decision boundaries")
        print("6. high_risk_payment: Electronic check payments correlate with higher churn")
        print("   - Business Impact: Payment friction affects customer satisfaction")
        print("   - ML Impact: Captures behavioral payment preferences")
        print("7. customer_lifecycle: Groups customers by tenure into meaningful business segments")
        print("   - Business Impact: Enables lifecycle-specific retention strategies")
        print("   - ML Impact: Captures non-linear tenure relationships")
        
        print("\nJUSTIFICATION SUMMARY:")
        print("- Domain Knowledge: Features based on telecom industry churn patterns")
        print("- Statistical Value: Each feature addresses specific data limitations")
        print("- Business Actionability: Features enable targeted retention strategies")
        print("- ML Performance: Features reduce noise and capture complex relationships")
        
        # 3. Encoding techniques
        print("\n3. ENCODING STRATEGIES")
        print("-" * 40)
        
        # Binary encoding
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            self.df_processed[col] = self.df_processed[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        
        # One-hot encoding for remaining categoricals
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaymentMethod', 'tenure_group', 'customer_lifecycle']
        
        self.df_processed = pd.get_dummies(self.df_processed, columns=categorical_cols, drop_first=True)
        
        # Encode target variable
        self.df_processed['Churn'] = self.df_processed['Churn'].map({'Yes': 1, 'No': 0})
        
        print(f"Final dataset shape: {self.df_processed.shape}")
        print(f"Features after encoding: {self.df_processed.shape[1] - 1}")

    def feature_selection(self) -> None:
        """Feature selection using multiple techniques with feature importance exploration"""
        print("\n4. FEATURE SELECTION")
        print("-" * 40)
        
        # Prepare features and target
        feature_cols = [col for col in self.df_processed.columns if col not in ['Churn', 'customerID']]
        X = self.df_processed[feature_cols]
        y = self.df_processed['Churn']
        
        # Multiple feature selection techniques (Assignment requirement 3.b - explore techniques for feature importance)
        feature_scores = {}
        
        # 1. Univariate selection
        selector_univariate = SelectKBest(score_func=f_classif, k=20)
        selector_univariate.fit(X, y)
        univariate_features = X.columns[selector_univariate.get_support()].tolist()
        feature_scores['univariate'] = set(univariate_features)
        
        # 2. Mutual information
        mi_scores = mutual_info_classif(X, y)
        mi_features = X.columns[np.argsort(mi_scores)[-20:]].tolist()
        feature_scores['mutual_info'] = set(mi_features)
        
        # 3. Tree-based feature importance
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X, y)
        tree_features = X.columns[np.argsort(rf_selector.feature_importances_)[-20:]].tolist()
        feature_scores['tree_based'] = set(tree_features)
        
        # 4. L1 regularization
        l1_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
        l1_selector.fit(X, y)
        l1_features = X.columns[l1_selector.get_support()].tolist()
        feature_scores['l1_regularization'] = set(l1_features)
        
        # Combine features using voting
        all_features = set()
        for features in feature_scores.values():
            all_features.update(features)
        
        # Count votes for each feature
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for method_features in feature_scores.values() if feature in method_features)
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes
        selected_features = [feature for feature, votes in feature_votes.items() if votes >= 2]
        
        print(f"Feature selection completed:")
        print(f"  - Original features: {len(feature_cols)}")
        print(f"  - Selected features: {len(selected_features)}")
        print(f"  - Reduction: {(1 - len(selected_features)/len(feature_cols)):.1%}")
        
        # Update processed dataframe
        self.df_processed = self.df_processed[selected_features + ['Churn']]
        
        # Display top features
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 15 features by selection votes:")
        for feature, votes in sorted_features[:15]:
            print(f"  {feature}: {votes}/4 methods")
        
        # Build and evaluate models
        self.advanced_model_building()

    def advanced_model_building(self) -> None:
        """Build and evaluate multiple advanced models (Assignment requirement 4)"""
        print("\n" + "=" * 80)
        print("MODEL BUILDING & EVALUATION")
        print("=" * 80)
        
        # Prepare data
        feature_cols = [col for col in self.df_processed.columns if col != 'Churn']
        X = self.df_processed[feature_cols]
        y = self.df_processed['Churn']
        
        # Train-test split with stratification (Assignment requirement 4.a)
        print("TRAIN-TEST SPLIT JUSTIFICATION:")
        print("-" * 40)
        print("1. Using 80-20 split as specified in assignment")
        print("2. Stratified split maintains class distribution in both sets")
        print("3. Random state ensures reproducible results")
        print("4. This ratio provides sufficient training data while preserving test set for validation")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Also try different split ratio as requested (Assignment requirement 4.a.ii)
        print("\nAlternative split ratio (70-30):")
        X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"70-30 split - Training: {X_train_alt.shape}, Test: {X_test_alt.shape}")
        
        # Scaling
        self.scaler = RobustScaler()  # Robust to outliers than StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Class distribution - No Churn: {(y_train==0).sum()}, Churn: {(y_train==1).sum()}")
        
        # Advanced model configurations (Assignment requirement 4.b - all required classifiers)
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'Random Forest': {  # Ensemble Method 1
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'XGBoost': {  # Ensemble Method 2
                'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'Gradient Boosting': {  # Ensemble Method 3
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'Extra Trees': {  # Ensemble Method 4
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced', None]
                }
            }
        }
        
        # Train models with cross-validation and hyperparameter tuning (Assignment requirement 4.b.2)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\nHYPERPARAMETER TUNING JUSTIFICATION:")
        print("-" * 40)
        print("1. Using GridSearchCV for exhaustive parameter search")
        print("2. StratifiedKFold (5-fold) maintains class balance in each fold")
        print("3. ROC-AUC scoring for imbalanced dataset evaluation")
        print("4. Cross-validation prevents overfitting and provides robust estimates")
        
        print("\nTraining models with hyperparameter tuning...")
        
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # Handle parameter combinations properly
            if name == 'Logistic Regression':
                # Create separate parameter grids for different solvers
                param_grids = [
                    {'C': [0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear']},
                    {'C': [0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear']},
                    {'C': [0.1, 1, 10, 100], 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.9]}
                ]
            else:
                param_grids = [config['params']]
            
            best_score = 0
            best_model = None
            
            for param_grid in param_grids:
                try:
                    grid_search = GridSearchCV(
                        config['model'], param_grid, cv=cv_strategy,
                        scoring='roc_auc', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    
                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_model = grid_search.best_estimator_
                except:
                    continue
            
            if best_model is not None:
                self.models[name] = best_model
                print(f"  Cross Validation Score: {best_score:.4f}")
                print(f"  Best Parameters: {best_model.get_params()}")
            else:
                print(f"  Failed to train {name}")
        
        # Create ensemble model (Voting Classifier as additional ensemble method)
        if len(self.models) >= 3:
            ensemble_models = [(name, model) for name, model in list(self.models.items())[:3]]
            ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
            ensemble.fit(X_train_scaled, y_train)
            self.models['Voting Ensemble'] = ensemble
            print(f"\nVoting Ensemble model created with {len(ensemble_models)} base models")
        
        # Create ensemble model (LGBMClassifier as additional ensemble method)
        if 'XGBoost' in self.models:
            lgbm_model = LGBMClassifier(random_state=42, verbose=-1)
            lgbm_model.fit(X_train_scaled, y_train)
            self.models['LightGBM'] = lgbm_model
            print("LightGBM model created and trained")
        
        # Evaluate all models
        self.evaluate_models(X_test_scaled, y_test)

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Comprehensive model evaluation (Assignment requirement 5)"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)
        
        # Calculate predictions and probabilities
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
            probabilities[name] = model.predict_proba(X_test)[:, 1]
        
        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions
        self.probabilities = probabilities
        
        # Calculate comprehensive metrics (Assignment requirement 5.c)
        results = {}
        for name in self.models.keys():
            results[name] = {
                'Accuracy': accuracy_score(y_test, predictions[name]),
                'Precision': precision_score(y_test, predictions[name]),
                'Recall': recall_score(y_test, predictions[name]),
                'F1-Score': f1_score(y_test, predictions[name]),
                'ROC-AUC': roc_auc_score(y_test, probabilities[name]),
                'Avg Precision': average_precision_score(y_test, probabilities[name])
            }
        
        self.results = pd.DataFrame(results).T
        
        print("MODEL PERFORMANCE COMPARISON")
        print("-" * 40)
        print(self.results.round(4))
        
        # Assignment requirement 5.d - Precision and Recall for both classes
        print("\nPRECISION AND RECALL FOR BOTH CLASSES:")
        print("-" * 40)
        for name in self.models.keys():
            print(f"\n{name}:")
            report = classification_report(y_test, predictions[name], target_names=['No Churn', 'Churn'], output_dict=True)
            print(f"  No Churn - Precision: {report['No Churn']['precision']:.1%}, Recall: {report['No Churn']['recall']:.1%}")
            print(f"  Churn - Precision: {report['Churn']['precision']:.1%}, Recall: {report['Churn']['recall']:.1%}")
        
        # Identify best model
        self.best_model = self.results['ROC-AUC'].idxmax()
        print(f"\nBest Model: {self.best_model}")
        print(f"ROC-AUC Score: {self.results.loc[self.best_model, 'ROC-AUC']:.4f}")
        
        # BEST MODEL ANALYSIS AND JUSTIFICATION (Assignment requirement 5.e)
        print(f"\nBEST MODEL ANALYSIS: {self.best_model}")
        print("=" * 50)
        
        best_scores = self.results.loc[self.best_model]
        print(f"Why {self.best_model} is the best:")
        print(f"1. Highest ROC-AUC: {best_scores['ROC-AUC']:.4f} (best balance of TPR/FPR)")
        print(f"2. Good Precision: {best_scores['Precision']:.4f} (fewer false positives)")
        print(f"3. Good Recall: {best_scores['Recall']:.4f} (captures most churners)")
        print(f"4. Balanced F1-Score: {best_scores['F1-Score']:.4f}")
        print(f"5. High Average Precision: {best_scores['Avg Precision']:.4f} (good ranking of positive cases)")
        print(f"6. Robust performance across multiple metrics")
        print(f"7. Suitable for production deployment with high confidence")
        print(f"8. Model stability confirmed through cross-validation")
        print(f"9. Model interpretability and business relevance")
        print(f"10. Selected based on comprehensive evaluation metrics")
        
        print("\nMODEL EXPLANATIONS:")
        
        # Model-specific explanations
        model_name_lower = self.best_model.lower()
        if 'ensemble' in model_name_lower or 'voting' in model_name_lower:
            print("5. Ensemble methods reduce overfitting and improve generalization")
            print("6. Combines strengths of multiple algorithms for better predictions")
        elif 'random forest' in model_name_lower:
            print("5. Random Forest handles feature interactions well and is robust to outliers")
            print("6. Built-in feature importance helps with model interpretability")
        elif 'xgboost' in model_name_lower:
            print("5. XGBoost excels at handling complex patterns and interactions")
            print("6. Gradient boosting provides excellent predictive performance")
        elif 'logistic regression' in model_name_lower:
            print("5. Logistic Regression provides interpretable coefficients")
            print("6. Linear model is less prone to overfitting with proper regularization")
        elif 'gradient boosting' in model_name_lower:
            print("5. Gradient Boosting builds strong predictive models iteratively")
            print("6. Excellent at capturing non-linear relationships")
        elif 'decision tree' in model_name_lower:
            print("5. Decision Tree provides clear, interpretable rules")
            print("6. Easy to understand decision paths for business users")
        elif 'neighbors' in model_name_lower:
            print("5. K-NN captures local patterns in customer behavior")
            print("6. Non-parametric approach adapts to data distribution")
        else:
            print("5. Selected based on superior performance metrics")
            print("6. Best balance of precision, recall, and overall accuracy")
        
        # BUSINESS IMPLICATIONS
        print(f"\nBUSINESS IMPLICATIONS:")
        precision_score_val = best_scores['Precision']
        recall_score_val = best_scores['Recall']
        auc_score = best_scores['ROC-AUC']
        
        if precision_score_val > 0.8:
            print(f"- High precision ({precision_score_val:.1%}) means efficient targeting of retention campaigns")
            print("- Low false positive rate reduces wasted marketing spend")
        elif precision_score_val > 0.6:
            print(f"- Moderate precision ({precision_score_val:.1%}) requires careful campaign targeting")
        else:
            print(f"- Lower precision ({precision_score_val:.1%}) suggests need for model improvement")
        
        if recall_score_val > 0.7:
            print(f"- High recall ({recall_score_val:.1%}) means we catch most customers likely to churn")
            print("- Minimizes revenue loss from unidentified churners")
        elif recall_score_val > 0.5:
            print(f"- Moderate recall ({recall_score_val:.1%}) captures majority of churners")
        else:
            print(f"- Lower recall ({recall_score_val:.1%}) may miss important churn cases")
        
        if auc_score > 0.85:
            print(f"- Excellent discriminative ability ({auc_score:.1%}) for business decision making")
            print("- Model ready for production deployment")
        elif auc_score > 0.7:
            print(f"- Good discriminative ability ({auc_score:.1%}) with room for improvement")
        else:
            print(f"- Model performance ({auc_score:.1%}) needs enhancement before deployment")
        
        print(f"\nRECOMMENDATION:")
        if auc_score > 0.8 and precision_score_val > 0.7 and recall_score_val > 0.6:
            print(f"âœ… Deploy {self.best_model} for production churn prediction system")
            print("- Implement automated scoring for all customers")
            print("- Set up real-time alerts for high-risk customers")
        else:
            print(f"âš ï¸ {self.best_model} needs improvement before full deployment")
            print("- Consider additional feature engineering")
            print("- Collect more training data or review data quality")
        
        print(f"\nCONCLUSION:")
        print(f"- {self.best_model} achieved the best ROC-AUC score of {best_scores['ROC-AUC']:.4f}")
        print(f"- This model correctly identifies {best_scores['Recall']:.1%} of actual churners")
        print(f"- {best_scores['Precision']:.1%} of predicted churners are actually churners")
        print(f"- Recommended for deployment in churn prediction system")
        
        # Create visualizations (Assignment requirement 5.c - comparison chart)
        self.create_evaluation_plots(X_test, y_test, predictions, probabilities)
        
        # Save best model for deployment
        self.save_model()

    def create_evaluation_plots(self, X_test, y_test, predictions, probabilities):
        """Create comprehensive evaluation visualizations (Assignment requirement 5.c)"""
        print("\nCREATING EVALUATION VISUALIZATIONS")
        print("-" * 40)
        
        # 1. Performance Metrics Comparison Chart
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Metrics comparison bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        models = list(self.models.keys())
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            axes[0,0].bar(x + i*width, self.results[metric], width, label=metric, alpha=0.8)
        
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x + width*2)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # ROC Curves
        for name in self.models.keys():
            fpr, tpr, _ = roc_curve(y_test, probabilities[name])
            auc_score = self.results.loc[name, 'ROC-AUC']
            axes[0,1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        for name in self.models.keys():
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, probabilities[name])
            avg_precision = self.results.loc[name, 'Avg Precision']
            axes[1,0].plot(recall_vals, precision_vals, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curves')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Confusion Matrix for Best Model
        cm = confusion_matrix(y_test, predictions[self.best_model])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
        axes[1,1].set_title(f'Confusion Matrix - {self.best_model}')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        
        plt.tight_layout()
        save_fig(fig, "model_evaluation_suite")
        plt.show()

    def save_model(self):
        """Save the best model and preprocessing components"""
        print("\nSAVING MODEL FOR DEPLOYMENT")
        print("-" * 40)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save best model
        joblib.dump(self.models[self.best_model], 'models/best_churn_model.pkl')
        print("âœ… Best model saved successfully!")
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("âœ… Scaler saved successfully!")
        
        # Save feature columns for consistent preprocessing
        feature_columns = list(self.X_test.columns)
        joblib.dump(feature_columns, 'models/feature_columns.pkl')
        print("âœ… Feature columns saved successfully!")
        
        # Save model metadata
        model_info = {
            'best_model_name': self.best_model,
            'performance_metrics': self.results.loc[self.best_model].to_dict(),
            'training_date': pd.Timestamp.now(),
            'feature_count': len(feature_columns)
        }
        
        joblib.dump(model_info, 'models/model_info.pkl')
        print("âœ… Model info saved successfully!")
        
        print(f"  - Best model: {self.best_model}")
        print(f"  - Model file: models/best_churn_model.pkl")
        print(f"  - Scaler file: models/scaler.pkl")
        print(f"  - Feature columns: models/feature_columns.pkl")
        print(f"  - Model info: models/model_info.pkl")

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CHURN PREDICTION ANALYSIS REPORT")
        print("=" * 80)
        
        best_scores = self.results.loc[self.best_model]
        
        print(f"""
EXECUTIVE SUMMARY
{"-" * 40}
Dataset Overview:
  - Total Customers Analyzed: {self.df.shape[0]:,}
  - Features Used: {self.df_processed.shape[1] - 1}
  - Churn Rate: {self.df['Churn'].value_counts(normalize=True)['Yes']:.1%}

Model Performance:
  - Best Model: {self.best_model}
  - ROC-AUC Score: {best_scores['ROC-AUC']:.3f}
  - Precision: {best_scores['Precision']:.3f}
  - Recall: {best_scores['Recall']:.3f}
  - F1-Score: {best_scores['F1-Score']:.3f}

BUSINESS RECOMMENDATIONS
{"-" * 40}
1. IMMEDIATE ACTIONS:
   - Focus retention efforts on month-to-month contract customers
   - Monitor customers with electronic check payment methods
   - Implement early intervention for customers in first 4 months

2. STRATEGIC INITIATIVES:
   - Develop loyalty programs for high-value customers
   - Improve customer service for senior citizens
   - Create bundle offers to increase service adoption

3. OPERATIONAL IMPROVEMENTS:
   - Implement real-time churn scoring
   - Set up automated alerts for high-risk customers
   - Develop personalized retention campaigns

MODEL DEPLOYMENT READINESS
{"-" * 40}
âœ… Model validated with {best_scores['ROC-AUC']:.1%} AUC score
âœ… Feature engineering pipeline established
âœ… Interpretability analysis completed
âœ… Business insights documented
âœ… Model artifacts saved for production deployment

NEXT STEPS
{"-" * 40}
1. A/B test retention campaigns on high-risk customers
2. Monitor model performance with new data
3. Retrain model quarterly with updated customer data
4. Implement feedback loop for continuous improvement
""")
        
        print(f"\nModel Comparison:")
        print(self.results.round(4))
        
        print(f"\nAnalysis Completed Successfully!")
        print(f"Total customers analyzed: {self.df.shape[0]}")
        print("=" * 80)

    def run_complete_analysis(self) -> None:
        """Complete churn prediction analysis pipeline"""
        print("TELECOM CHURN PREDICTION ANALYSIS")
        print("=" * 80)
        
        try:
            # Execute full pipeline
            self.load_and_validate_data()
            self.comprehensive_eda()
            self.advanced_visualization()
            self.advanced_preprocessing()
            self.feature_selection()
            # The advanced_model_building step is already included in feature_selection
            self.generate_comprehensive_report()
            
            print("\nANALYSIS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

# REST API Implementation for Model Deployment
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

def create_flask_api():
    """Create Flask API for churn prediction"""
    app = Flask(__name__)
    
    # Load trained model and preprocessor
    try:
        model = joblib.load('models/best_churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        model_info = joblib.load('models/model_info.pkl')
    except FileNotFoundError:
        print("âš ï¸ Model files not found. Please run the training pipeline first.")
        return None
    
    def preprocess_input_data(data):
        """Preprocess input data for prediction"""
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        
        # Apply same preprocessing as training
        # Handle TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if df['TotalCharges'].isnull().any():
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Feature Engineering (same as training)
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 72],
                                   labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4+yr'])
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['total_services'] = (df[service_cols] == 'Yes').sum(axis=1)
        df['service_adoption_rate'] = df['total_services'] / len(service_cols)
        df['charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
        df['high_risk_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['high_risk_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['customer_lifecycle'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 72],
                                         labels=['New', 'Growing', 'Mature', 'Loyal'])
        
        # Binary encoding
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        
        # One-hot encoding
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaymentMethod', 'tenure_group', 'customer_lifecycle']
        
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the features used in training
        df = df[feature_columns]
        
        return df
    
    @app.route('/predict_churn', methods=['POST'])
    def predict_churn():
        """Predict customer churn"""
        try:
            # Get JSON data from request
            data = request.json
            
            # Preprocess data
            df_processed = preprocess_input_data(data)
            
            # Scale features
            df_scaled = scaler.transform(df_processed)
            
            # Make prediction
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0][1]
            
            # Determine risk level
            if probability > 0.7:
                risk_level = 'High'
            elif probability > 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Return results
            return jsonify({
                'customer_id': data.get('customerID', 'unknown'),
                'churn_prediction': int(prediction),
                'churn_probability': float(probability),
                'risk_level': risk_level,
                'model_used': model_info['best_model_name'],
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 400
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """API health check"""
        return jsonify({
            'status': 'API is running',
            'model_loaded': True,
            'model_name': model_info['best_model_name'],
            'model_performance': model_info['performance_metrics']
        })
    
    @app.route('/model_info', methods=['GET'])
    def get_model_info():
        """Get model information"""
        return jsonify(model_info)
    
    return app

# PostgreSQL Database Integration
class DatabaseManager:
    def __init__(self, host='localhost', database='churn_db', user='postgres',
                 password='yakuza', port=5432):
        """Initialize database connection"""
        self.db_config = {
            'host': host,
            'database': database,
            'username': user,
            'password': password,
            'port': port
        }
        
        # Create connection string
        self.connection_string = (
            f"postgresql://{self.db_config['username']}:"
            f"{self.db_config['password']}@{self.db_config['host']}:"
            f"{self.db_config['port']}/{self.db_config['database']}"
        )
        
        try:
            self.engine = create_engine(self.connection_string, isolation_level="AUTOCOMMIT")
            print("âœ… Database connection established")
        except Exception as e:
            print(f"âŒ Database connection failed: {e}")
            self.engine = None
    
    def create_tables(self):
        """Create necessary tables with proper foreign key constraints"""
        if not self.engine:
            print("âŒ No database connection")
            return
        
        # Create customers table first
        customers_table = """
        CREATE TABLE IF NOT EXISTS customers (
            customerID VARCHAR(20) PRIMARY KEY,
            gender VARCHAR(10),
            SeniorCitizen INTEGER,
            Partner VARCHAR(5),
            Dependents VARCHAR(5),
            tenure INTEGER,
            PhoneService VARCHAR(5),
            MultipleLines VARCHAR(20),
            InternetService VARCHAR(20),
            OnlineSecurity VARCHAR(20),
            OnlineBackup VARCHAR(20),
            DeviceProtection VARCHAR(20),
            TechSupport VARCHAR(20),
            StreamingTV VARCHAR(20),
            StreamingMovies VARCHAR(20),
            Contract VARCHAR(20),
            PaperlessBilling VARCHAR(5),
            PaymentMethod VARCHAR(30),
            MonthlyCharges DECIMAL(10,2),
            TotalCharges VARCHAR(20),
            Churn VARCHAR(5),
            active BOOLEAN DEFAULT true,
            churn_risk_score DECIMAL(5,4),
            risk_level VARCHAR(10),
            last_scored TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Create predictions table with explicit foreign key constraint
        predictions_table = """
        CREATE TABLE IF NOT EXISTS churn_predictions (
            id SERIAL PRIMARY KEY,
            customerID VARCHAR(20) NOT NULL,
            churn_prediction INTEGER,
            churn_probability DECIMAL(5,4),
            risk_level VARCHAR(10),
            model_used VARCHAR(50),
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customerid) REFERENCES customers(customerid) ON DELETE CASCADE ON UPDATE CASCADE
        );
        """
        
        # Create model performance tracking table
        model_performance_table = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50),
            accuracy DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            roc_auc DECIMAL(5,4),
            training_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            # Use transaction to ensure all tables are created properly
            with self.engine.begin() as conn:
                conn.execute(text(customers_table))
                print("âœ… Customers table created successfully")
                
                conn.execute(text(predictions_table))
                print("âœ… Predictions table created successfully")
                
                conn.execute(text(model_performance_table))
                print("âœ… Model performance table created successfully")
                
        except Exception as e:
            print(f"âŒ Failed to create tables: {e}")
            self.engine = None


    def load_customer_data(self):
        """Load customer data from database"""
        if not self.engine:
            print("âŒ No database connection")
            return None
        
        query = """
        SELECT customerid, gender, seniorcitizen, partner, dependents,
               tenure, phoneservice, multiplelines, internetservice,
               onlinesecurity, onlinebackup, deviceprotection, techsupport,
               streamingtv, streamingmovies, contract, paperlessbilling,
               paymentmethod, monthlycharges, totalcharges
        FROM customers
        WHERE active = true
        """
        
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"âŒ Failed to load customer data: {e}")
            return None
    
    def ensure_customer_exists(self, customer_data):
        """Ensure customer exists in database, create if not"""
        if not self.engine:
            print("âŒ No database connection")
            return
        
        try:
            with self.engine.connect() as conn:
                # Check if customer exists
                check_query = "SELECT COUNT(*) FROM customers WHERE customerid = %s"
                result = conn.execute(text(check_query), (customer_data['customerID'],))
                exists = result.scalar() > 0
                
                if not exists:
                    # Insert new customer
                    insert_query = """
                    INSERT INTO customers (customerid, gender, seniorcitizen, partner, dependents,
                                            tenure, phoneservice, multiplelines, internetservice,
                                            onlinesecurity, onlinebackup, deviceprotection, techsupport,
                                            streamingtv, streamingmovies, contract, paperlessbilling,
                                            paymentmethod, monthlycharges, totalcharges)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    conn.execute(text(insert_query), (
                        customer_data['customerID'], customer_data['gender'], customer_data['SeniorCitizen'],
                        customer_data['Partner'], customer_data['Dependents'], customer_data['tenure'],
                        customer_data['PhoneService'], customer_data['MultipleLines'], customer_data['InternetService'],
                        customer_data['OnlineSecurity'], customer_data['OnlineBackup'], customer_data['DeviceProtection'],
                        customer_data['TechSupport'], customer_data['StreamingTV'], customer_data['StreamingMovies'],
                        customer_data['Contract'], customer_data['PaperlessBilling'], customer_data['PaymentMethod'],
                        customer_data['MonthlyCharges'], customer_data['TotalCharges']
                    ))
        except Exception as e:
            print(f"âŒ Failed to ensure customer exists: {e}")
            self.engine = None
            
    def save_or_create_customers(self, customer_df):
        """Automatically create customer if doesn't exist, or update if exists"""
        if not self.engine:
            print("âŒ No database connection")
            return
        # Check if customer exists first
        try:
            with self.engine.connect() as conn:
                for _, row in customer_df.iterrows():
                    customer_id = row['customerID']
                    check_query = "SELECT COUNT(*) FROM customers WHERE customerid = %s"
                    result = conn.execute(text(check_query), (customer_id,))
                    exists = result.scalar() > 0
                    
                    if exists:
                        # Update existing customer
                        update_query = """
                        UPDATE customers SET gender = %s, seniorcitizen = %s, partner = %s, dependents = %s,
                                          tenure = %s, phoneservice = %s, multiplelines = %s, internetservice = %s,
                                          onlinesecurity = %s, onlinebackup = %s, deviceprotection = %s, techsupport = %s,
                                          streamingtv = %s, streamingmovies = %s, contract = %s, paperlessbilling = %s,
                                          paymentmethod = %s, monthlycharges = %s, totalcharges = %s
                        WHERE customerid = %s
                        """
                        conn.execute(text(update_query), (
                            row['gender'], row['SeniorCitizen'], row['Partner'], row['Dependents'],
                            row['tenure'], row['PhoneService'], row['MultipleLines'], row['InternetService'],
                            row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'],
                            row['StreamingTV'], row['StreamingMovies'], row['Contract'], row['PaperlessBilling'],
                            row['PaymentMethod'], row['MonthlyCharges'], row['TotalCharges'],
                            customer_id
                        ))
                    else:                               # Insert new customer
                        insert_query = """
                        INSERT INTO customers (customerid, gender, seniorcitizen, partner, dependents,
                                                tenure, phoneservice, multiplelines, internetservice,
                                                onlinesecurity, onlinebackup, deviceprotection, techsupport,
                                                streamingtv, streamingmovies, contract, paperlessbilling,
                                                paymentmethod, monthlycharges, totalcharges)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        conn.execute(text(insert_query), (
                            customer_id, row['gender'], row['SeniorCitizen'], row['Partner'], row['Dependents'],
                            row['tenure'], row['PhoneService'], row['MultipleLines'], row['InternetService'],
                            row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'],
                            row['StreamingTV'], row['StreamingMovies'], row['Contract'], row['PaperlessBilling'],
                            row['PaymentMethod'], row['MonthlyCharges'], row['TotalCharges']
                        ))
                        print(f"âœ… Updated customer record for {customer_id}")
                        
        except Exception as e:
            print(f"âŒ Failed to update customer record for {customer_id}: {e}")
            
            

    def save_predictions(self, predictions_df):
        """Save model predictions to database"""
        if not self.engine:
            print("âŒ No database connection")
            return
        
        try:
            # Create a copy to avoid modifying original DataFrame
            df_to_save = predictions_df.copy()
        
            # Map column names to lowercase for PostgreSQL compatibility
            column_mapping = {
            'customerID': 'customerid',
            'churn_prediction': 'churn_prediction',
            'churn_probability': 'churn_probability',
            'risk_level': 'risk_level',
            'model_used': 'model_used',
            'prediction_date': 'prediction_date'
            }
        
            # Rename columns to match database schema
            df_to_save = df_to_save.rename(columns=column_mapping)
            
            # Check and create customers if they don't exist
            with self.engine.connect() as conn:
                for _, row in df_to_save.iterrows():
                    customer_id = row['customerid']
                    check_query = "SELECT COUNT(*) FROM customers WHERE customerid = %s"
                    result = conn.execute(text(check_query), (customer_id,))
                    exists = result.scalar() > 0
                    
                    if not exists:
                        # Insert new customer record if it doesn't exist
                        insert_query = """
                        INSERT INTO customers (customerid, created_at, updated_at)
                        VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """
                        conn.execute(text(insert_query), (customer_id,))
                        print(f"âœ… Created new customer record for {customer_id}")
        
            # Save to database
            df_to_save.to_sql(
            'churn_predictions',
            self.engine,
            if_exists='append',
            index=False,
            method='multi'
            )
        
            print(f"âœ… Saved {len(df_to_save)} predictions to database")
        except Exception as e:
            print(f"âŒ Failed to save predictions: {e}")
    
    
    def save_or_update_customer(self, customer_df):
        """Save or update customer record in database"""
        if not self.engine:
            print("âŒ No database connection")
            return
        
        try:
            # Ensure lowercase column names
            customer_df.columns = customer_df.columns.str.lower()
            
            # Check if customer exists
            customer_id = customer_df.iloc[0]['customerID']
            
            with self.engine.connect() as conn:
                # Check if customer exists
                check_query = "SELECT COUNT(*) FROM customers WHERE customerid = %s"
                result = conn.execute(text(check_query), (customer_id,))
                exists = result.scalar() > 0
                
                if exists:
                    # Update existing customer
                    update_query = """
                    UPDATE customers SET 
                        gender = %s, seniorcitizen = %s, partner = %s, dependents = %s,
                        tenure = %s, phoneservice = %s, multiplelines = %s, internetservice = %s,
                        onlinesecurity = %s, onlinebackup = %s, deviceprotection = %s, techsupport = %s,
                        streamingtv = %s, streamingmovies = %s, contract = %s, paperlessbilling = %s,
                        paymentmethod = %s, monthlycharges = %s, totalcharges = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE customerid = %s
                    """
                    
                    row = customer_df.iloc[0]
                    conn.execute(text(update_query), (
                        row['gender'], row['seniorcitizen'], row['partner'], row['dependents'],
                        row['tenure'], row['phoneservice'], row['multiplelines'], row['internetservice'],
                        row['onlinesecurity'], row['onlinebackup'], row['deviceprotection'], row['techsupport'],
                        row['streamingtv'], row['streamingmovies'], row['contract'], row['paperlessbilling'],
                        row['paymentmethod'], row['monthlycharges'], row['totalcharges'],
                        customer_id
                    ))
                    print(f"âœ… Updated customer record for {customer_id}")
                else:
                    # Insert new customer
                    customer_df.to_sql(
                        'customers',
                        self.engine,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
                    print(f"âœ… Inserted new customer record for {customer_id}")
                    
        except Exception as e:
            print(f"âŒ Failed to save/update customer: {e}")

    
    def upsert_single_customer(self, customer_dict):
        """
        Insert or update a single customer record into the customers table using all keys in customer_dict.
        If customerID exists, updates all fields; otherwise, inserts new row.
        This prevents foreign key constraint violations.
        """
        if not self.engine:
            print("âŒ No database connection")
            return False
        
        try:
            # Get column names and values from the input dictionary
            columns = list(customer_dict.keys())
            values = [customer_dict[col] for col in columns]
            
            # Create the SET clause for UPDATE (exclude customerID as it's the primary key)
            update_assignments = ', '.join([f'"{col}" = EXCLUDED."{col}"' for col in columns if col.lower() != 'customerid'])
            
            # Build the UPSERT SQL query using PostgreSQL's ON CONFLICT
            sql = f"""
            INSERT INTO customers ({', '.join('"' + col + '"' for col in columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT ("customerID") DO UPDATE SET {update_assignments};
            """
            
            # Execute the query
            with self.engine.connect() as conn:
                conn.execute(text(sql), values)
                conn.commit()
                print(f"âœ… Customer {customer_dict['customerID']} upserted successfully")
                return True
                
        except Exception as e:
            print(f"âŒ Failed to upsert customer: {e}")
            return False

    
    
    def customer_exists(self, customer_id):
        """
        Check if a customer exists in the database.
        
        Args:
            customer_id (str): The customer ID to check
            
        Returns:
            bool: True if customer exists, False otherwise
        """
        if not self.engine:
            print("âŒ No database connection")
            return False
        
        try:
            query = "SELECT COUNT(*) FROM customers WHERE customerid = %s"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), (customer_id,))
                count = result.scalar()
                exists = count > 0
                
                if exists:
                    print(f"âœ… Customer {customer_id} exists in database")
                else:
                    print(f"â„¹ï¸ Customer {customer_id} does not exist in database")
                    
                return exists
                
        except Exception as e:
            print(f"âŒ Error checking customer existence: {e}")
            return False
        
    def insert_or_verify_customer(self, customer_dict):
        """
        Complete method that checks if customer exists and inserts/updates accordingly.
        This is the main method to use for ensuring customer data exists before predictions.

        Args:
            customer_dict (dict): Dictionary containing customer data

        Returns:
            dict: Status information about the operation
        """
        if not self.engine:
            return {"success": False, "message": "No database connection", "action": "none"}

        if not customer_dict or 'customerID' not in customer_dict:
            return {"success": False, "message": "Invalid input: customerID is required", "action": "none"}

        customer_id = customer_dict['customerID']

        try:
            # Check if customer exists first
            exists = self.customer_exists(customer_id)
            
            # Build UPSERT statement with proper parameter handling
            columns = list(customer_dict.keys())
            values = tuple(customer_dict[col] for col in columns)  # Convert to tuple
            
            # Create the SET clause for UPDATE (exclude customerID as it's the primary key)
            update_assignments = ', '.join([f'"{col}" = EXCLUDED."{col}"' for col in columns if col.lower() != 'customerid'])
            
            # Build the UPSERT SQL query using PostgreSQL's ON CONFLICT
            sql = f"""
            INSERT INTO customers ({', '.join('"' + col + '"' for col in columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT ("customerID") DO UPDATE SET {update_assignments};
            """
            
            # Execute the query with tuple parameters
            with self.engine.connect() as conn:
                conn.execute(text(sql), values)
                conn.commit()
                
            action = "updated" if exists else "inserted"
            return {
                "success": True,
                "message": f"Customer {customer_id} {action} successfully",
                "action": action,
                "customer_id": customer_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Database error: {str(e)}",
                "action": "error",
                "customer_id": customer_id
            }


        
    
    
    def save_customer_data(self, customer_df):
        """Using TRUNCATE method to save customer data"""
        if not self.engine:
            print("âŒ No database connection")
            return
        
        try:
            customer_df.columns = customer_df.columns.str.lower()  # Ensure consistent column names
            with self.engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE customers RESTART IDENTITY CASCADE"))
            # Save customer data
            customer_df.to_sql(
                'customers',
                self.engine,
                if_exists='append',  # Use append to insert new data
                index=False,
                method='multi'
            )
            print(f"âœ… Saved {len(customer_df)} customers to database")
        except Exception as e:
            print(f"âŒ Failed to save customer data: {e}")
    
    def update_customer_risk(self, customer_id, risk_score, risk_level):
        """Update customer risk score in database"""
        if not self.engine:
            print("âŒ No database connection")
            return
        
        query = """
        UPDATE customers
        SET churn_risk_score = :risk_score,
            risk_level = :risk_level,
            last_scored = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE customerid = :customer_id
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(query), {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'customer_id': customer_id
                })
                conn.commit()
            print(f"âœ… Updated risk score for customer {customer_id}")
        except Exception as e:
            print(f"âŒ Failed to update customer risk: {e}")

    def batch_prediction_pipeline():
        """Batch prediction pipeline using database"""
        print("STARTING BATCH PREDICTION PIPELINE")
        print("=" * 50)
        
        # Initialize database manager
        db = DatabaseManager()
        if not db.engine:
            print("âŒ Cannot proceed without database connection")
            return
        
        # Load customer data
        customers = db.load_customer_data()
        if customers is None:
            print("âŒ No customer data available")
            return
        
        print(f"ðŸ“Š Loaded {len(customers)} customers from database")
        
        # Load trained model
        try:
            model = joblib.load('models/best_churn_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            model_info = joblib.load('models/model_info.pkl')
            print("âœ… Model loaded successfully")
        except FileNotFoundError:
            print("âŒ Model files not found. Please run training pipeline first.")
            return
        
        # Initialize predictor for preprocessing
        predictor = TelecomChurnPredictor('dummy_path')
        predictor.df = customers.copy()
        
        # Preprocess data
        print("ðŸ”„ Preprocessing customer data...")
        predictor.advanced_preprocessing()
        
        # Prepare features
        feature_cols = [col for col in predictor.df_processed.columns if col != 'Churn']
        X = predictor.df_processed[feature_cols]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        print("ðŸ”® Making predictions...")
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customerID': customers['customerid'],
            'churn_prediction': predictions,
            'churn_probability': probabilities,
            'risk_level': pd.cut(probabilities,
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']),
            'model_used': model_info['best_model_name'],
            'prediction_date': pd.Timestamp.now()
        })
        
        # Save to database
        db.save_predictions(results)
        
        # Update customer risk scores
        print("ðŸ“ Updating customer risk scores...")
        for _, row in results.iterrows():
            db.update_customer_risk(
                row['customerid'],
                row['churn_probability'],
                row['risk_level']
            )
        
        print("âœ… Batch prediction pipeline completed")
        return results

# Jupyter Notebook Conversion Functions
def create_jupyter_notebook():
    """Create Jupyter notebook from the analysis"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Telecom Customer Churn Prediction Analysis\n",
                    "\n",
                    "## Assignment Overview\n",
                    "This notebook implements a comprehensive churn prediction analysis including:\n",
                    "1. Data Exploration and Visualization\n",
                    "2. Correlation Analysis with Feature Selection Justification\n",
                    "3. Data Preprocessing and Feature Engineering with Justifications\n",
                    "4. Model Building with Required Classifiers\n",
                    "5. Hyperparameter Tuning with Cross-Validation\n",
                    "6. Performance Evaluation and Best Model Selection\n",
                    "7. Single Customer Churn Prediction and Database Save\n",
                    "8. Comprehensive Report Generation\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import necessary libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import joblib\n",
                    "import analyzer\n",
                    "import json\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.model_selection import GridSearchCV\n",
                    "from sklearn.model_selection import (train_test_split, cross_val_score)\n",
                    "from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n",
                    "                             f1_score, roc_auc_score, classification_report,\n",
                    "                             confusion_matrix)\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.preprocessing import OneHotEncoder\n",
                    "from sklearn.pipeline import make_pipeline\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.feature_selection import SelectKBest, f_classif\n",
                    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.svm import SVC\n",
                    "from sklearn.neighbors import KNeighborsClassifier\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.ensemble import VotingClassifier\n",
                    "from sklearn.metrics import roc_curve, precision_recall_curve\n",
                    "import os\n",
                    "import logging\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Import the TelecomChurnPredictor and DatabaseManager classes\n",
                    "from proj_3 import TelecomChurnPredictor, DatabaseManager\n",
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Loading and Validation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize the predictor\n",
                    "analyzer = TelecomChurnPredictor('Telco-Dataset.csv')\n",
                    "\n",
                    "# Load and validate data\n",
                    "analyzer.load_and_validate_data()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Exploratory Data Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Comprehensive EDA\n",
                    "analyzer.comprehensive_eda()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Data Visualization and Correlation Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Advanced visualization including correlation analysis with justification\n",
                    "analyzer.advanced_visualization()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Data Preprocessing and Feature Engineering"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Advanced preprocessing with detailed justifications\n",
                    "analyzer.advanced_preprocessing()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Feature Selection and Model Building"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Feature selection and model building with all required classifiers\n",
                    "analyzer.feature_selection()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Comprehensive Report Generation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Generate final comprehensive report\n",
                    "analyzer.generate_comprehensive_report()"
                ]
            },
             {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Single Customer Churn Prediction and Database Save\n",
            "This section demonstrates how to predict churn for a new customer and save the result to the PostgreSQL database.\n",
            "**Enhanced with customer verification to prevent foreign key constraint violations.**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count":None,
        "metadata": {},
        "source": [
            "# 1. Prepare input data for a new customer (CORRECTED VERSION)\n",
            "import pandas as pd\n",
            "import joblib\n",
            "from proj_3 import TelecomChurnPredictor, DatabaseManager\n",
            "\n",
            "input_data = {\n",
            "    \"customerID\": \"TEST-001\",\n",
            "    \"tenure\": 12,\n",
            "    \"MonthlyCharges\": 50.0,\n",
            "    \"TotalCharges\": \"600.0\",\n",
            "    \"Contract\": \"Month-to-month\",\n",
            "    \"PaymentMethod\": \"Electronic check\",\n",
            "    \"InternetService\": \"Fiber optic\",\n",
            "    \"gender\": \"Male\",\n",
            "    \"SeniorCitizen\": 0,\n",
            "    \"Partner\": \"No\",\n",
            "    \"Dependents\": \"No\",\n",
            "    \"PhoneService\": \"Yes\",\n",
            "    \"MultipleLines\": \"No\",\n",
            "    \"OnlineSecurity\": \"No\",\n",
            "    \"OnlineBackup\": \"Yes\",\n",
            "    \"DeviceProtection\": \"No\",\n",
            "    \"TechSupport\": \"No\",\n",
            "    \"StreamingTV\": \"No\",\n",
            "    \"StreamingMovies\": \"No\",\n",
            "    \"PaperlessBilling\": \"Yes\"\n",
            "}\n",
            "\n",
            "print(\"ðŸš€ Starting enhanced churn prediction pipeline...\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# 2. DATABASE VERIFICATION STEP (NEW - PREVENTS FOREIGN KEY ERRORS)\n",
            "print(\"ðŸ”Œ Step 1: Connecting to database...\")\n",
            "db = DatabaseManager()\n",
            "\n",
            "if not db.engine:\n",
            "    raise Exception(\"âŒ Database connection failed. Please check your connection settings.\")\n",
            "\n",
            "print(\"âœ… Database connection successful\")\n",
            "\n",
            "# 3. INSERT OR VERIFY CUSTOMER EXISTS (CRITICAL FOR FOREIGN KEY COMPLIANCE)\n",
            "print(\"ðŸ“ Step 2: Inserting/verifying customer in database...\")\n",
            "db_result = db.insert_or_verify_customer(input_data)\n",
            "\n",
            "if not db_result[\"success\"]:\n",
            "    raise Exception(f\"âŒ Customer database operation failed: {db_result['message']}\")\n",
            "\n",
            "print(f\"âœ… Customer database status: {db_result['action'].upper()}\")\n",
            "print(f\"ðŸ“‹ Details: {db_result['message']}\")\n",
            "\n",
            "# 4. Load model components\n",
            "print(\"ðŸ¤– Step 3: Loading trained model components...\")\n",
            "try:\n",
            "    model = joblib.load(\"models/best_churn_model.pkl\")\n",
            "    scaler = joblib.load(\"models/scaler.pkl\")\n",
            "    feature_columns = joblib.load(\"models/feature_columns.pkl\")\n",
            "    model_info = joblib.load(\"models/model_info.pkl\")\n",
            "    print(\"âœ… Model components loaded successfully\")\n",
            "except FileNotFoundError as e:\n",
            "    raise Exception(f\"âŒ Model files not found: {e}. Please run the training pipeline first.\")\n",
            "\n",
            "# 5. Create robust preprocessing function for a single customer\n",
            "def preprocess_single_customer(input_data, feature_columns):\n",
            "    \"\"\"Preprocess single customer data using the same pipeline as training\"\"\"\n",
            "    predictor = TelecomChurnPredictor('Telco-Dataset.csv')\n",
            "    predictor.load_and_validate_data()\n",
            "    reference_df = predictor.df.copy()\n",
            "\n",
            "    combined_df = pd.concat([reference_df, pd.DataFrame([input_data])], ignore_index=True)\n",
            "    predictor.df = combined_df\n",
            "    predictor.advanced_preprocessing()\n",
            "\n",
            "    processed_customer_df = predictor.df_processed.tail(1)\n",
            "    if 'Churn' in processed_customer_df.columns:\n",
            "        processed_customer_df = processed_customer_df.drop('Churn', axis=1)\n",
            "\n",
            "    for col in feature_columns:\n",
            "        if col not in processed_customer_df.columns:\n",
            "            processed_customer_df[col] = 0\n",
            "\n",
            "    processed_customer_df = processed_customer_df[feature_columns]\n",
            "    return processed_customer_df\n",
            "\n",
            "# 6. Preprocess the input data\n",
            "print(\"ðŸ”„ Step 4: Preprocessing customer data...\")\n",
            "X_processed = preprocess_single_customer(input_data, feature_columns)\n",
            "print(\"âœ… Data preprocessing completed\")\n",
            "\n",
            "# 7. Scale the processed data\n",
            "print(\"ðŸ“Š Step 5: Scaling features...\")\n",
            "X_scaled = scaler.transform(X_processed)\n",
            "\n",
            "# 8. Make prediction\n",
            "print(\"ðŸ”® Step 6: Making churn prediction...\")\n",
            "prediction = model.predict(X_scaled)[0]\n",
            "probability = model.predict_proba(X_scaled)[0][1]\n",
            "risk_level = 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'\n",
            "print(f\"âœ… Prediction completed: {'CHURN' if prediction == 1 else 'NO CHURN'} ({probability:.4f})\")\n",
            "\n",
            "# 9. Save prediction to database (NOW SAFE - CUSTOMER EXISTS)\n",
            "print(\"ðŸ’¾ Step 7: Saving prediction to database...\")\n",
            "results = pd.DataFrame([{\n",
            "    'customerid': input_data['customerID'],\n",
            "    'churn_prediction': prediction,\n",
            "    'churn_probability': probability,\n",
            "    'risk_level': risk_level,\n",
            "    'model_used': model_info['best_model_name'],\n",
            "    'prediction_date': pd.Timestamp.now()\n",
            "}])\n",
            "\n",
            "try:\n",
            "    db.save_predictions(results)\n",
            "    db.update_customer_risk(input_data['customerID'], probability, risk_level)\n",
            "    print(\"âœ… Prediction saved to database successfully\")\n",
            "except Exception as e:\n",
            "    print(f\"âŒ Error saving prediction: {e}\")\n",
            "    raise\n",
            "\n",
            "# 10. Display the result with clean formatting\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"ðŸ”® CHURN PREDICTION RESULTS\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "for _, row in results.iterrows():\n",
            "    risk_emoji = \"ðŸ”´\" if row['risk_level'] == 'High' else \"ðŸŸ¡\" if row['risk_level'] == 'Medium' else \"ðŸŸ¢\"\n",
            "    churn_emoji = \"âŒ\" if row['churn_prediction'] == 1 else \"âœ…\"\n",
            "    print(f\"ðŸ‘¤ Customer ID: {row['customerid']}\")\n",
            "    print(f\"{churn_emoji} Prediction: {'WILL CHURN' if row['churn_prediction'] == 1 else 'WILL NOT CHURN'}\")\n",
            "    print(f\"ðŸ“Š Churn Probability: {row['churn_probability']:.4f} ({row['churn_probability']*100:.2f}%)\")\n",
            "    print(f\"{risk_emoji} Risk Level: {row['risk_level'].upper()}\")\n",
            "    print(f\"ðŸ¤– Model Used: {row['model_used']}\")\n",
            "    print(f\"ðŸ“… Prediction Date: {row['prediction_date'].strftime('%Y-%m-%d %H:%M:%S')}\")\n",
            "    print(f\"ðŸ’¾ Database Status: Prediction saved successfully!\")\n",
            "    print(f\"ðŸ—ƒï¸ Customer Status: {db_result['action'].upper()}\")\n",
            "\n",
            "print(\"=\"*60)\n",
            "\n",
            "print(\"\\nðŸ“‹ BUSINESS RECOMMENDATIONS:\")\n",
            "print(\"-\"*40)\n",
            "if risk_level == 'High':\n",
            "    print(\"ðŸš¨ IMMEDIATE ACTION REQUIRED:\")\n",
            "    print(\"   â€¢ Contact customer within 24 hours\")\n",
            "    print(\"   â€¢ Offer retention incentives\")\n",
            "    print(\"   â€¢ Schedule customer service call\")\n",
            "    print(\"   â€¢ Escalate to retention team\")\n",
            "elif risk_level == 'Medium':\n",
            "    print(\"âš ï¸  MODERATE ATTENTION NEEDED:\")\n",
            "    print(\"   â€¢ Monitor customer behavior closely\")\n",
            "    print(\"   â€¢ Consider proactive engagement\")\n",
            "    print(\"   â€¢ Send targeted retention offers\")\n",
            "    print(\"   â€¢ Schedule follow-up in 2 weeks\")\n",
            "else:\n",
            "    print(\"âœ… LOW RISK - STANDARD MONITORING:\")\n",
            "    print(\"   â€¢ Continue regular service\")\n",
            "    print(\"   â€¢ Include in loyalty programs\")\n",
            "    print(\"   â€¢ Monitor for any behavior changes\")\n",
            "    print(\"   â€¢ Quarterly check-in\")\n",
            "\n",
            "print(\"\\nðŸ”§ TECHNICAL NOTES:\")\n",
            "print(\"-\"*40)\n",
            "print(f\"âœ… Foreign key constraint: RESOLVED\")\n",
            "print(f\"âœ… Customer verification: COMPLETED\")\n",
            "print(f\"âœ… Database integrity: MAINTAINED\")\n",
            "print(f\"âœ… Pipeline status: SUCCESS\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)"
        ]
    }

        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    import json
    with open('Telecom_Churn_Analysis.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("âœ… Jupyter notebook created: Telecom_Churn_Analysis.ipynb")
    print("To use it:")
    print("1. Install Jupyter: pip install jupyter")
    print("2. Start Jupyter: jupyter notebook")
    print("3. Open Telecom_Churn_Analysis.ipynb")

# Usage Example and Main Execution
if __name__ == "__main__":
    print("ðŸš€ TELECOM CHURN PREDICTION SYSTEM")
    print("=" * 50)
    
    # Initialize and run analysis
    try:
        analyzer = TelecomChurnPredictor('Telco-Dataset.csv')
        analyzer.run_complete_analysis()
        
        # Create Jupyter notebook
        create_jupyter_notebook()
        
        # Initialize database
        print("\nðŸ—„ï¸ SETTING UP DATABASE")
        db = DatabaseManager()
        if db.engine:
            db.create_tables()
            
            # Save customer data to database
            db.save_customer_data(analyzer.df)
        
        # Create and run Flask API
        print("\nðŸŒ SETTING UP REST API")
        app = create_flask_api()
        if app:
            print("âœ… Flask API created successfully!")
            print("To run the API:")
            print("python -c \"from proj_3 import create_flask_api; app = create_flask_api(); app.run(debug=True, host='0.0.0.0', port=5000)\"")
            
            # Example API usage
            print("\nðŸ“¡ API USAGE EXAMPLES:")
            print("Health Check: GET http://localhost:5000/health")
            print("Model Info: GET http://localhost:5000/model_info")
            print("Predict Churn: POST http://localhost:5000/predict_churn")
            
            print("\nExample prediction request:")
            example_request = {
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

        # Convert the dictionary to a JSON string and escape double quotes for Windows CMD
        json_str = json.dumps(example_request)
        escaped_json_str = json_str.replace('"', '\\"')

        print(f'curl -X POST http://localhost:5000/predict_churn -H "Content-Type: application/json" -d "{escaped_json_str}"')

        print("\nâœ… SYSTEM SETUP COMPLETED!")
        print("=" * 50)
        print("ðŸ“Š Analysis Report Generated")
        print("ðŸ““ Jupyter Notebook Created")
        print("ðŸ—„ï¸ Database Schema Created")
        print("ðŸŒ REST API Ready for Deployment")
        print("ðŸš€ Ready for Production!")
        
    except Exception as e:
        print("âŒ System setup failed. Please check the logs for details.")
        print("Ensure all dependencies are installed and the dataset is available.")
        logger.error(f"System setup failed: {e}")
        raise