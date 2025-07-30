"""
Performance Optimization Module for Telecom Churn Prediction System

This module addresses key performance bottlenecks identified in the main codebase:
1. Memory optimization for large datasets
2. Parallel processing for model training
3. Caching for expensive computations
4. Database connection pooling
5. API response optimization
6. Bundle size reduction
"""

import pandas as pd
import numpy as np
import joblib
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import gc
import psutil
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_time = time.time()
        logger.info(f"Starting {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if self.start_time is None:
            return 0.0
        
        duration = time.time() - self.start_time
        self.metrics[operation] = duration
        logger.info(f"Completed {operation} in {duration:.2f} seconds")
        return duration
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

class OptimizedDataLoader:
    """Optimized data loading with memory management"""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.monitor = PerformanceMonitor()
    
    def load_data_optimized(self, file_path: str) -> pd.DataFrame:
        """Load data with memory optimization"""
        self.monitor.start_timer("data_loading")
        
        try:
            # Use chunked reading for large files
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            
            # Optimize data types to reduce memory usage
            df = self._optimize_dtypes(df)
            
            self.monitor.end_timer("data_loading")
            logger.info(f"Loaded dataset: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df

class ParallelModelTrainer:
    """Parallel model training with optimized hyperparameter search"""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.monitor = PerformanceMonitor()
    
    def train_models_parallel(self, X: pd.DataFrame, y: pd.Series, 
                            models_config: Dict) -> Dict:
        """Train models in parallel"""
        self.monitor.start_timer("parallel_model_training")
        
        # Prepare data for parallel processing
        train_data = (X, y)
        
        # Train models in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for name, config in models_config.items():
                future = executor.submit(self._train_single_model, 
                                      name, config, train_data)
                futures.append(future)
            
            # Collect results
            models = {}
            for future in futures:
                name, model = future.result()
                if model is not None:
                    models[name] = model
        
        self.monitor.end_timer("parallel_model_training")
        return models
    
    def _train_single_model(self, name: str, config: Dict, 
                           train_data: Tuple) -> Tuple[str, object]:
        """Train a single model (for parallel processing)"""
        try:
            X, y = train_data
            
            # Use RandomizedSearchCV instead of GridSearchCV for faster search
            from sklearn.model_selection import RandomizedSearchCV
            from scipy.stats import uniform, randint
            
            # Define parameter distributions for faster search
            param_distributions = self._get_param_distributions(name, config)
            
            model = RandomizedSearchCV(
                config['model'], param_distributions,
                n_iter=20,  # Reduced from exhaustive search
                cv=3,  # Reduced from 5-fold
                scoring='roc_auc',
                n_jobs=1,  # Single job per process
                random_state=42,
                verbose=0
            )
            
            model.fit(X, y)
            return name, model.best_estimator_
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            return name, None
    
    def _get_param_distributions(self, name: str, config: Dict) -> Dict:
        """Get parameter distributions for RandomizedSearchCV"""
        if name == 'Logistic Regression':
            return {
                'C': uniform(0.1, 10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif name == 'Random Forest':
            return {
                'n_estimators': randint(50, 200),
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif name == 'XGBoost':
            return {
                'n_estimators': randint(50, 150),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 8),
                'subsample': uniform(0.8, 0.2)
            }
        else:
            return config['params']

class CachedPreprocessor:
    """Cached preprocessing for repeated operations"""
    
    def __init__(self):
        self.cache = {}
        self.monitor = PerformanceMonitor()
    
    @lru_cache(maxsize=128)
    def preprocess_cached(self, data_hash: str, data: pd.DataFrame) -> pd.DataFrame:
        """Cached preprocessing operation"""
        self.monitor.start_timer("preprocessing")
        
        # Apply preprocessing steps
        df_processed = self._apply_preprocessing(data)
        
        self.monitor.end_timer("preprocessing")
        return df_processed
    
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps"""
        # Handle missing values
        df = df.copy()
        
        # Convert TotalCharges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Feature engineering
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 72],
                                   labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4+yr'])
        
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Service features
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['total_services'] = (df[service_cols] == 'Yes').sum(axis=1)
        df['service_adoption_rate'] = df['total_services'] / len(service_cols)
        
        # Binary encoding
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        
        # One-hot encoding
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                          'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                          'StreamingTV', 'StreamingMovies', 'Contract', 
                          'PaymentMethod', 'tenure_group']
        
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df

class OptimizedAPIServer:
    """Optimized Flask API with caching and connection pooling"""
    
    def __init__(self, model_path: str, scaler_path: str):
        self.model = None
        self.scaler = None
        self.preprocessor = CachedPreprocessor()
        self.monitor = PerformanceMonitor()
        self._load_models(model_path, scaler_path)
    
    def _load_models(self, model_path: str, scaler_path: str):
        """Load models with error handling"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def predict_cached(self, data_hash: str, features: np.ndarray) -> Tuple[int, float]:
        """Cached prediction for repeated inputs"""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return int(prediction), float(probability)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability > 0.7:
            return 'High'
        elif probability > 0.3:
            return 'Medium'
        else:
            return 'Low'

class DatabaseOptimizer:
    """Optimized database operations with connection pooling"""
    
    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.engine = None
        self._create_engine()
    
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling"""
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool
        
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    
    def batch_insert(self, data: List[Dict], table_name: str, batch_size: int = 1000):
        """Batch insert for better performance"""
        from sqlalchemy import text
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Create INSERT statement
            columns = list(batch[0].keys())
            placeholders = ', '.join([f':{col}' for col in columns])
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            with self.engine.begin() as conn:
                conn.execute(text(insert_stmt), batch)

class BundleOptimizer:
    """Optimize bundle size and load times"""
    
    @staticmethod
    def create_requirements_minimal():
        """Create minimal requirements.txt with only essential dependencies"""
        minimal_requirements = [
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "flask>=2.3.0",
            "sqlalchemy>=2.0.0"
        ]
        
        with open('requirements_minimal.txt', 'w') as f:
            f.write('\n'.join(minimal_requirements))
        
        logger.info("Created minimal requirements.txt")
    
    @staticmethod
    def optimize_model_size(model_path: str, output_path: str):
        """Optimize model size by removing unnecessary attributes"""
        model = joblib.load(model_path)
        
        # Remove unnecessary attributes to reduce size
        if hasattr(model, 'feature_names_in_'):
            delattr(model, 'feature_names_in_')
        
        # Save optimized model
        joblib.dump(model, output_path, compress=3)
        
        original_size = Path(model_path).stat().st_size / 1024 / 1024
        optimized_size = Path(output_path).stat().st_size / 1024 / 1024
        
        logger.info(f"Model size reduced from {original_size:.2f}MB to {optimized_size:.2f}MB")

def main():
    """Main function to demonstrate optimizations"""
    logger.info("Starting performance optimization analysis")
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Example usage of optimizations
    optimizer = BundleOptimizer()
    optimizer.create_requirements_minimal()
    
    logger.info("Performance optimization analysis completed")

if __name__ == "__main__":
    main()