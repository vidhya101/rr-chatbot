import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import logging
import io
import base64
from datetime import datetime
import statsmodels.api as sm
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import shap
import lime
import lime.lime_tabular
from interpret import show
from interpret.blackbox import LimeTabular, ShapKernel
from interpret.glassbox import ExplainableBoostingMachine
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame for analysis"""
    if df is None or df.empty:
        raise AnalysisError("DataFrame is empty or None")
    if df.columns.duplicated().any():
        raise AnalysisError("DataFrame contains duplicate column names")
    return True

def detect_time_series(df: pd.DataFrame) -> List[str]:
    """Detect potential time series columns"""
    time_cols = []
    for col in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_cols.append(col)
            elif pd.to_datetime(df[col], errors='coerce').notna().all():
                time_cols.append(col)
        except:
            continue
    return time_cols

def convert_to_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict('records'))
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    return obj

class DataAnalyzer:
    """Class for analyzing and visualizing datasets"""
    
    def __init__(self, file_path=None, df=None):
        """Initialize with either a file path or a DataFrame"""
        self.file_path = file_path
        self.df = df
        self.original_df = None
        self.numeric_features = []
        self.categorical_features = []
        self.time_series_features = []
        self.target_column = None
        self.model = None
        self.preprocessing_pipeline = None
        self.analysis_results = {}
        self.visualizations = {}
        self.clustering_results = {}
        self.time_series_analysis = {}
        self.advanced_stats = {}
        self.model_evaluation = {}
        self.feature_importance = {}
        self.model_interpretability = {}
        self.anomaly_detection = {}
    
    def load_data(self):
        """Load data from file path"""
        try:
            if self.df is not None:
                self.original_df = self.df.copy()
                return self.df
            
            if self.file_path is None:
                raise ValueError("No file path or DataFrame provided")
            
            file_extension = os.path.splitext(self.file_path)[1].lower()
            
            if file_extension == '.csv':
                self.df = pd.read_csv(self.file_path)
            elif file_extension in ['.xls', '.xlsx']:
                self.df = pd.read_excel(self.file_path)
            elif file_extension == '.json':
                self.df = pd.read_json(self.file_path)
            elif file_extension == '.txt':
                # Try to infer delimiter
                self.df = pd.read_csv(self.file_path, sep=None, engine='python')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.original_df = self.df.copy()
            return self.df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error loading data: {str(e)}")
    
    def analyze_data(self):
        """Perform exploratory data analysis"""
        try:
            if self.df is None:
                self.load_data()
            
            # Correlation analysis for numeric features
            if len(self.numeric_features) > 1:
                correlation_matrix = self.df[self.numeric_features].corr().round(2).to_dict('split')
                
                # Find top correlations
                corr_df = self.df[self.numeric_features].corr().unstack()
                # Remove self-correlations
                corr_df = corr_df[corr_df < 1.0]
                # Convert index tuples to strings for JSON serialization
                top_correlations = {f"{idx[0]}_{idx[1]}": val for idx, val in corr_df.head(10).items()}
            else:
                correlation_matrix = {}
                top_correlations = {}
            
            # Statistical tests and distributions
            distributions = {}
            for col in self.numeric_features:
                if col in self.df.columns:
                    distributions[col] = {
                        'mean': float(self.df[col].mean()),
                        'median': float(self.df[col].median()),
                        'std': float(self.df[col].std()),
                        'min': float(self.df[col].min()),
                        'max': float(self.df[col].max()),
                        'skew': float(self.df[col].skew()),
                        'kurtosis': float(self.df[col].kurtosis())
                    }
            
            # Categorical analysis
            categorical_analysis = {}
            for col in self.categorical_features:
                if col in self.df.columns:
                    value_counts = self.df[col].value_counts().head(10)
                    categorical_analysis[col] = {
                        'unique_values': int(self.df[col].nunique()),
                        'top_values': {str(k): int(v) for k, v in value_counts.items()}
                    }
            
            # Target variable analysis if specified
            target_analysis = {}
            if self.target_column and self.target_column in self.df.columns:
                if self.target_column in self.numeric_features:
                    target_analysis = {
                        'mean': float(self.df[self.target_column].mean()),
                        'median': float(self.df[self.target_column].median()),
                        'std': float(self.df[self.target_column].std()),
                        'min': float(self.df[self.target_column].min()),
                        'max': float(self.df[self.target_column].max()),
                        'skew': float(self.df[self.target_column].skew()),
                        'kurtosis': float(self.df[self.target_column].kurtosis())
                    }
                    
                    # Relationship with other features
                    if len(self.numeric_features) > 1:
                        correlations_with_target = self.df[self.numeric_features].corr()[self.target_column]
                        target_analysis['correlations'] = {
                            str(col): float(corr) 
                            for col, corr in correlations_with_target.items()
                        }
                
                elif self.target_column in self.categorical_features:
                    value_counts = self.df[self.target_column].value_counts()
                    target_analysis = {
                        'unique_values': int(self.df[self.target_column].nunique()),
                        'value_counts': {str(k): int(v) for k, v in value_counts.items()}
                    }
            
            # Store analysis results
            self.analysis_results['data_analysis'] = {
                'correlation_matrix': correlation_matrix,
                'top_correlations': top_correlations,
                'distributions': distributions,
                'categorical_analysis': categorical_analysis,
                'target_analysis': target_analysis
            }
            
            return convert_to_serializable(self.analysis_results['data_analysis'])
        
        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Error in analyze_data: {str(e)}")
    
    def explore_data(self):
        """Explore the dataset and return basic statistics"""
        if self.df is None:
            self.load_data()
            
        # Basic info
        info_buffer = io.StringIO()
        self.df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()
        
        # Summary statistics
        summary_stats = self.df.describe(include='all').fillna('').to_dict()
        
        # Missing values
        missing_values = self.df.isnull().sum().to_dict()
        missing_percent = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        
        # Data types
        data_types = self.df.dtypes.astype(str).to_dict()
        
        # Identify numeric and categorical features
        self.numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Sample data
        sample_data = self.df.head(5).to_dict(orient='records')
        
        # Store results
        self.analysis_results['basic_info'] = {
            'info': info_str,
            'summary_stats': summary_stats,
            'missing_values': missing_values,
            'missing_percent': missing_percent,
            'data_types': data_types,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'sample_data': sample_data,
            'shape': self.df.shape
        }
        
        return self.analysis_results['basic_info']
    
    def clean_data(self, handle_missing='mean', handle_outliers=None, drop_columns=None):
        """Clean the dataset by handling missing values and outliers"""
        if self.df is None:
            self.load_data()
            
        # Store original shape
        original_shape = self.df.shape
        
        # Drop specified columns
        if drop_columns:
            self.df = self.df.drop(columns=drop_columns, errors='ignore')
            
        # Handle missing values
        if handle_missing == 'drop':
            self.df = self.df.dropna()
        elif handle_missing == 'mean':
            for col in self.numeric_features:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif handle_missing == 'median':
            for col in self.numeric_features:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        elif handle_missing == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan)
                
        # For categorical features, fill with mode
        for col in self.categorical_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
                
        # Handle outliers
        if handle_outliers == 'remove':
            for col in self.numeric_features:
                if col in self.df.columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        elif handle_outliers == 'cap':
            for col in self.numeric_features:
                if col in self.df.columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                    
        # Store cleaning results
        self.analysis_results['cleaning'] = {
            'original_shape': original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': original_shape[0] - self.df.shape[0],
            'columns_removed': original_shape[1] - self.df.shape[1],
            'missing_values_after': self.df.isnull().sum().to_dict(),
            'missing_percent_after': (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        }
        
        return self.analysis_results['cleaning']
    
    def engineer_features(self, target_column=None):
        """Engineer features for analysis and modeling"""
        if self.df is None:
            self.load_data()
            
        # Set target column
        self.target_column = target_column
        
        # Store original columns
        original_columns = self.df.columns.tolist()
        
        # Create date features if date columns exist
        date_columns = []
        for col in self.df.columns:
            try:
                if pd.api.types.is_string_dtype(self.df[col]):
                    # Try to convert to datetime
                    self.df[f'{col}_parsed'] = pd.to_datetime(self.df[col], errors='coerce')
                    if not self.df[f'{col}_parsed'].isnull().all():
                        date_columns.append(f'{col}_parsed')
                        # Extract date components
                        self.df[f'{col}_year'] = self.df[f'{col}_parsed'].dt.year
                        self.df[f'{col}_month'] = self.df[f'{col}_parsed'].dt.month
                        self.df[f'{col}_day'] = self.df[f'{col}_parsed'].dt.day
                        self.df[f'{col}_dayofweek'] = self.df[f'{col}_parsed'].dt.dayofweek
                    else:
                        # If conversion failed for all values, drop the parsed column
                        self.df = self.df.drop(columns=[f'{col}_parsed'])
            except:
                # If conversion fails, continue
                if f'{col}_parsed' in self.df.columns:
                    self.df = self.df.drop(columns=[f'{col}_parsed'])
                continue
        
        # Create interaction features for numeric columns
        if len(self.numeric_features) > 1:
            for i, col1 in enumerate(self.numeric_features):
                for col2 in self.numeric_features[i+1:]:
                    if col1 in self.df.columns and col2 in self.df.columns:
                        self.df[f'{col1}_{col2}_interaction'] = self.df[col1] * self.df[col2]
        
        # One-hot encode categorical features with low cardinality
        for col in self.categorical_features:
            if col in self.df.columns and self.df[col].nunique() < 10:  # Only encode if fewer than 10 categories
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
        
        # Update numeric and categorical features
        self.numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Store feature engineering results
        self.analysis_results['feature_engineering'] = {
            'original_columns': original_columns,
            'new_columns': [col for col in self.df.columns if col not in original_columns],
            'date_columns_processed': date_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        
        return self.analysis_results['feature_engineering']
    
    def create_visualizations(self):
        """Create various visualizations for the dataset"""
        try:
            if self.df is None:
                self.load_data()
            
            visualizations = {}
            
            # Distribution plots for numeric features
            for col in self.numeric_features:
                if col in self.df.columns:
                    fig = px.histogram(self.df, x=col, title=f'Distribution of {col}')
                    visualizations[f'dist_{col}'] = {
                        'type': 'histogram',
                        'data': convert_to_serializable({
                            'x': self.df[col].tolist(),
                            'name': col
                        }),
                        'layout': fig.to_dict()['layout']
                    }
            
            # Box plots for numeric features
            for col in self.numeric_features:
                if col in self.df.columns:
                    fig = px.box(self.df, y=col, title=f'Box Plot of {col}')
                    visualizations[f'box_{col}'] = {
                        'type': 'box',
                        'data': convert_to_serializable({
                            'y': self.df[col].tolist(),
                            'name': col
                        }),
                        'layout': fig.to_dict()['layout']
                    }
            
            # Scatter plots for pairs of numeric features
            if len(self.numeric_features) > 1:
                for i, col1 in enumerate(self.numeric_features[:-1]):
                    for col2 in self.numeric_features[i+1:]:
                        if col1 in self.df.columns and col2 in self.df.columns:
                            fig = px.scatter(self.df, x=col1, y=col2, 
                                          title=f'Scatter Plot: {col1} vs {col2}')
                            visualizations[f'scatter_{col1}_{col2}'] = {
                                'type': 'scatter',
                                'data': convert_to_serializable({
                                    'x': self.df[col1].tolist(),
                                    'y': self.df[col2].tolist(),
                                    'mode': 'markers'
                                }),
                                'layout': fig.to_dict()['layout']
                            }
            
            # Bar plots for categorical features
            for col in self.categorical_features:
                if col in self.df.columns:
                    value_counts = self.df[col].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f'Distribution of {col}')
                    visualizations[f'bar_{col}'] = {
                        'type': 'bar',
                        'data': convert_to_serializable({
                            'x': value_counts.index.tolist(),
                            'y': value_counts.values.tolist()
                        }),
                        'layout': fig.to_dict()['layout']
                    }
            
            # Correlation heatmap for numeric features
            if len(self.numeric_features) > 1:
                corr_matrix = self.df[self.numeric_features].corr()
                fig = px.imshow(corr_matrix, 
                              labels=dict(color="Correlation"),
                              title='Correlation Heatmap')
                visualizations['correlation_heatmap'] = {
                    'type': 'heatmap',
                    'data': convert_to_serializable({
                        'z': corr_matrix.values.tolist(),
                        'x': corr_matrix.columns.tolist(),
                        'y': corr_matrix.index.tolist()
                    }),
                    'layout': fig.to_dict()['layout']
                }
            
            # Target variable visualizations if specified
            if self.target_column:
                if self.target_column in self.numeric_features:
                    # Scatter plots of features vs target
                    for feature in self.numeric_features:
                        if feature != self.target_column and feature in self.df.columns:
                            fig = px.scatter(self.df, x=feature, y=self.target_column,
                                          title=f'{feature} vs {self.target_column}')
                            visualizations[f'target_scatter_{feature}'] = {
                                'type': 'scatter',
                                'data': convert_to_serializable({
                                    'x': self.df[feature].tolist(),
                                    'y': self.df[self.target_column].tolist(),
                                    'mode': 'markers'
                                }),
                                'layout': fig.to_dict()['layout']
                            }
                
                elif self.target_column in self.categorical_features:
                    # Box plots of numeric features by target categories
                    for feature in self.numeric_features:
                        if feature in self.df.columns:
                            fig = px.box(self.df, x=self.target_column, y=feature,
                                      title=f'{feature} by {self.target_column}')
                            visualizations[f'target_box_{feature}'] = {
                                'type': 'box',
                                'data': convert_to_serializable({
                                    'x': self.df[self.target_column].tolist(),
                                    'y': self.df[feature].tolist()
                                }),
                                'layout': fig.to_dict()['layout']
                            }
            
            self.visualizations = visualizations
            return convert_to_serializable(visualizations)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error creating visualizations: {str(e)}")
    
    def train_model(self, model_type='linear'):
        """Train a predictive model if target column is specified"""
        if self.df is None:
            self.load_data()
            
        if not self.target_column or self.target_column not in self.df.columns:
            return {"error": "Target column not specified or not found in dataframe"}
            
        # Prepare data for modeling
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            return {"error": f"Unsupported model type: {model_type}"}
            
        # Create and train pipeline
        self.preprocessing_pipeline = preprocessor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (for random forest)
        feature_importance = {}
        if model_type == 'random_forest':
            # Get feature names after preprocessing
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == 'cat':
                    # For categorical features, get the one-hot encoded feature names
                    for col in cols:
                        categories = trans.named_steps['onehot'].categories_[cols.index(col)]
                        feature_names.extend([f"{col}_{cat}" for cat in categories])
                else:
                    # For numeric features, use the column names
                    feature_names.extend(cols)
            
            # Get feature importances
            importances = pipeline.named_steps['model'].feature_importances_
            
            # Create a dictionary of feature importances
            if len(feature_names) == len(importances):
                feature_importance = dict(zip(feature_names, importances))
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Create residual plot
        fig = px.scatter(
            x=y_test,
            y=y_pred - y_test,
            labels={'x': 'Actual', 'y': 'Residuals'},
            title='Residual Plot'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        residual_plot = fig.to_json()
        
        # Create actual vs predicted plot
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title='Actual vs Predicted'
        )
        fig.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Perfect Prediction'
            )
        )
        actual_vs_predicted = fig.to_json()
        
        # Store model results
        self.analysis_results['model'] = {
            'model_type': model_type,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'residual_plot': residual_plot,
            'actual_vs_predicted': actual_vs_predicted
        }
        
        return self.analysis_results['model']
    
    def generate_dashboard_data(self):
        """Generate data for an interactive dashboard"""
        if not self.analysis_results:
            self.explore_data()
            self.analyze_data()
            self.create_visualizations()
            
        # Combine all results into a dashboard-friendly format
        dashboard_data = {
            'dataset_info': {
                'name': os.path.basename(self.file_path) if self.file_path else 'Uploaded Dataset',
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'target_column': self.target_column
            },
            'summary': self.analysis_results.get('basic_info', {}),
            'cleaning': self.analysis_results.get('cleaning', {}),
            'feature_engineering': self.analysis_results.get('feature_engineering', {}),
            'analysis': self.analysis_results.get('data_analysis', {}),
            'visualizations': self.visualizations,
            'model': self.analysis_results.get('model', {})
        }
        
        return dashboard_data
    
    def generate_report(self):
        """Generate a summary report of the analysis"""
        if not self.analysis_results:
            self.explore_data()
            self.analyze_data()
            
        # Create a report with key insights
        report = {
            'title': f"Data Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
            'dataset_info': {
                'name': os.path.basename(self.file_path) if self.file_path else 'Uploaded Dataset',
                'rows': self.df.shape[0],
                'columns': self.df.shape[1],
                'size_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'key_insights': []
        }
        
        # Add insights about data quality
        missing_data = self.analysis_results.get('basic_info', {}).get('missing_percent', {})
        if missing_data:
            high_missing = {k: v for k, v in missing_data.items() if v > 10}
            if high_missing:
                report['key_insights'].append({
                    'title': 'Data Quality Issues',
                    'description': f"Found {len(high_missing)} columns with >10% missing values: {', '.join(high_missing.keys())}"
                })
        
        # Add insights about distributions
        distributions = self.analysis_results.get('data_analysis', {}).get('distributions', {})
        if distributions:
            skewed_features = {k: v['skew'] for k, v in distributions.items() if abs(v['skew']) > 1}
            if skewed_features:
                report['key_insights'].append({
                    'title': 'Skewed Distributions',
                    'description': f"Found {len(skewed_features)} features with significant skew: {', '.join(skewed_features.keys())}"
                })
        
        # Add insights about correlations
        correlations = self.analysis_results.get('data_analysis', {}).get('top_correlations', {})
        if correlations:
            top_corr = list(correlations.items())[:5]
            report['key_insights'].append({
                'title': 'Strong Correlations',
                'description': f"Top correlation: {top_corr[0][0]} with value {top_corr[0][1]:.2f}"
            })
        
        # Add insights about target variable
        target_analysis = self.analysis_results.get('data_analysis', {}).get('target_analysis', {})
        if target_analysis and self.target_column:
            if 'correlations' in target_analysis:
                top_predictors = list(target_analysis['correlations'].items())[:3]
                report['key_insights'].append({
                    'title': 'Top Predictors',
                    'description': f"Top features correlated with {self.target_column}: " + 
                                  ", ".join([f"{k} ({v:.2f})" for k, v in top_predictors if k != self.target_column])
                })
        
        # Add model insights if available
        model_results = self.analysis_results.get('model', {})
        if model_results and 'r2' in model_results:
            report['key_insights'].append({
                'title': 'Model Performance',
                'description': f"The {model_results.get('model_type', 'predictive')} model achieved an R² of {model_results['r2']:.2f}"
            })
            
            if 'feature_importance' in model_results:
                top_features = list(model_results['feature_importance'].items())[:3]
                report['key_insights'].append({
                    'title': 'Important Features',
                    'description': f"Top important features: " + 
                                  ", ".join([f"{k} ({v:.2f})" for k, v in top_features])
                })
        
        return report

    def perform_time_series_analysis(self, time_col: str, value_col: str) -> Dict:
        """Perform time series analysis on specified columns"""
        try:
            # Ensure data is sorted by time
            ts_df = self.df.sort_values(time_col)
            ts_df[time_col] = pd.to_datetime(ts_df[time_col])
            
            # Perform seasonal decomposition
            ts_data = ts_df.set_index(time_col)[value_col]
            decomposition = seasonal_decompose(ts_data, period=min(len(ts_data), 12))
            
            # Perform stationarity test
            adf_test = adfuller(ts_data.dropna())
            
            # Calculate basic time series metrics
            metrics = {
                'mean': float(ts_data.mean()),
                'std': float(ts_data.std()),
                'trend_direction': 'increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'decreasing',
                'seasonality_strength': float(decomposition.seasonal.std() / ts_data.std()),
                'stationarity': {
                    'adf_statistic': float(adf_test[0]),
                    'p_value': float(adf_test[1]),
                    'is_stationary': adf_test[1] < 0.05
                }
            }
            
            # Create time series visualizations
            fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
            fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'), row=1, col=1)
            fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, name='Residual'), row=4, col=1)
            
            return {
                'metrics': metrics,
                'decomposition_plot': convert_to_serializable(fig.to_dict()),
                'components': {
                    'trend': convert_to_serializable(decomposition.trend.tolist()),
                    'seasonal': convert_to_serializable(decomposition.seasonal.tolist()),
                    'residual': convert_to_serializable(decomposition.resid.tolist())
                }
            }
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Time series analysis failed: {str(e)}")

    def perform_clustering(self, n_clusters: int = None, method: str = 'kmeans') -> Dict:
        """Perform clustering analysis on numeric features"""
        try:
            # Prepare data for clustering
            numeric_data = self.df[self.numeric_features].copy()
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            numeric_data = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns
            )
            
            # Scale the features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                max_clusters = min(10, len(numeric_data) // 2)
                silhouette_scores = []
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(scaled_data)
                    silhouette_scores.append(silhouette_score(scaled_data, labels))
                n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
            
            # Perform clustering
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=5)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Fit clustering model
            labels = clusterer.fit_predict(scaled_data)
            
            # Perform PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster in range(max(labels) + 1):
                cluster_mask = labels == cluster
                cluster_stats[f'cluster_{cluster}'] = {
                    'size': int(sum(cluster_mask)),
                    'percentage': float(sum(cluster_mask) / len(labels) * 100),
                    'feature_means': {
                        col: float(numeric_data[col][cluster_mask].mean())
                        for col in numeric_data.columns
                    }
                }
            
            # Create clustering visualization
            fig = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                color=labels.astype(str),
                title='Cluster Visualization (PCA)'
            )
            
            # Store clustering results
            self.clustering_results = {
                'method': method,
                'n_clusters': n_clusters,
                'labels': labels.tolist(),
                'cluster_stats': cluster_stats,
                'visualization': convert_to_serializable(fig.to_dict()),
                'pca_coords': convert_to_serializable(pca_result.tolist()),
                'feature_importance': {
                    'pc1': dict(zip(numeric_data.columns, pca.components_[0])),
                    'pc2': dict(zip(numeric_data.columns, pca.components_[1]))
                }
            }
            
            return self.clustering_results
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Clustering analysis failed: {str(e)}")

    def analyze_outliers(self) -> Dict:
        """Perform detailed outlier analysis"""
        try:
            outlier_analysis = {}
            
            for col in self.numeric_features:
                if col in self.df.columns:
                    data = self.df[col].dropna()
                    
                    # Calculate Z-scores
                    z_scores = np.abs(stats.zscore(data))
                    
                    # Calculate IQR bounds
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Identify outliers
                    z_score_outliers = data[z_scores > 3]
                    iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
                    
                    outlier_analysis[col] = {
                        'z_score_outliers': {
                            'count': len(z_score_outliers),
                            'percentage': float(len(z_score_outliers) / len(data) * 100),
                            'values': convert_to_serializable(z_score_outliers.tolist())
                        },
                        'iqr_outliers': {
                            'count': len(iqr_outliers),
                            'percentage': float(len(iqr_outliers) / len(data) * 100),
                            'values': convert_to_serializable(iqr_outliers.tolist())
                        },
                        'bounds': {
                            'lower': float(lower_bound),
                            'upper': float(upper_bound)
                        }
            
            return outlier_analysis
        
        except Exception as e:
            logger.error(f"Error in outlier analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Outlier analysis failed: {str(e)}")

    def perform_advanced_statistics(self) -> Dict:
        """Perform advanced statistical analysis on the dataset"""
        try:
            stats_results = {}
            
            # Normality tests for numeric features
            normality_tests = {}
            for col in self.numeric_features:
                if col in self.df.columns:
                    data = self.df[col].dropna()
                    if len(data) > 3:  # Minimum required for normality test
                        shapiro_test = stats.shapiro(data)
                        normality_tests[col] = {
                            'shapiro_statistic': float(shapiro_test[0]),
                            'shapiro_p_value': float(shapiro_test[1]),
                            'is_normal': shapiro_test[1] > 0.05
                        }
            stats_results['normality_tests'] = normality_tests
            
            # Correlation analysis with p-values
            if len(self.numeric_features) > 1:
                correlation_analysis = {}
                for i, col1 in enumerate(self.numeric_features):
                    for col2 in self.numeric_features[i+1:]:
                        if col1 in self.df.columns and col2 in self.df.columns:
                            pearson_corr = stats.pearsonr(
                                self.df[col1].dropna(),
                                self.df[col2].dropna()
                            )
                            spearman_corr = stats.spearmanr(
                                self.df[col1].dropna(),
                                self.df[col2].dropna()
                            )
                            correlation_analysis[f"{col1}_vs_{col2}"] = {
                                'pearson': {
                                    'correlation': float(pearson_corr[0]),
                                    'p_value': float(pearson_corr[1])
                                },
                                'spearman': {
                                    'correlation': float(spearman_corr[0]),
                                    'p_value': float(spearman_corr[1])
                                }
                stats_results['correlation_analysis'] = correlation_analysis
            
            # Chi-square tests for categorical features
            if len(self.categorical_features) > 1:
                chi_square_tests = {}
                for i, col1 in enumerate(self.categorical_features):
                    for col2 in self.categorical_features[i+1:]:
                        if col1 in self.df.columns and col2 in self.df.columns:
                            contingency = pd.crosstab(
                                self.df[col1].fillna('Missing'),
                                self.df[col2].fillna('Missing')
                            )
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            chi_square_tests[f"{col1}_vs_{col2}"] = {
                                'chi2_statistic': float(chi2),
                                'p_value': float(p_value),
                                'degrees_of_freedom': int(dof),
                                'is_dependent': p_value < 0.05
                            }
                stats_results['chi_square_tests'] = chi_square_tests
            
            self.advanced_stats = stats_results
            return convert_to_serializable(stats_results)
            
        except Exception as e:
            logger.error(f"Error in advanced statistical analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Advanced statistical analysis failed: {str(e)}")
    
    def evaluate_model_performance(self, cv_folds=5) -> Dict:
        """Perform detailed model evaluation including cross-validation and learning curves"""
        try:
            if not self.model or not self.target_column:
                raise AnalysisError("Model not trained or target column not specified")
            
            evaluation_results = {}
            
            # Prepare data
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            # Determine if classification or regression
            is_classification = self.target_column in self.categorical_features
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv_folds,
                scoring='accuracy' if is_classification else 'r2'
            )
            evaluation_results['cross_validation'] = {
                'scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std())
            }
            
            # Learning curves
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, X, y, cv=cv_folds,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy' if is_classification else 'r2'
            )
            evaluation_results['learning_curves'] = {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
                'test_scores_std': np.std(test_scores, axis=1).tolist()
            }
            
            # Classification specific metrics
            if is_classification:
                y_pred = self.model.predict(X)
                evaluation_results['classification_metrics'] = {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'precision': float(precision_score(y, y_pred, average='weighted')),
                    'recall': float(recall_score(y, y_pred, average='weighted')),
                    'f1': float(f1_score(y, y_pred, average='weighted')),
                    'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                    'classification_report': classification_report(y, y_pred, output_dict=True)
                }
                
                # ROC curve for binary classification
                if len(np.unique(y)) == 2:
                    y_prob = self.model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y, y_prob)
                    evaluation_results['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': float(auc(fpr, tpr))
                    }
            
            # Feature importance analysis
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    X.columns,
                    self.model.feature_importances_
                ))
                evaluation_results['feature_importance'] = {
                    k: float(v) for k, v in 
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                }
            
            self.model_evaluation = evaluation_results
            return convert_to_serializable(evaluation_results)
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Model evaluation failed: {str(e)}")

    def optimize_model(self, param_grid: Dict = None) -> Dict:
        """Perform hyperparameter optimization using grid search"""
        try:
            if not self.model or not self.target_column:
                raise AnalysisError("Model not trained or target column not specified")
            
            # Default parameter grids for different model types
            default_grids = {
                'linear': {},  # Linear regression doesn't have hyperparameters
                'random_forest': {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.1, 0.3],
                    'model__max_depth': [3, 4, 5],
                    'model__min_samples_split': [2, 5, 10]
                }
            }
            
            # Use provided param_grid or default based on model type
            if param_grid is None:
                model_type = self.analysis_results.get('model', {}).get('model_type', '')
                param_grid = default_grids.get(model_type, {})
            
            if not param_grid:
                return {"message": "No parameters to optimize"}
            
            # Prepare data
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5,
                scoring='accuracy' if self.target_column in self.categorical_features else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            
            # Store optimization results
            optimization_results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': {
                    'params': [str(p) for p in grid_search.cv_results_['params']],
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist()
                }
            }
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            
            return convert_to_serializable(optimization_results)
            
        except Exception as e:
            logger.error(f"Error in model optimization: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Model optimization failed: {str(e)}")

    def perform_feature_selection(self, method='mutual_info', k=10) -> Dict:
        """Perform feature selection using various methods"""
        try:
            if not self.target_column:
                raise AnalysisError("Target column not specified")
                
            feature_selection_results = {}
            
            # Prepare data
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            # Handle categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            X_encoded = pd.get_dummies(X, columns=categorical_cols)
            
            if method == 'mutual_info':
                # Mutual Information Selection
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
                selector.fit(X_encoded, y)
                selected_features = pd.DataFrame({
                    'feature': X_encoded.columns,
                    'importance': selector.scores_
                }).sort_values('importance', ascending=False)
                
            elif method == 'rfe':
                # Recursive Feature Elimination
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                selector = RFE(estimator=base_model, n_features_to_select=k)
                selector.fit(X_encoded, y)
                selected_features = pd.DataFrame({
                    'feature': X_encoded.columns,
                    'selected': selector.support_,
                    'rank': selector.ranking_
                }).sort_values('rank')
                
            elif method == 'polynomial':
                # Polynomial Feature Selection
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X_encoded)
                feature_names = poly.get_feature_names_out(X_encoded.columns)
                
                # Use mutual information to select top k polynomial features
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
                selector.fit(X_poly, y)
                selected_features = pd.DataFrame({
                    'feature': feature_names,
                    'importance': selector.scores_
                }).sort_values('importance', ascending=False)
            
            feature_selection_results = {
                'method': method,
                'selected_features': selected_features.to_dict('records'),
                'n_features_selected': k
            }
            
            self.feature_importance = feature_selection_results
            return convert_to_serializable(feature_selection_results)
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Feature selection failed: {str(e)}")
    
    def interpret_model_predictions(self, sample_size=100) -> Dict:
        """Generate model interpretability insights using SHAP and LIME"""
        try:
            if not self.model or not self.target_column:
                raise AnalysisError("Model not trained or target column not specified")
            
            interpretability_results = {}
            
            # Prepare data
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            # Sample data for interpretation
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X
            
            # SHAP Analysis
            explainer = shap.KernelExplainer(self.model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            # Global feature importance
            feature_importance = np.abs(shap_values).mean(0)
            global_importance = dict(zip(X.columns, feature_importance))
            
            # Local explanations for a few examples
            local_explanations = []
            for i in range(min(5, len(X_sample))):
                local_exp = {
                    'instance_id': i,
                    'feature_values': X_sample.iloc[i].to_dict(),
                    'prediction': float(self.model.predict(X_sample.iloc[i:i+1])[0]),
                    'shap_values': dict(zip(X.columns, shap_values[i]))
                }
                local_explanations.append(local_exp)
            
            # LIME Analysis
            categorical_features = [i for i, col in enumerate(X.columns) 
                                 if col in self.categorical_features]
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=X.columns,
                class_names=[self.target_column],
                categorical_features=categorical_features,
                mode='regression' if self.target_column in self.numeric_features else 'classification'
            )
            
            # Generate LIME explanations
            lime_explanations = []
            for i in range(min(5, len(X_sample))):
                exp = explainer.explain_instance(
                    X_sample.iloc[i].values, 
                    self.model.predict,
                    num_features=10
                )
                lime_explanations.append({
                    'instance_id': i,
                    'feature_values': X_sample.iloc[i].to_dict(),
                    'prediction': float(self.model.predict(X_sample.iloc[i:i+1])[0]),
                    'lime_weights': dict(exp.as_list())
                })
            
            interpretability_results = {
                'global_importance': global_importance,
                'local_explanations': {
                    'shap': local_explanations,
                    'lime': lime_explanations
                },
                'feature_interactions': self._analyze_feature_interactions(X_sample, shap_values)
            }
            
            self.model_interpretability = interpretability_results
            return convert_to_serializable(interpretability_results)
            
        except Exception as e:
            logger.error(f"Error in model interpretation: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Model interpretation failed: {str(e)}")
    
    def _analyze_feature_interactions(self, X_sample, shap_values) -> Dict:
        """Analyze feature interactions using SHAP values"""
        try:
            interactions = {}
            
            # Calculate pairwise feature interactions
            for i, feat1 in enumerate(X_sample.columns):
                for j, feat2 in enumerate(X_sample.columns[i+1:], i+1):
                    interaction_values = np.abs(
                        shap_values[:, i] * shap_values[:, j]
                    ).mean()
                    interactions[f"{feat1}_x_{feat2}"] = float(interaction_values)
            
            # Sort interactions by strength
            interactions = dict(sorted(
                interactions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Keep top 10 interactions
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {str(e)}")
            return {}
    
    def detect_anomalies(self, method='isolation_forest', contamination=0.1) -> Dict:
        """Detect anomalies in the dataset using various methods"""
        try:
            anomaly_results = {}
            
            # Prepare data
            X = self.df[self.numeric_features].copy()
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns
            )
            
            if method == 'isolation_forest':
                detector = IsolationForest(
                    contamination=contamination,
                    random_state=42
                )
            elif method == 'local_outlier_factor':
                detector = LocalOutlierFactor(
                    contamination=contamination,
                    novelty=True
                )
            else:
                raise ValueError(f"Unsupported anomaly detection method: {method}")
            
            # Fit detector and predict
            labels = detector.fit_predict(X)
            scores = detector.score_samples(X) if hasattr(detector, 'score_samples') else None
            
            # Calculate anomaly statistics
            anomaly_mask = labels == -1
            anomaly_stats = {
                'total_anomalies': int(sum(anomaly_mask)),
                'anomaly_percentage': float(sum(anomaly_mask) / len(labels) * 100),
                'feature_statistics': {}
            }
            
            # Calculate statistics for each feature
            for col in X.columns:
                anomaly_stats['feature_statistics'][col] = {
                    'mean_normal': float(X[col][~anomaly_mask].mean()),
                    'mean_anomaly': float(X[col][anomaly_mask].mean()),
                    'std_normal': float(X[col][~anomaly_mask].std()),
                    'std_anomaly': float(X[col][anomaly_mask].std())
                }
            
            # Store anomaly indices and scores
            anomaly_results = {
                'method': method,
                'contamination': contamination,
                'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
                'anomaly_scores': scores.tolist() if scores is not None else None,
                'statistics': anomaly_stats
            }
            
            self.anomaly_detection = anomaly_results
            return convert_to_serializable(anomaly_results)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Anomaly detection failed: {str(e)}")

def analyze_dataset(file_path, target_column=None, model_type='linear'):
    """Analyze a dataset and return results"""
    analyzer = DataAnalyzer(file_path=file_path)
    analyzer.load_data()
    analyzer.explore_data()
    analyzer.clean_data()
    analyzer.engineer_features(target_column=target_column)
    analyzer.analyze_data()
    analyzer.create_visualizations()
    
    if target_column:
        analyzer.train_model(model_type=model_type)
    
    dashboard_data = analyzer.generate_dashboard_data()
    report = analyzer.generate_report()
    
    return {
        'dashboard_data': dashboard_data,
        'report': report
    } 