import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import logging
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import warnings
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from flask import Blueprint, request, jsonify, current_app

# Create a Blueprint for data analysis routes
data_analysis_bp = Blueprint('data_analysis', __name__)

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
                    }
            
            return outlier_analysis
        
        except Exception as e:
            logger.error(f"Error in outlier analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Outlier analysis failed: {str(e)}")

    def analyze_time_series(self) -> Dict:
        """Perform time series analysis"""
        try:
            time_series_analysis = {}
            time_cols = detect_time_series(self.df)
            
            for time_col in time_cols:
                if time_col in self.df.columns:
                    # Convert to datetime if not already
                    try:
                        time_series = pd.to_datetime(self.df[time_col])
                        self.df[time_col] = time_series
                    except:
                        continue
                    
                    # Basic time series statistics
                    time_stats = {
                        'start_date': time_series.min().isoformat(),
                        'end_date': time_series.max().isoformat(),
                        'duration_days': (time_series.max() - time_series.min()).days,
                        'frequency': time_series.diff().mode().iloc[0].total_seconds() if not time_series.diff().mode().empty else None
                    }
                    
                    # Analyze patterns for each numeric feature
                    feature_patterns = {}
                    for feature in self.numeric_features:
                        if feature != time_col and feature in self.df.columns:
                            try:
                                # Create time series object
                                ts_data = self.df.set_index(time_col)[feature]
                                
                                # Calculate basic statistics
                                feature_patterns[feature] = {
                                    'mean': float(ts_data.mean()),
                                    'std': float(ts_data.std()),
                                    'min': float(ts_data.min()),
                                    'max': float(ts_data.max()),
                                    'first_value': float(ts_data.iloc[0]),
                                    'last_value': float(ts_data.iloc[-1]),
                                    'change': float(ts_data.iloc[-1] - ts_data.iloc[0]),
                                    'pct_change': float((ts_data.iloc[-1] - ts_data.iloc[0]) / ts_data.iloc[0] * 100) if ts_data.iloc[0] != 0 else 0
                                }
                            except Exception as e:
                                logger.warning(f"Error analyzing patterns for feature {feature}: {str(e)}")
                                continue
                    
                    time_series_analysis[time_col] = {
                        'stats': time_stats,
                        'patterns': feature_patterns
                    }
            
            return time_series_analysis
        
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Time series analysis failed: {str(e)}")

    def analyze_correlations(self) -> Dict:
        """Perform detailed correlation analysis"""
        try:
            correlation_analysis = {}
            
            # Pearson correlation for numeric features
            if len(self.numeric_features) > 1:
                pearson_corr = self.df[self.numeric_features].corr(method='pearson')
                spearman_corr = self.df[self.numeric_features].corr(method='spearman')
                
                for col1 in self.numeric_features:
                    for col2 in self.numeric_features:
                        if col1 != col2:
                            key = f"{col1}_{col2}"
                            correlation_analysis[key] = {
                                'pearson': {
                                    'correlation': float(pearson_corr.loc[col1, col2]),
                                    'p_value': float(stats.pearsonr(self.df[col1].dropna(), self.df[col2].dropna())[1])
                                },
                                'spearman': {
                                    'correlation': float(spearman_corr.loc[col1, col2]),
                                    'p_value': float(stats.spearmanr(self.df[col1].dropna(), self.df[col2].dropna())[1])
                                }
                            }
            
            # Chi-square test for categorical features
            if len(self.categorical_features) > 1:
                for col1 in self.categorical_features:
                    for col2 in self.categorical_features:
                        if col1 != col2:
                            contingency_table = pd.crosstab(self.df[col1], self.df[col2])
                            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                            
                            key = f"{col1}_{col2}"
                            correlation_analysis[key] = {
                                'chi_square': {
                                    'statistic': float(chi2),
                                    'p_value': float(p_value),
                                    'contingency_table': contingency_table.to_dict()
                                }
                            }
            
            return correlation_analysis
        
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Correlation analysis failed: {str(e)}")

    def analyze_distributions(self) -> Dict:
        """Analyze distributions of numeric and categorical features"""
        try:
            distribution_analysis = {}
            
            # Analyze numeric features
            for col in self.numeric_features:
                if col in self.df.columns:
                    data = self.df[col].dropna()
                    
                    # Basic statistics
                    stats_dict = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'skewness': float(data.skew()),
                        'kurtosis': float(data.kurtosis()),
                        'min': float(data.min()),
                        'max': float(data.max())
                    }
                    
                    # Normality tests
                    if len(data) >= 3:  # Minimum sample size for normality tests
                        shapiro_test = stats.shapiro(data)
                        ks_test = stats.kstest(data, 'norm')
                        
                        stats_dict.update({
                            'normality_tests': {
                                'shapiro': {
                                    'statistic': float(shapiro_test[0]),
                                    'p_value': float(shapiro_test[1])
                                },
                                'kolmogorov_smirnov': {
                                    'statistic': float(ks_test[0]),
                                    'p_value': float(ks_test[1])
                                }
                            }
                        })
                    
                    # Quantiles
                    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                    stats_dict['quantiles'] = {
                        str(int(q * 100)): float(data.quantile(q))
                        for q in quantiles
                    }
                    
                    distribution_analysis[col] = stats_dict
            
            # Analyze categorical features
            for col in self.categorical_features:
                if col in self.df.columns:
                    value_counts = self.df[col].value_counts()
                    proportions = self.df[col].value_counts(normalize=True)
                    
                    distribution_analysis[col] = {
                        'unique_values': int(len(value_counts)),
                        'mode': str(value_counts.index[0]) if not value_counts.empty else None,
                        'frequencies': {
                            str(k): int(v) for k, v in value_counts.items()
                        },
                        'proportions': {
                            str(k): float(v) for k, v in proportions.items()
                        },
                        'entropy': float(stats.entropy(proportions))
                    }
            
            return distribution_analysis
        
        except Exception as e:
            logger.error(f"Error in distribution analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise AnalysisError(f"Distribution analysis failed: {str(e)}")

def analyze_dataset(file_path: str = None, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Analyze a dataset and return comprehensive analysis results
    
    Args:
        file_path: Path to the data file
        df: Pandas DataFrame object
        
    Returns:
        Dict containing analysis results
    """
    try:
        analyzer = DataAnalyzer(file_path=file_path, df=df)
        
        # Basic exploration
        basic_info = analyzer.explore_data()
        
        # Detailed analysis
        analysis_results = analyzer.analyze_data()
        
        # Outlier analysis
        outlier_analysis = analyzer.analyze_outliers()
        
        # Time series analysis
        time_series_analysis = analyzer.analyze_time_series()
        
        # Correlation analysis
        correlation_analysis = analyzer.analyze_correlations()
        
        # Distribution analysis
        distribution_analysis = analyzer.analyze_distributions()
        
        # Combine all results
        results = {
            'basic_info': basic_info,
            'analysis_results': analysis_results,
            'outlier_analysis': outlier_analysis,
            'time_series_analysis': time_series_analysis,
            'correlation_analysis': correlation_analysis,
            'distribution_analysis': distribution_analysis
        }
        
        return convert_to_serializable(results)
    
    except Exception as e:
        logger.error(f"Error in dataset analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise AnalysisError(f"Dataset analysis failed: {str(e)}")

@data_analysis_bp.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        analysis_type = data.get('analysis_type', 'basic')
        
        # Convert relative path to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {file_path}'
            }), 404
        
        # Create analyzer instance and load data
        analyzer = DataAnalyzer(file_path=file_path)
        analyzer.load_data()
        
        if analysis_type == 'basic':
            result = analyzer.explore_data()
        elif analysis_type == 'advanced':
            result = analyzer.analyze_data()
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid analysis type'
            }), 400
        
        return jsonify({
            'success': True,
            'analysis': result,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def perform_basic_analysis(df):
    """Perform basic statistical analysis on the dataset"""
    try:
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        basic_stats = df[numeric_cols].describe().to_dict()
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        
        # Data types
        data_types = df.dtypes.astype(str).to_dict()
        
        # Correlation matrix for numeric columns
        correlation = df[numeric_cols].corr().to_dict() if len(numeric_cols) > 0 else {}
        
        return {
            'basic_stats': basic_stats,
            'missing_values': missing_values,
            'data_types': data_types,
            'correlation': correlation
        }
        
    except Exception as e:
        logger.error(f"Error in perform_basic_analysis: {str(e)}")
        raise

def perform_advanced_analysis(df):
    """Perform advanced statistical analysis on the dataset"""
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric columns found for advanced analysis'
            }
        
        # Standardize numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        
        # PCA Analysis
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_.tolist()
        
        # Clustering Analysis (K-means)
        n_clusters_range = range(2, min(6, len(df)))
        silhouette_scores = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        
        # Perform final clustering with optimal number of clusters
        final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = final_kmeans.fit_predict(scaled_data)
        
        return {
            'pca_analysis': {
                'explained_variance_ratio': explained_variance,
                'cumulative_variance_ratio': np.cumsum(explained_variance).tolist(),
                'n_components': len(explained_variance)
            },
            'clustering_analysis': {
                'optimal_clusters': optimal_clusters,
                'silhouette_scores': silhouette_scores,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': final_kmeans.cluster_centers_.tolist()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in perform_advanced_analysis: {str(e)}")
        raise

@data_analysis_bp.route('/outliers', methods=['POST'])
def analyze_outliers():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        columns = data.get('columns')
        
        df = pd.read_csv(file_path)
        
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_results = {}
        
        for column in columns:
            if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
                continue
                
            # Calculate Z-scores
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            z_score_outliers = df[z_scores > 3].index.tolist()
            
            # Calculate IQR bounds
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
            
            outlier_results[column] = {
                'z_score': {
                    'outliers': z_score_outliers,
                    'count': len(z_score_outliers)
                },
                'iqr': {
                    'outliers': iqr_outliers,
                    'count': len(iqr_outliers),
                    'bounds': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    }
                }
            }
        
        return jsonify({
            'success': True,
            'outliers': outlier_results,
            'message': 'Outlier analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_outliers: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_analysis_bp.route('/time-series', methods=['POST'])
def analyze_time_series():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        time_column = data.get('time_column')
        target_columns = data.get('target_columns')
        
        df = pd.read_csv(file_path)
        
        # Convert time column to datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        if not target_columns:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            target_columns.remove(time_column) if time_column in target_columns else None
        
        time_series_results = {}
        
        for column in target_columns:
            if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
                continue
            
            # Basic time series statistics
            stats = {
                'start_date': df[time_column].min().isoformat(),
                'end_date': df[time_column].max().isoformat(),
                'duration_days': (df[time_column].max() - df[time_column].min()).days,
                'frequency': pd.infer_freq(df[time_column]),
                'statistics': {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'first_value': float(df[column].iloc[0]),
                    'last_value': float(df[column].iloc[-1]),
                    'change': float(df[column].iloc[-1] - df[column].iloc[0]),
                    'pct_change': float((df[column].iloc[-1] - df[column].iloc[0]) / df[column].iloc[0] * 100)
                }
            }
            
            time_series_results[column] = stats
        
        return jsonify({
            'success': True,
            'time_series_analysis': time_series_results,
            'message': 'Time series analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_time_series: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 