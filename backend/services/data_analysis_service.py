import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
import io
import base64
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Class for analyzing and visualizing datasets"""
    
    def __init__(self, file_path=None, df=None):
        """Initialize with either a file path or a DataFrame"""
        self.file_path = file_path
        self.df = df
        self.original_df = None
        self.numeric_features = []
        self.categorical_features = []
        self.target_column = None
        self.model = None
        self.preprocessing_pipeline = None
        self.analysis_results = {}
        self.visualizations = {}
        
    def load_data(self):
        """Load data from file path"""
        if self.df is not None:
            self.original_df = self.df.copy()
            return self.df
            
        if self.file_path is None:
            raise ValueError("No file path or DataFrame provided")
            
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        try:
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
            raise
    
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
    
    def analyze_data(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            self.load_data()
            
        # Correlation analysis for numeric features
        if len(self.numeric_features) > 1:
            correlation_matrix = self.df[self.numeric_features].corr().round(2).to_dict()
            
            # Find top correlations
            corr_df = self.df[self.numeric_features].corr().unstack().sort_values(ascending=False)
            # Remove self-correlations
            corr_df = corr_df[corr_df < 1.0]
            top_correlations = corr_df.head(10).to_dict()
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
                value_counts = self.df[col].value_counts().head(10).to_dict()
                categorical_analysis[col] = {
                    'unique_values': self.df[col].nunique(),
                    'top_values': value_counts
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
                    correlations_with_target = self.df[self.numeric_features].corr()[self.target_column].sort_values(ascending=False).to_dict()
                    target_analysis['correlations'] = correlations_with_target
            
            elif self.target_column in self.categorical_features:
                value_counts = self.df[self.target_column].value_counts().to_dict()
                target_analysis = {
                    'unique_values': self.df[self.target_column].nunique(),
                    'value_counts': value_counts
                }
        
        # Store analysis results
        self.analysis_results['data_analysis'] = {
            'correlation_matrix': correlation_matrix,
            'top_correlations': top_correlations,
            'distributions': distributions,
            'categorical_analysis': categorical_analysis,
            'target_analysis': target_analysis
        }
        
        return self.analysis_results['data_analysis']
    
    def create_visualizations(self):
        """Create visualizations for the dataset"""
        if self.df is None:
            self.load_data()
            
        visualizations = {}
        
        # Distribution plots for numeric features
        for col in self.numeric_features[:10]:  # Limit to first 10 features
            if col in self.df.columns:
                fig = px.histogram(self.df, x=col, marginal="box", title=f"Distribution of {col}")
                visualizations[f'dist_{col}'] = fig.to_json()
        
        # Correlation heatmap
        if len(self.numeric_features) > 1:
            fig = px.imshow(
                self.df[self.numeric_features].corr(),
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation")
            )
            visualizations['correlation_heatmap'] = fig.to_json()
        
        # Categorical feature plots
        for col in self.categorical_features[:5]:  # Limit to first 5 features
            if col in self.df.columns and self.df[col].nunique() < 15:  # Only plot if fewer than 15 categories
                fig = px.bar(
                    self.df[col].value_counts().reset_index(),
                    x='index',
                    y=col,
                    title=f"Count of {col}"
                )
                visualizations[f'cat_{col}'] = fig.to_json()
        
        # Scatter plots for numeric features vs target (if target is specified)
        if self.target_column and self.target_column in self.numeric_features:
            for col in self.numeric_features[:5]:  # Limit to first 5 features
                if col != self.target_column and col in self.df.columns:
                    fig = px.scatter(
                        self.df,
                        x=col,
                        y=self.target_column,
                        title=f"{col} vs {self.target_column}",
                        trendline="ols"
                    )
                    visualizations[f'scatter_{col}_vs_{self.target_column}'] = fig.to_json()
        
        # Pair plot for top correlated features
        if len(self.numeric_features) > 2:
            # Get top 4 correlated features with target or among themselves
            if self.target_column and self.target_column in self.numeric_features:
                corr_with_target = abs(self.df[self.numeric_features].corr()[self.target_column]).sort_values(ascending=False)
                top_features = [self.target_column] + corr_with_target[corr_with_target.index != self.target_column].head(3).index.tolist()
            else:
                # If no target, get features with highest average correlation
                corr_matrix = self.df[self.numeric_features].corr().abs()
                avg_corr = corr_matrix.mean().sort_values(ascending=False)
                top_features = avg_corr.head(4).index.tolist()
            
            fig = px.scatter_matrix(
                self.df,
                dimensions=top_features,
                title="Pair Plot of Top Features"
            )
            visualizations['pair_plot'] = fig.to_json()
        
        # Box plots for categorical vs numeric
        if self.categorical_features and self.numeric_features:
            for cat_col in self.categorical_features[:2]:  # Limit to first 2 categorical features
                if cat_col in self.df.columns and self.df[cat_col].nunique() < 10:
                    for num_col in self.numeric_features[:2]:  # Limit to first 2 numeric features
                        if num_col in self.df.columns:
                            fig = px.box(
                                self.df,
                                x=cat_col,
                                y=num_col,
                                title=f"{num_col} by {cat_col}"
                            )
                            visualizations[f'box_{cat_col}_{num_col}'] = fig.to_json()
        
        # Store visualizations
        self.visualizations = visualizations
        return visualizations
    
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
        fig.add_line(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], line_dash="dash", line_color="red")
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