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
from typing import Dict, List, Any, Tuple
from utils.data_utils import convert_to_serializable

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
        
        # Convert to serializable format
        return convert_to_serializable(self.analysis_results['basic_info'])
    
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
        """Create 4-6 focused visualizations from the dataset."""
        visualizations = []
        
        try:
            # Get numeric and categorical columns
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            date_cols = self.df.select_dtypes(include=['datetime64']).columns
            
            # 1. Correlation Heatmap (if we have numeric columns)
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                heatmap = {
                    'type': 'heatmap',
                    'x': corr_matrix.columns.tolist(),
                    'y': corr_matrix.columns.tolist(),
                    'z': corr_matrix.values.tolist(),
                    'colorscale': 'RdBu'
                }
                visualizations.append({
                    'type': 'correlation',
                    'title': 'Feature Correlation Analysis',
                    'plot': json.dumps({
                        'data': [heatmap],
                        'layout': {
                            'title': 'Feature Correlations',
                            'height': 500,
                            'annotations': [{
                                'x': corr_matrix.columns[i],
                                'y': corr_matrix.columns[j],
                                'text': f'{corr_matrix.values[i, j]:.2f}',
                                'showarrow': False,
                                'font': {'size': 10}
                            } for i in range(len(corr_matrix.columns)) for j in range(len(corr_matrix.columns))]
                        }
                    })
                })

            # 2. Time Series Analysis (if date column exists)
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                numeric_col = numeric_cols[0]
                # Aggregate by date
                daily_data = self.df.groupby(self.df[date_col].dt.date)[numeric_col].agg(['mean', 'min', 'max']).reset_index()
                
                line_data = {
                    'type': 'scatter',
                    'mode': 'lines',
                    'x': daily_data[date_col].astype(str).tolist(),
                    'y': daily_data['mean'].tolist(),
                    'name': 'Average',
                    'line': {'color': 'rgb(31, 119, 180)'}
                }
                range_data = {
                    'type': 'scatter',
                    'mode': 'lines',
                    'x': daily_data[date_col].astype(str).tolist() + daily_data[date_col].astype(str).tolist()[::-1],
                    'y': daily_data['max'].tolist() + daily_data['min'].tolist()[::-1],
                    'fill': 'toself',
                    'fillcolor': 'rgba(31, 119, 180, 0.2)',
                    'line': {'color': 'rgba(31, 119, 180, 0)'},
                    'name': 'Range'
                }
                visualizations.append({
                    'type': 'trend',
                    'title': f'Time Series Analysis: {numeric_col}',
                    'plot': json.dumps({
                        'data': [range_data, line_data],
                        'layout': {
                            'title': f'Trend Analysis of {numeric_col} Over Time',
                            'height': 500,
                            'xaxis': {'title': date_col},
                            'yaxis': {'title': numeric_col},
                            'showlegend': True,
                            'hovermode': 'x unified'
                        }
                    })
                })

            # 3. Distribution Analysis (Top 2 numeric features)
            if len(numeric_cols) >= 2:
                for col in numeric_cols[:2]:
                    hist_data = {
                        'type': 'histogram',
                        'x': self.df[col].dropna().tolist(),
                        'name': 'Distribution',
                        'opacity': 0.7,
                        'nbinsx': 30
                    }
                    box_data = {
                        'type': 'box',
                        'y': self.df[col].dropna().tolist(),
                        'name': 'Box Plot',
                        'boxpoints': 'outliers'
                    }
                    visualizations.append({
                        'type': 'distribution',
                        'title': f'Distribution Analysis: {col}',
                        'plot': json.dumps({
                            'data': [hist_data, box_data],
                            'layout': {
                                'title': f'Distribution Analysis of {col}',
                                'height': 500,
                                'showlegend': True,
                                'grid': {'rows': 1, 'columns': 2},
                                'annotations': [{
                                    'text': f'Mean: {self.df[col].mean():.2f}<br>Median: {self.df[col].median():.2f}',
                                    'showarrow': False,
                                    'x': 0.5,
                                    'y': 1.1,
                                    'xref': 'paper',
                                    'yref': 'paper'
                                }]
                            }
                        })
                    })

            # 4. Category Distribution (Top categorical feature)
            if len(cat_cols) > 0:
                col = cat_cols[0]
                value_counts = self.df[col].value_counts()
                pie_data = {
                    'type': 'pie',
                    'labels': value_counts.index.tolist()[:8],  # Top 8 categories
                    'values': value_counts.values.tolist()[:8],
                    'hole': 0.4,
                    'textinfo': 'label+percent',
                    'textposition': 'outside'
                }
                visualizations.append({
                    'type': 'category',
                    'title': f'Category Distribution: {col}',
                    'plot': json.dumps({
                        'data': [pie_data],
                        'layout': {
                            'title': f'Distribution of {col}',
                            'height': 500,
                            'showlegend': True,
                            'annotations': [{
                                'text': f'Total Categories: {len(value_counts)}',
                                'showarrow': False,
                                'x': 0.5,
                                'y': -0.1,
                                'xref': 'paper',
                                'yref': 'paper'
                            }]
                        }
                    })
                })

            # 5. Scatter Plot (if enough numeric columns with strong correlation)
            if len(numeric_cols) >= 2:
                corr_matrix = self.df[numeric_cols].corr()
                # Find the most correlated pair
                max_corr = 0
                max_pair = None
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if abs(corr_matrix.iloc[i,j]) > max_corr:
                            max_corr = abs(corr_matrix.iloc[i,j])
                            max_pair = (numeric_cols[i], numeric_cols[j])
                
                if max_pair and max_corr > 0.5:  # Only show if correlation is strong
                    col1, col2 = max_pair
                    scatter_data = {
                        'type': 'scatter',
                        'mode': 'markers',
                        'x': self.df[col1].tolist(),
                        'y': self.df[col2].tolist(),
                        'marker': {
                            'size': 8,
                            'opacity': 0.6,
                            'color': self.df[numeric_cols[0]].tolist(),
                            'colorscale': 'Viridis',
                            'showscale': True
                        }
                    }
                    visualizations.append({
                        'type': 'scatter',
                        'title': f'Feature Relationship: {col1} vs {col2}',
                        'plot': json.dumps({
                            'data': [scatter_data],
                            'layout': {
                                'title': f'{col1} vs {col2}<br>Correlation: {max_corr:.2f}',
                                'height': 500,
                                'xaxis': {'title': col1},
                                'yaxis': {'title': col2},
                                'hovermode': 'closest'
                            }
                        })
                    })

            # 6. Parallel Coordinates (if enough numeric columns)
            if len(numeric_cols) >= 3:
                # Normalize the data for better visualization
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(self.df[numeric_cols])
                
                parallel_data = {
                    'type': 'parcoords',
                    'line': {
                        'color': self.df[numeric_cols[0]].tolist(),
                        'colorscale': 'Viridis'
                    },
                    'dimensions': [{
                        'label': col,
                        'values': normalized_data[:, i].tolist()
                    } for i, col in enumerate(numeric_cols[:6])]  # Limit to 6 dimensions
                }
                visualizations.append({
                    'type': 'parallel',
                    'title': 'Multi-dimensional Feature Analysis',
                    'plot': json.dumps({
                        'data': [parallel_data],
                        'layout': {
                            'title': 'Parallel Coordinates Plot',
                            'height': 500
                        }
                    })
                })

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            # Return a basic visualization if there's an error
            visualizations.append({
                'type': 'error',
                'title': 'Error in Visualization',
                'plot': json.dumps({
                    'data': [{'type': 'scatter', 'x': [], 'y': []}],
                    'layout': {'title': 'Error creating visualization'}
                })
            })

        # Ensure we have at least 4 but no more than 6 visualizations
        return visualizations[:min(6, max(4, len(visualizations)))]
    
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

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file into a pandas DataFrame
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Explore the dataset to understand its structure and characteristics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing exploration results
    """
    try:
        logger.info("Exploring data")
        
        # Basic dataset information
        info = {
            "shape": tuple(df.shape),  # Convert to tuple for serialization
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": float(df.memory_usage(deep=True).sum() / 1024**2)  # Convert to float for serialization
        }
        
        # Handle numeric statistics carefully
        numeric_stats = df.describe().to_dict()
        for col in numeric_stats:
            numeric_stats[col] = {k: float(v) if not pd.isna(v) else None 
                                for k, v in numeric_stats[col].items()}
        info["numeric_stats"] = numeric_stats
        
        # Sample data with NaN handling
        sample_data = df.head().fillna("N/A").to_dict(orient='records')
        info["sample_data"] = sample_data
        
        logger.info("Data exploration completed")
        
        # Convert to serializable format
        return convert_to_serializable(info)
        
    except Exception as e:
        logger.error(f"Error exploring data: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the dataset by handling missing values and inconsistencies
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    try:
        logger.info("Starting data cleaning")
        cleaning_report = {"actions": []}
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        initial_missing = df_clean.isnull().sum()
        for column in df_clean.columns:
            missing_count = initial_missing[column]
            if missing_count > 0:
                if df_clean[column].dtype in ['int64', 'float64']:
                    # Fill numeric missing values with median
                    median_value = df_clean[column].median()
                    df_clean[column].fillna(median_value, inplace=True)
                    cleaning_report["actions"].append({
                        "column": column,
                        "action": "fill_missing",
                        "method": "median",
                        "count": int(missing_count)
                    })
                else:
                    # Fill categorical missing values with mode
                    mode_value = df_clean[column].mode()[0]
                    df_clean[column].fillna(mode_value, inplace=True)
                    cleaning_report["actions"].append({
                        "column": column,
                        "action": "fill_missing",
                        "method": "mode",
                        "count": int(missing_count)
                    })
        
        # Remove duplicates if any
        initial_rows = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df_clean)
        if duplicates_removed > 0:
            cleaning_report["actions"].append({
                "action": "remove_duplicates",
                "count": duplicates_removed
            })
        
        logger.info("Data cleaning completed")
        return df_clean, cleaning_report
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Conduct exploratory data analysis and identify key insights
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing analysis results and insights
    """
    try:
        logger.info("Starting data analysis")
        insights = []
        
        # 1. Purchase frequency analysis
        if 'purchase_date' in df.columns:
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            purchase_trends = df.groupby(df['purchase_date'].dt.date).size()
            insights.append({
                "title": "Purchase Frequency Trends",
                "description": f"Average daily purchases: {purchase_trends.mean():.2f}",
                "type": "trend",
                "data": {
                    "dates": purchase_trends.index.astype(str).tolist(),
                    "counts": purchase_trends.values.tolist()
                }
            })
        
        # 2. Product category analysis
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            insights.append({
                "title": "Top Product Categories",
                "description": f"Most popular category: {category_counts.index[0]}",
                "type": "category",
                "data": {
                    "categories": category_counts.index.tolist()[:5],
                    "counts": category_counts.values.tolist()[:5]
                }
            })
        
        # 3. Customer demographics
        if 'customer_age' in df.columns:
            age_stats = df['customer_age'].describe()
            insights.append({
                "title": "Customer Age Distribution",
                "description": f"Average customer age: {age_stats['mean']:.1f} years",
                "type": "distribution",
                "data": {
                    "stats": age_stats.to_dict()
                }
            })
        
        # 4. Price analysis
        if 'price' in df.columns:
            price_stats = df['price'].describe()
            insights.append({
                "title": "Price Distribution",
                "description": f"Average price: ${price_stats['mean']:.2f}",
                "type": "distribution",
                "data": {
                    "stats": price_stats.to_dict()
                }
            })
        
        # 5. Numeric feature correlations
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Create correlation heatmap
            heatmap_data = {
                "type": "heatmap",
                "x": numeric_cols.tolist(),
                "y": numeric_cols.tolist(),
                "z": corr_matrix.values.tolist(),
                "colorscale": "RdBu"
            }
            
            insights.append({
                "title": "Feature Correlations",
                "description": "Correlation heatmap of numeric features",
                "type": "correlation",
                "data": {
                    "heatmap": heatmap_data
                }
            })
            
            # Add scatter plots for highly correlated features
            top_correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > 0.5:  # Only show strong correlations
                        top_correlations.append((numeric_cols[i], numeric_cols[j], corr_value))
            
            for col1, col2, corr in sorted(top_correlations, key=lambda x: x[2], reverse=True)[:3]:
                insights.append({
                    "title": f"{col1} vs {col2}",
                    "description": f"Correlation: {corr:.2f}",
                    "type": "scatter",
                    "data": {
                        "x": df[col1].tolist(),
                        "y": df[col2].tolist(),
                        "text": df.index.tolist()
                    }
                })
        
        # 6. Time series analysis
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            for col in numeric_cols[:2]:  # First 2 numeric columns
                insights.append({
                    "title": f"{col} Over Time",
                    "description": f"Time series analysis of {col}",
                    "type": "trend",
                    "data": {
                        "x": df[date_col].astype(str).tolist(),
                        "y": df[col].tolist()
                    }
                })
        
        # 7. Distribution analysis for numeric features
        for col in numeric_cols[:3]:  # First 3 numeric columns
            insights.append({
                "title": f"Distribution of {col}",
                "description": f"Statistical distribution analysis",
                "type": "distribution",
                "data": {
                    "values": df[col].dropna().tolist(),
                    "stats": df[col].describe().to_dict()
                }
            })
        
        # 8. Category distribution
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols[:2]:  # First 2 categorical columns
            value_counts = df[col].value_counts()
            insights.append({
                "title": f"Distribution of {col}",
                "description": f"Category distribution analysis",
                "type": "category",
                "data": {
                    "labels": value_counts.index.tolist()[:10],
                    "values": value_counts.values.tolist()[:10]
                }
            })
        
        # 9. 3D scatter plot if enough numeric features
        if len(numeric_cols) >= 3:
            insights.append({
                "title": "3D Feature Relationship",
                "description": "3D visualization of feature relationships",
                "type": "3d",
                "data": {
                    "x": df[numeric_cols[0]].tolist(),
                    "y": df[numeric_cols[1]].tolist(),
                    "z": df[numeric_cols[2]].tolist()
                }
            })
        
        # 10. Parallel coordinates for multi-dimensional analysis
        if len(numeric_cols) >= 4:
            parallel_data = {
                "dimensions": [
                    {
                        "label": col,
                        "values": df[col].tolist()
                    } for col in numeric_cols[:6]  # First 6 numeric columns
                ]
            }
            insights.append({
                "title": "Multi-dimensional Analysis",
                "description": "Parallel coordinates plot of multiple features",
                "type": "parallel",
                "data": parallel_data
            })
        
        logger.info(f"Generated {len(insights)} insights")
        return {"insights": insights}
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return {"insights": [], "error": str(e)}

def create_visualizations(df: pd.DataFrame, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create 4-6 focused visualizations from the dataset."""
    visualizations = []
    
    try:
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # 1. Correlation Heatmap (if we have numeric columns)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            heatmap = {
                'type': 'heatmap',
                'x': corr_matrix.columns.tolist(),
                'y': corr_matrix.columns.tolist(),
                'z': corr_matrix.values.tolist(),
                'colorscale': 'RdBu'
            }
            visualizations.append({
                'type': 'correlation',
                'title': 'Feature Correlation Analysis',
                'plot': json.dumps({
                    'data': [heatmap],
                    'layout': {
                        'title': 'Feature Correlations',
                        'height': 500,
                        'annotations': [{
                            'x': corr_matrix.columns[i],
                            'y': corr_matrix.columns[j],
                            'text': f'{corr_matrix.values[i, j]:.2f}',
                            'showarrow': False,
                            'font': {'size': 10}
                        } for i in range(len(corr_matrix.columns)) for j in range(len(corr_matrix.columns))]
                    }
                })
            })

        # 2. Time Series Analysis (if date column exists)
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            numeric_col = numeric_cols[0]
            # Aggregate by date
            daily_data = df.groupby(df[date_col].dt.date)[numeric_col].agg(['mean', 'min', 'max']).reset_index()
            
            line_data = {
                'type': 'scatter',
                'mode': 'lines',
                'x': daily_data[date_col].astype(str).tolist(),
                'y': daily_data['mean'].tolist(),
                'name': 'Average',
                'line': {'color': 'rgb(31, 119, 180)'}
            }
            range_data = {
                'type': 'scatter',
                'mode': 'lines',
                'x': daily_data[date_col].astype(str).tolist() + daily_data[date_col].astype(str).tolist()[::-1],
                'y': daily_data['max'].tolist() + daily_data['min'].tolist()[::-1],
                'fill': 'toself',
                'fillcolor': 'rgba(31, 119, 180, 0.2)',
                'line': {'color': 'rgba(31, 119, 180, 0)'},
                'name': 'Range'
            }
            visualizations.append({
                'type': 'trend',
                'title': f'Time Series Analysis: {numeric_col}',
                'plot': json.dumps({
                    'data': [range_data, line_data],
                    'layout': {
                        'title': f'Trend Analysis of {numeric_col} Over Time',
                        'height': 500,
                        'xaxis': {'title': date_col},
                        'yaxis': {'title': numeric_col},
                        'showlegend': True,
                        'hovermode': 'x unified'
                    }
                })
            })

        # 3. Distribution Analysis (Top 2 numeric features)
        if len(numeric_cols) >= 2:
            for col in numeric_cols[:2]:
                hist_data = {
                    'type': 'histogram',
                    'x': df[col].dropna().tolist(),
                    'name': 'Distribution',
                    'opacity': 0.7,
                    'nbinsx': 30
                }
                box_data = {
                    'type': 'box',
                    'y': df[col].dropna().tolist(),
                    'name': 'Box Plot',
                    'boxpoints': 'outliers'
                }
                visualizations.append({
                    'type': 'distribution',
                    'title': f'Distribution Analysis: {col}',
                    'plot': json.dumps({
                        'data': [hist_data, box_data],
                        'layout': {
                            'title': f'Distribution Analysis of {col}',
                            'height': 500,
                            'showlegend': True,
                            'grid': {'rows': 1, 'columns': 2},
                            'annotations': [{
                                'text': f'Mean: {df[col].mean():.2f}<br>Median: {df[col].median():.2f}',
                                'showarrow': False,
                                'x': 0.5,
                                'y': 1.1,
                                'xref': 'paper',
                                'yref': 'paper'
                            }]
                        }
                    })
                })

        # 4. Category Distribution (Top categorical feature)
        if len(cat_cols) > 0:
            col = cat_cols[0]
            value_counts = df[col].value_counts()
            pie_data = {
                'type': 'pie',
                'labels': value_counts.index.tolist()[:8],  # Top 8 categories
                'values': value_counts.values.tolist()[:8],
                'hole': 0.4,
                'textinfo': 'label+percent',
                'textposition': 'outside'
            }
            visualizations.append({
                'type': 'category',
                'title': f'Category Distribution: {col}',
                'plot': json.dumps({
                    'data': [pie_data],
                    'layout': {
                        'title': f'Distribution of {col}',
                        'height': 500,
                        'showlegend': True,
                        'annotations': [{
                            'text': f'Total Categories: {len(value_counts)}',
                            'showarrow': False,
                            'x': 0.5,
                            'y': -0.1,
                            'xref': 'paper',
                            'yref': 'paper'
                        }]
                    }
                })
            })

        # 5. Scatter Plot (if enough numeric columns with strong correlation)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Find the most correlated pair
            max_corr = 0
            max_pair = None
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i,j]) > max_corr:
                        max_corr = abs(corr_matrix.iloc[i,j])
                        max_pair = (numeric_cols[i], numeric_cols[j])
            
            if max_pair and max_corr > 0.5:  # Only show if correlation is strong
                col1, col2 = max_pair
                scatter_data = {
                    'type': 'scatter',
                    'mode': 'markers',
                    'x': df[col1].tolist(),
                    'y': df[col2].tolist(),
                    'marker': {
                        'size': 8,
                        'opacity': 0.6,
                        'color': df[numeric_cols[0]].tolist(),
                        'colorscale': 'Viridis',
                        'showscale': True
                    }
                }
                visualizations.append({
                    'type': 'scatter',
                    'title': f'Feature Relationship: {col1} vs {col2}',
                    'plot': json.dumps({
                        'data': [scatter_data],
                        'layout': {
                            'title': f'{col1} vs {col2}<br>Correlation: {max_corr:.2f}',
                            'height': 500,
                            'xaxis': {'title': col1},
                            'yaxis': {'title': col2},
                            'hovermode': 'closest'
                        }
                    })
                })

        # 6. Parallel Coordinates (if enough numeric columns)
        if len(numeric_cols) >= 3:
            # Normalize the data for better visualization
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(df[numeric_cols])
            
            parallel_data = {
                'type': 'parcoords',
                'line': {
                    'color': df[numeric_cols[0]].tolist(),
                    'colorscale': 'Viridis'
                },
                'dimensions': [{
                    'label': col,
                    'values': normalized_data[:, i].tolist()
                } for i, col in enumerate(numeric_cols[:6])]  # Limit to 6 dimensions
            }
            visualizations.append({
                'type': 'parallel',
                'title': 'Multi-dimensional Feature Analysis',
                'plot': json.dumps({
                    'data': [parallel_data],
                    'layout': {
                        'title': 'Parallel Coordinates Plot',
                        'height': 500
                    }
                })
            })

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        # Return a basic visualization if there's an error
        visualizations.append({
            'type': 'error',
            'title': 'Error in Visualization',
            'plot': json.dumps({
                'data': [{'type': 'scatter', 'x': [], 'y': []}],
                'layout': {'title': 'Error creating visualization'}
            })
        })

    # Ensure we have at least 4 but no more than 6 visualizations
    return visualizations[:min(6, max(4, len(visualizations)))]

def prepare_dashboard_data(df: pd.DataFrame, insights: List[Dict[str, Any]], 
                         visualizations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepare data for dashboard presentation
    
    Args:
        df: Input DataFrame
        insights: Analysis insights
        visualizations: Visualization specifications
        
    Returns:
        Dashboard configuration and data
    """
    try:
        logger.info("Preparing dashboard data")
        
        # Ensure visualizations is a list
        if not isinstance(visualizations, list):
            visualizations = []
            
        # Ensure insights is a list
        if not isinstance(insights, list):
            insights = []
        
        # Safely get date range if purchase_date exists
        time_period = {"start": "N/A", "end": "N/A"}
        if 'purchase_date' in df.columns:
            try:
                df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
                if not df['purchase_date'].isna().all():
                    start_date = df['purchase_date'].min()
                    end_date = df['purchase_date'].max()
                    if pd.notna(start_date) and pd.notna(end_date):
                        time_period = {
                            "start": start_date.strftime('%Y-%m-%d'),
                            "end": end_date.strftime('%Y-%m-%d')
                        }
            except Exception as e:
                logger.warning(f"Error processing purchase dates: {str(e)}")
        
        # Ensure all visualizations have required fields
        formatted_visualizations = []
        for viz in visualizations:
            if isinstance(viz, dict):
                try:
                    formatted_viz = {
                        "title": str(viz.get("title", "Untitled Visualization")),
                        "type": str(viz.get("type", "overview")),
                        "plot": viz.get("plot", "{}")
                    }
                    formatted_visualizations.append(formatted_viz)
                except Exception as viz_error:
                    logger.warning(f"Error formatting visualization: {str(viz_error)}")
                    continue
        
        # Create sections with proper error handling
        overview_section = {
            "name": "Overview",
            "type": "metrics",
            "components": [
                {
                    "type": "metric",
                    "title": "Total Records",
                    "value": str(len(df))
                },
                {
                    "type": "metric",
                    "title": "Time Range",
                    "value": f"{time_period['start']} to {time_period['end']}"
                },
                {
                    "type": "metric",
                    "title": "Features",
                    "value": f"{len(df.columns)} columns"
                }
            ]
        }
        
        # Filter visualizations by type
        def get_visualizations_by_type(viz_type):
            return [viz for viz in formatted_visualizations 
                   if viz.get("type", "").lower() == viz_type.lower()]
        
        trends_section = {
            "name": "Trends",
            "type": "charts",
            "components": get_visualizations_by_type("trend")
        }
        
        comparison_section = {
            "name": "Comparison",
            "type": "charts",
            "components": get_visualizations_by_type("comparison")
        }
        
        predictions_section = {
            "name": "Predictions",
            "type": "charts",
            "components": get_visualizations_by_type("prediction")
        }
        
        # Add a default visualization to empty sections
        for section in [trends_section, comparison_section, predictions_section]:
            if not section["components"]:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No {section['name']} visualizations available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )
                fig.update_layout(template="plotly_white")
                section["components"].append({
                    "title": f"No {section['name']} Data",
                    "type": section["name"].lower(),
                    "plot": fig.to_json()
                })
        
        dashboard = {
            "title": "Advanced Sales Analytics Dashboard",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_summary": {
                "total_records": len(df),
                "time_period": time_period,
                "total_features": len(df.columns),
                "numeric_features": len(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns)
            },
            "insights": insights,
            "visualizations": formatted_visualizations,
            "layout": {
                "sections": [
                    overview_section,
                    trends_section,
                    comparison_section,
                    predictions_section
                ]
            }
        }
        
        logger.info("Dashboard data prepared successfully")
        return dashboard
        
    except Exception as e:
        logger.error(f"Error preparing dashboard data: {str(e)}")
        # Return a minimal dashboard on error
        return {
            "title": "Advanced Sales Analytics Dashboard",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_summary": {
                "total_records": 0,
                "time_period": {"start": "N/A", "end": "N/A"},
                "total_features": 0,
                "numeric_features": 0,
                "categorical_features": 0
            },
            "insights": [],
            "visualizations": [],
            "layout": {
                "sections": [
                    {
                        "name": "Error",
                        "type": "message",
                        "components": [
                            {
                                "type": "message",
                                "title": "Error",
                                "value": "An error occurred while preparing the dashboard"
                            }
                        ]
                    }
                ]
            }
        }

def analyze_product_recommendations(file_path: str) -> Dict[str, Any]:
    """
    Main function to analyze product recommendations data
    
    Args:
        file_path: Path to the input CSV file
        
    Returns:
        Complete analysis results and dashboard configuration
    """
    try:
        # 1. Load data
        df = load_data(file_path)
        
        # 2. Explore data
        exploration_results = explore_data(df)
        
        # 3. Clean data
        df_clean, cleaning_report = clean_data(df)
        
        # 4. Analyze data
        analysis_results = analyze_data(df_clean)
        
        # 5. Create visualizations
        visualizations = create_visualizations(df_clean, analysis_results["insights"])
        
        # 6. Prepare dashboard
        dashboard = prepare_dashboard_data(df_clean, analysis_results["insights"], visualizations)
        
        # Compile all results
        results = {
            "exploration": exploration_results,
            "cleaning": cleaning_report,
            "analysis": analysis_results,
            "dashboard": dashboard
        }
        
        # Pre-process data to handle special types
        def preprocess_data(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: preprocess_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [preprocess_data(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(preprocess_data(item) for item in obj)
            elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
                return None
            return obj
        
        # First preprocess the data
        preprocessed_results = preprocess_data(results)
        
        # Then convert to JSON-serializable format
        serializable_results = convert_to_serializable(preprocessed_results)
        
        return serializable_results
        
    except Exception as e:
        logger.error(f"Error in product recommendations analysis: {str(e)}")
        raise 