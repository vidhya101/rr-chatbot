import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import current_app, Blueprint, request, jsonify
import logging
import json
from datetime import datetime
import traceback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64
from scipy import stats
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fastapi.responses import StreamingResponse
import asyncio
from datetime import timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Create a Blueprint for visualization routes
visualization_bp = Blueprint('visualization', __name__)

def get_visualization_folder():
    """Get the folder where visualizations should be stored"""
    viz_folder = os.path.join(current_app.config.get('UPLOAD_FOLDER', 'uploads'), 'visualizations')
    os.makedirs(viz_folder, exist_ok=True)
    return viz_folder

def read_data_file(file_path):
    """Read a data file into a pandas DataFrame"""
    try:
        # Determine file type from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext == '.txt' or file_ext == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        else:
            # Try to infer the format
            try:
                return pd.read_csv(file_path)
            except:
                try:
                    return pd.read_excel(file_path)
                except:
                    raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        logger.error(f"Error reading data file: {str(e)}")
        raise ValueError(f"Failed to read data file: {str(e)}")

def get_data_summary(df):
    """Get a summary of the data"""
    try:
        # Get basic info
        rows, cols = df.shape
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get basic statistics
        stats = {
            'rows': rows,
            'columns': cols,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'missing_values': df.isnull().sum().sum(),
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        return {
            'rows': 0,
            'columns': 0,
            'numeric_columns': [],
            'categorical_columns': [],
            'missing_values': 0,
            'column_types': {}
        }

def clean_and_transform_data(df):
    """Clean and transform data for visualization"""
    # Handle missing values
    for col in df.columns:
        if df[col].dtype.kind in 'ifc':  # integer, float, complex
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna('Unknown')
    
    # Normalize numerical columns for better visualization
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def apply_ml_prediction(df, target_col):
    """Apply simple linear regression to predict values"""
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        return None
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns
    features = [col for col in numeric_cols if col != target_col]
    
    if not features:
        return None
    
    # Create X and y
    X = df[features].values
    y = df[target_col].values
    
    # Split data
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'predictions': y_pred.tolist(),
        'actual': y_test.tolist(),
        'mse': mse,
        'r2': r2
    }

def generate_visualization(file_path, viz_type='auto', params=None):
    """Generate a visualization for a data file"""
    try:
        # Default params
        if params is None:
            params = {}
        
        # Read the data
        df = read_data_file(file_path)
        
        # Get visualization folder
        viz_folder = get_visualization_folder()
        
        # Generate a unique filename
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        viz_filename = f"viz_{timestamp}_{unique_id}.png"
        viz_path = os.path.join(viz_folder, viz_filename)
        
        # Set up the figure
        plt.figure(figsize=(10, 6))
        
        # Clean data
        df = clean_and_transform_data(df)
        
        # Determine visualization type if auto
        if viz_type == 'auto':
            if 'column' in params:
                column = params['column']
                if pd.api.types.is_numeric_dtype(df[column]):
                    viz_type = 'histogram'
                else:
                    viz_type = 'bar'
            elif len(df.select_dtypes(include=['number']).columns) >= 2:
                viz_type = 'scatter'
            elif len(df.select_dtypes(include=['number']).columns) == 1:
                viz_type = 'histogram'
            else:
                viz_type = 'bar'
        
        # Generate visualization based on type
        title = params.get('title', f'{viz_type.capitalize()} Visualization')
        
        if viz_type == 'heatmap':
            # Generate correlation heatmap
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) < 2:
                return {
                    'success': False,
                    'error': 'Not enough numeric columns for a heatmap'
                }
            
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title(title)
            
        elif viz_type == 'histogram':
            # Generate histogram
            column = params.get('column')
            if not column:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    return {
                        'success': False,
                        'error': 'No numeric columns for histogram'
                    }
                column = numeric_cols[0]
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                return {
                    'success': False,
                    'error': f'Column {column} is not numeric'
                }
            
            sns.histplot(df[column], kde=True)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel('Frequency')
            
        elif viz_type == 'bar':
            # Generate bar chart
            column = params.get('column')
            if not column:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) == 0:
                    return {
                        'success': False,
                        'error': 'No categorical columns for bar chart'
                    }
                column = categorical_cols[0]
            
            # Get value counts and limit to top 10
            value_counts = df[column].value_counts().head(10)
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
        elif viz_type == 'scatter':
            # Generate scatter plot
            x_column = params.get('x_column')
            y_column = params.get('y_column')
            
            if not x_column or not y_column:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) < 2:
                    return {
                        'success': False,
                        'error': 'Not enough numeric columns for scatter plot'
                    }
                x_column = numeric_cols[0]
                y_column = numeric_cols[1]
            
            if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
                return {
                    'success': False,
                    'error': 'Both columns must be numeric for scatter plot'
                }
            
            sns.scatterplot(x=df[x_column], y=df[y_column])
            plt.title(title)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        else:
            return {
                'success': False,
                'error': f'Unsupported visualization type: {viz_type}'
            }
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        
        # Get the URL for the visualization
        viz_url = f"/api/visualization/visualizations/{viz_filename}"
        
        return {
            'success': True,
            'visualization': {
                'url': viz_url,
                'title': title,
                'type': viz_type,
                'filename': viz_filename
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def generate_plotly_visualization(df, viz_type, params=None):
    """Generate a Plotly visualization for a DataFrame"""
    if params is None:
        params = {}
    
    # Get chart title
    title = params.get('title', f'{viz_type.capitalize()} Chart')
    
    # Create figure based on visualization type
    if viz_type == 'bar':
        column = params.get('column')
        if not column:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) == 0:
                return None
            column = categorical_cols[0]
        
        # Get value counts and limit to top 10
        value_counts = df[column].value_counts().head(10)
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=title,
            labels={'x': column, 'y': 'Count'}
        )
        
    elif viz_type == 'line':
        x_column = params.get('x_column')
        y_column = params.get('y_column')
        
        if not x_column or not y_column:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return None
            x_column = numeric_cols[0]
            y_column = numeric_cols[1]
        
        fig = px.line(
            df, 
            x=x_column, 
            y=y_column,
            title=title,
            labels={'x': x_column, 'y': y_column}
        )
        
    elif viz_type == 'scatter':
        x_column = params.get('x_column')
        y_column = params.get('y_column')
        
        if not x_column or not y_column:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return None
            x_column = numeric_cols[0]
            y_column = numeric_cols[1]
        
        fig = px.scatter(
            df, 
            x=x_column, 
            y=y_column,
            title=title,
            labels={'x': x_column, 'y': y_column}
        )
        
    elif viz_type == 'pie':
        column = params.get('column')
        if not column:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) == 0:
                return None
            column = categorical_cols[0]
        
        # Get value counts and limit to top 10
        value_counts = df[column].value_counts().head(10)
        fig = px.pie(
            names=value_counts.index, 
            values=value_counts.values,
            title=title
        )
        
    elif viz_type == 'histogram':
        column = params.get('column')
        if not column:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None
            column = numeric_cols[0]
        
        fig = px.histogram(
            df, 
            x=column,
            title=title,
            labels={'x': column, 'y': 'Count'}
        )
        
    elif viz_type == 'heatmap':
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) < 2:
            return None
        
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            title=title,
            labels={'color': 'Correlation'}
        )
        
    elif viz_type == 'box':
        column = params.get('column')
        if not column:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None
            column = numeric_cols[0]
        
        fig = px.box(
            df, 
            y=column,
            title=title,
            labels={'y': column}
        )
        
    else:
        return None
    
    # Update layout
    fig.update_layout(
        template='plotly_dark' if params.get('dark_mode', False) else 'plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Convert to JSON
    return json.loads(fig.to_json())

def generate_dashboard(file_path):
    """Generate a comprehensive dashboard for a data file"""
    try:
        # Read the data
        df = read_data_file(file_path)
        
        # Clean data
        df = clean_and_transform_data(df)
        
        # Get data summary
        stats = get_data_summary(df)
        
        # List to store visualizations
        visualizations = []
        plotly_charts = []
        
        # 1. Generate correlation heatmap for numeric columns
        if len(df.select_dtypes(include=['number']).columns) >= 2:
            heatmap_result = generate_visualization(
                file_path, 
                viz_type='heatmap',
                params={'title': 'Correlation Heatmap'}
            )
            if heatmap_result.get('success', False):
                visualizations.append(heatmap_result['visualization'])
            
            # Add Plotly heatmap
            heatmap_plotly = generate_plotly_visualization(
                df,
                'heatmap',
                {'title': 'Correlation Heatmap'}
            )
            if heatmap_plotly:
                plotly_charts.append({
                    'type': 'heatmap',
                    'title': 'Correlation Heatmap',
                    'data': heatmap_plotly
                })
        
        # 2. Generate histograms for numeric columns (up to 5)
        for col in df.select_dtypes(include=['number']).columns[:5]:
            hist_result = generate_visualization(
                file_path,
                viz_type='histogram',
                params={'column': col, 'title': f'Distribution of {col}'}
            )
            if hist_result.get('success', False):
                visualizations.append(hist_result['visualization'])
            
            # Add Plotly histogram
            hist_plotly = generate_plotly_visualization(
                df,
                'histogram',
                {'column': col, 'title': f'Distribution of {col}'}
            )
            if hist_plotly:
                plotly_charts.append({
                    'type': 'histogram',
                    'title': f'Distribution of {col}',
                    'data': hist_plotly
                })
        
        # 3. Generate bar charts for categorical columns (up to 5)
        for col in df.select_dtypes(include=['object', 'category']).columns[:5]:
            if df[col].nunique() <= 20:  # Only if there aren't too many categories
                bar_result = generate_visualization(
                    file_path,
                    viz_type='bar',
                    params={'column': col, 'title': f'Frequency of {col}'}
                )
                if bar_result.get('success', False):
                    visualizations.append(bar_result['visualization'])
                
                # Add Plotly bar chart
                bar_plotly = generate_plotly_visualization(
                    df,
                    'bar',
                    {'column': col, 'title': f'Frequency of {col}'}
                )
                if bar_plotly:
                    plotly_charts.append({
                        'type': 'bar',
                        'title': f'Frequency of {col}',
                        'data': bar_plotly
                    })
        
        # 4. Generate scatter plots for pairs of numeric columns (up to 3 pairs)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols) - 1)):
                scatter_result = generate_visualization(
                    file_path,
                    viz_type='scatter',
                    params={
                        'x_column': numeric_cols[i],
                        'y_column': numeric_cols[i+1],
                        'title': f'{numeric_cols[i]} vs {numeric_cols[i+1]}'
                    }
                )
                if scatter_result.get('success', False):
                    visualizations.append(scatter_result['visualization'])
                
                # Add Plotly scatter plot
                scatter_plotly = generate_plotly_visualization(
                    df,
                    'scatter',
                    {
                        'x_column': numeric_cols[i],
                        'y_column': numeric_cols[i+1],
                        'title': f'{numeric_cols[i]} vs {numeric_cols[i+1]}'
                    }
                )
                if scatter_plotly:
                    plotly_charts.append({
                        'type': 'scatter',
                        'title': f'{numeric_cols[i]} vs {numeric_cols[i+1]}',
                        'data': scatter_plotly
                    })
        
        # 5. Apply ML prediction if possible
        ml_insights = None
        if len(numeric_cols) >= 2:
            target_col = numeric_cols[0]
            ml_insights = apply_ml_prediction(df, target_col)
        
        # Get top 5 rows for preview
        top5 = df.head(5).to_dict(orient='records')
        
        # Get column names and types
        columns = [
            {'name': col, 'type': 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'}
            for col in df.columns
        ]
        
        return {
            'success': True,
            'visualizations': visualizations,
            'plotly_charts': plotly_charts,
            'stats': stats,
            'top5': top5,
            'columns': columns,
            'ml_insights': ml_insights
        }
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def suggest_chart_type(df):
    """Suggest the best chart type based on the data"""
    if df is None or len(df.columns) == 0:
        return 'bar'  # Default
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) >= 2:
        return 'scatter'
    elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
        return 'bar'
    elif len(numeric_cols) == 1:
        return 'histogram'
    elif len(categorical_cols) > 0:
        return 'pie'
    
    return 'bar'  # Default fallback 

# Additional configuration models
class AdvancedVisualizationConfig(BaseModel):
    plot_type: str
    dimensions: Optional[List[str]] = None
    color_by: Optional[str] = None
    size_by: Optional[str] = None
    facet_by: Optional[str] = None
    animation_frame: Optional[str] = None
    statistical_tests: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None

class InteractiveConfig(BaseModel):
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    enable_lasso: bool = True
    hover_data: Optional[List[str]] = None
    click_data: Optional[List[str]] = None
    animation_opts: Optional[Dict[str, Any]] = None

class RealTimeConfig(BaseModel):
    update_interval: int
    window_size: Optional[int] = None
    aggregation_window: Optional[str] = None
    alert_conditions: Optional[Dict[str, Any]] = None

@visualization_bp.route('/advanced_visualization', methods=['POST'])
def create_advanced_visualization():
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        config = AdvancedVisualizationConfig(**data.get('config', {}))
        
        # Your visualization logic here
        result = {
            'success': True,
            'chart_data': {},  # Add your chart data here
            'message': 'Chart created successfully'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@visualization_bp.route('/interactive_chart', methods=['POST'])
def create_interactive_chart():
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        config = data.get('config', {})
        interactive_config = InteractiveConfig(**data.get('interactive_config', {}))
        
        # Your interactive chart logic here
        result = {
            'success': True,
            'chart_data': {},  # Add your chart data here
            'message': 'Interactive chart created successfully'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@visualization_bp.route('/realtime_chart', methods=['POST'])
def create_realtime_chart():
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        config = data.get('config', {})
        realtime_config = RealTimeConfig(**data.get('realtime_config', {}))
        
        # Your realtime chart logic here
        result = {
            'success': True,
            'chart_data': {},  # Add your chart data here
            'message': 'Realtime chart created successfully'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@visualization_bp.route('/export_chart/<chart_id>', methods=['GET'])
def export_chart(chart_id):
    try:
        format = request.args.get('format', 'png')
        
        # Your chart export logic here
        result = {
            'success': True,
            'export_url': f'/exports/{chart_id}.{format}',
            'message': 'Chart exported successfully'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@visualization_bp.route('/chart_statistics/<chart_id>', methods=['POST'])
def calculate_chart_statistics(chart_id):
    try:
        # Your statistics calculation logic here
        result = {
            'success': True,
            'statistics': {},  # Add your statistics here
            'message': 'Statistics calculated successfully'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400 