import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import current_app
import logging
import json
from datetime import datetime
import traceback
from typing import List, Dict, Any
from utils.db_utils import get_db_connection

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        self.db = get_db_connection()
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure the visualizations table exists"""
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS visualizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                type TEXT,
                plot TEXT,
                source TEXT,
                file_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.db.commit()

    def save_visualizations(self, visualizations: List[Dict[str, Any]], source: str = None, file_id: str = None) -> List[Dict[str, Any]]:
        """Save visualizations to the database"""
        saved_visualizations = []
        try:
            for viz in visualizations:
                cursor = self.db.execute('''
                    INSERT INTO visualizations (title, type, plot, source, file_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    viz.get('title', 'Untitled'),
                    viz.get('type', 'unknown'),
                    json.dumps(viz.get('plot', {})),
                    source,
                    file_id
                ))
                viz_id = cursor.lastrowid
                
                # Fetch the saved visualization
                saved_viz = self.db.execute('''
                    SELECT * FROM visualizations WHERE id = ?
                ''', (viz_id,)).fetchone()
                
                if saved_viz:
                    saved_visualizations.append({
                        'id': saved_viz[0],
                        'title': saved_viz[1],
                        'type': saved_viz[2],
                        'plot': json.loads(saved_viz[3]),
                        'source': saved_viz[4],
                        'file_id': saved_viz[5],
                        'created_at': saved_viz[6]
                    })
            
            self.db.commit()
            return saved_visualizations
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to save visualizations: {str(e)}")

    def get_all_visualizations(self) -> List[Dict[str, Any]]:
        """Retrieve all visualizations from the database"""
        try:
            cursor = self.db.execute('SELECT * FROM visualizations ORDER BY created_at DESC')
            visualizations = []
            
            for row in cursor.fetchall():
                visualizations.append({
                    'id': row[0],
                    'title': row[1],
                    'type': row[2],
                    'plot': json.loads(row[3]),
                    'source': row[4],
                    'file_id': row[5],
                    'created_at': row[6]
                })
            
            return visualizations
        except Exception as e:
            raise Exception(f"Failed to retrieve visualizations: {str(e)}")

    def get_visualization_by_id(self, viz_id: int) -> Dict[str, Any]:
        """Retrieve a specific visualization by ID"""
        try:
            cursor = self.db.execute('SELECT * FROM visualizations WHERE id = ?', (viz_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception(f"Visualization with ID {viz_id} not found")
            
            return {
                'id': row[0],
                'title': row[1],
                'type': row[2],
                'plot': json.loads(row[3]),
                'source': row[4],
                'file_id': row[5],
                'created_at': row[6]
            }
        except Exception as e:
            raise Exception(f"Failed to retrieve visualization: {str(e)}")

    def delete_visualization(self, viz_id: int) -> bool:
        """Delete a visualization by ID"""
        try:
            self.db.execute('DELETE FROM visualizations WHERE id = ?', (viz_id,))
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to delete visualization: {str(e)}")

visualization_service = VisualizationService()

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
        
        # Generate the visualization based on type
        title = params.get('title', 'Data Visualization')
        
        if viz_type == 'auto':
            # Determine the best visualization based on data
            if len(df.select_dtypes(include=['number']).columns) >= 2:
                viz_type = 'scatter'
            elif len(df.select_dtypes(include=['number']).columns) >= 1:
                viz_type = 'histogram'
            else:
                viz_type = 'bar'
        
        # Generate specific visualization
        if viz_type == 'scatter':
            x_col = params.get('x_column', df.select_dtypes(include=['number']).columns[0])
            y_col = params.get('y_column', df.select_dtypes(include=['number']).columns[1] if len(df.select_dtypes(include=['number']).columns) > 1 else df.select_dtypes(include=['number']).columns[0])
            
            plt.scatter(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(title or f'Scatter Plot: {x_col} vs {y_col}')
            
            description = f"Scatter plot showing the relationship between {x_col} and {y_col}"
            
        elif viz_type == 'histogram':
            col = params.get('column', df.select_dtypes(include=['number']).columns[0])
            bins = params.get('bins', 20)
            
            plt.hist(df[col], bins=bins)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title(title or f'Histogram of {col}')
            
            description = f"Histogram showing the distribution of {col} with {bins} bins"
            
        elif viz_type == 'bar':
            col = params.get('column', df.select_dtypes(include=['object', 'category']).columns[0] if len(df.select_dtypes(include=['object', 'category']).columns) > 0 else df.columns[0])
            
            # Get value counts and sort
            value_counts = df[col].value_counts()
            limit = params.get('limit', 10)
            if len(value_counts) > limit:
                value_counts = value_counts.head(limit)
            
            plt.bar(value_counts.index, value_counts.values)
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.title(title or f'Bar Chart of {col}')
            plt.xticks(rotation=45)
            
            description = f"Bar chart showing the frequency of values in {col}"
            
        elif viz_type == 'line':
            x_col = params.get('x_column', df.columns[0])
            y_col = params.get('y_column', df.select_dtypes(include=['number']).columns[0])
            
            plt.plot(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(title or f'Line Plot: {y_col} over {x_col}')
            
            description = f"Line plot showing {y_col} over {x_col}"
            
        elif viz_type == 'heatmap':
            # Select numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            
            # Calculate correlation matrix
            corr = numeric_df.corr()
            
            # Create heatmap
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title(title or 'Correlation Heatmap')
            
            description = "Heatmap showing correlations between numeric variables"
            
        elif viz_type == 'boxplot':
            col = params.get('column', df.select_dtypes(include=['number']).columns[0])
            group_by = params.get('group_by', None)
            
            if group_by and group_by in df.columns:
                sns.boxplot(x=group_by, y=col, data=df)
                plt.title(title or f'Box Plot of {col} by {group_by}')
                description = f"Box plot showing the distribution of {col} grouped by {group_by}"
            else:
                sns.boxplot(y=col, data=df)
                plt.title(title or f'Box Plot of {col}')
                description = f"Box plot showing the distribution of {col}"
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        
        # Create URL for the visualization
        viz_url = f"/api/visualization/visualizations/{viz_filename}"
        
        return {
            'success': True,
            'visualization': {
                'type': viz_type,
                'title': title,
                'description': description,
                'url': viz_url,
                'file_path': viz_path
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def generate_dashboard(file_path):
    """Generate a comprehensive dashboard for a data file"""
    try:
        # Read the data
        df = read_data_file(file_path)
        
        # Get data summary
        stats = get_data_summary(df)
        
        # List to store visualizations
        visualizations = []
        
        # 1. Generate correlation heatmap for numeric columns
        if len(df.select_dtypes(include=['number']).columns) >= 2:
            heatmap_result = generate_visualization(
                file_path, 
                viz_type='heatmap',
                params={'title': 'Correlation Heatmap'}
            )
            if heatmap_result.get('success', False):
                visualizations.append(heatmap_result['visualization'])
        
        # 2. Generate histograms for numeric columns (up to 5)
        for col in df.select_dtypes(include=['number']).columns[:5]:
            hist_result = generate_visualization(
                file_path,
                viz_type='histogram',
                params={'column': col, 'title': f'Distribution of {col}'}
            )
            if hist_result.get('success', False):
                visualizations.append(hist_result['visualization'])
        
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
        
        return {
            'success': True,
            'visualizations': visualizations,
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        } 