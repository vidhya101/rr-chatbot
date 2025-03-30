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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

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

def create_visualizations(df):
    """Create comprehensive set of visualizations based on data characteristics"""
    MIN_VISUALIZATIONS = 12
    MAX_VISUALIZATIONS = 15
    
    print(f"Starting visualization generation for dataframe with shape {df.shape}")
    
    visualizations = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")

    # Helper function to convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return [convert_to_serializable(x) for x in obj.tolist()]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, pd.Series):
            return [convert_to_serializable(x) for x in obj.tolist()]
        elif isinstance(obj, list):
            return [convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif pd.isna(obj):
            return None
        return obj

    # Core visualizations that must be generated
    core_visualizations = [
        {
            'type': 'correlation_heatmap',
            'generator': lambda: create_correlation_heatmap(df, numeric_cols),
            'title': 'Feature Correlation Analysis'
        },
        {
            'type': 'distribution_histogram',
            'generator': lambda: create_distribution_histogram(df, numeric_cols),
            'title': 'Distribution Analysis'
        },
        {
            'type': 'box_plot',
            'generator': lambda: create_box_plot_matrix(df, numeric_cols),
            'title': 'Box Plot Analysis'
        },
        {
            'type': 'scatter_plot',
            'generator': lambda: create_scatter_plot(df, numeric_cols),
            'title': 'Feature Relationships'
        },
        {
            'type': 'parallel_coordinates',
            'generator': lambda: create_parallel_coordinates(df, numeric_cols),
            'title': 'Multi-dimensional Feature Analysis'
        },
        {
            'type': 'violin_plot',
            'generator': lambda: create_violin_plot_matrix(df, numeric_cols),
            'title': 'Violin Distribution Analysis'
        },
        {
            'type': 'kde_plot',
            'generator': lambda: create_kde_plot(df, numeric_cols),
            'title': 'Kernel Density Estimation'
        },
        {
            'type': 'heatmap',
            'generator': lambda: create_correlation_heatmap(df, numeric_cols),
            'title': 'Data Heatmap Analysis'
        },
        {
            'type': '3d_scatter',
            'generator': lambda: create_3d_scatter(df, numeric_cols),
            'title': '3D Feature Analysis'
        },
        {
            'type': 'joint_plot',
            'generator': lambda: create_joint_plot(df, numeric_cols),
            'title': 'Joint Distribution Analysis'
        },
        {
            'type': 'hexbin_plot',
            'generator': lambda: create_hexbin_plot(df, numeric_cols),
            'title': 'Hexbin Analysis'
        }
    ]

    # Generate all core visualizations
    for viz_config in core_visualizations:
        try:
            print(f"Generating visualization: {viz_config['type']}...")
            result = viz_config['generator']()
            if result is not None:
                # Process the figure data to ensure it's serializable
                if 'figure' in result:
                    if isinstance(result['figure'], dict):
                        # Process data array
                        if 'data' in result['figure']:
                            result['figure']['data'] = [
                                {k: convert_to_serializable(v) for k, v in trace.items()}
                                for trace in result['figure']['data']
                            ]
                        # Process layout
                        if 'layout' in result['figure']:
                            result['figure']['layout'] = convert_to_serializable(result['figure']['layout'])
                
                visualizations.append({
                    "type": viz_config['type'],
                    "title": viz_config['title'],
                    "figure": result['figure'],
                    "insight": convert_to_serializable(result.get('insight', ''))
                })
                print(f"Successfully generated {viz_config['type']}")
            else:
                print(f"Warning: {viz_config['type']} generator returned None")
        except Exception as e:
            print(f"Error generating {viz_config['type']}: {str(e)}")
            continue

    # If we still don't have enough visualizations, add categorical visualizations
    if len(visualizations) < MIN_VISUALIZATIONS and len(categorical_cols) > 0:
        for col in categorical_cols:
            if len(visualizations) >= MIN_VISUALIZATIONS:
                break
            try:
                # Add pie chart
                pie_result = create_pie_chart(df, col)
                if pie_result is not None:
                    # Process the figure data
                    if isinstance(pie_result['figure'], dict):
                        if 'data' in pie_result['figure']:
                            pie_result['figure']['data'] = [
                                {k: convert_to_serializable(v) for k, v in trace.items()}
                                for trace in pie_result['figure']['data']
                            ]
                        if 'layout' in pie_result['figure']:
                            pie_result['figure']['layout'] = convert_to_serializable(pie_result['figure']['layout'])
                    
                    visualizations.append({
                        "type": "pie_chart",
                        "title": f"Category Distribution: {col}",
                        "figure": pie_result['figure'],
                        "insight": convert_to_serializable(pie_result.get('insight', ''))
                    })

                # Add treemap
                treemap_result = create_treemap(df, col)
                if treemap_result is not None:
                    # Process the figure data
                    if isinstance(treemap_result['figure'], dict):
                        if 'data' in treemap_result['figure']:
                            treemap_result['figure']['data'] = [
                                {k: convert_to_serializable(v) for k, v in trace.items()}
                                for trace in treemap_result['figure']['data']
                            ]
                        if 'layout' in treemap_result['figure']:
                            treemap_result['figure']['layout'] = convert_to_serializable(treemap_result['figure']['layout'])
                    
                    visualizations.append({
                        "type": "treemap",
                        "title": f"Hierarchical View: {col}",
                        "figure": treemap_result['figure'],
                        "insight": convert_to_serializable(treemap_result.get('insight', ''))
                    })
            except Exception as e:
                print(f"Error generating categorical visualization for {col}: {str(e)}")
                continue

    # If we still don't have enough visualizations, add detailed column analyses
    if len(visualizations) < MIN_VISUALIZATIONS:
        for col in numeric_cols:
            if len(visualizations) >= MIN_VISUALIZATIONS:
                break
            try:
                result = create_detailed_column_analysis(df[col])
                if result is not None:
                    # Process the figure data
                    if isinstance(result['figure'], dict):
                        if 'data' in result['figure']:
                            result['figure']['data'] = [
                                {k: convert_to_serializable(v) for k, v in trace.items()}
                                for trace in result['figure']['data']
                            ]
                        if 'layout' in result['figure']:
                            result['figure']['layout'] = convert_to_serializable(result['figure']['layout'])
                    
                    visualizations.append({
                        "type": "detailed_analysis",
                        "title": f"Detailed Analysis of {col}",
                        "figure": result['figure'],
                        "insight": convert_to_serializable(result.get('insight', ''))
                    })
            except Exception as e:
                print(f"Error creating detailed analysis for {col}: {str(e)}")
                continue

    # Final check to ensure all data is serializable
    visualizations = convert_to_serializable(visualizations[:MAX_VISUALIZATIONS])
    print(f"Successfully generated {len(visualizations)} visualizations")
    return visualizations

def create_correlation_heatmap(df, numeric_cols=None):
    """Create correlation heatmap for numeric columns"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        corr_matrix = df[numeric_cols].corr()
        max_corr = corr_matrix.max().max()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Feature Correlation Analysis',
            height=600
        )
        
        return {
            'figure': fig.to_dict(),
            'insight': f'Shows correlation strength between features. Maximum correlation: {max_corr:.2f}'
        }
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return None

def create_distribution_histogram(df, numeric_cols=None):
    """Create distribution histograms for numeric columns"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
        
        fig = make_subplots(rows=2, cols=3, subplot_titles=numeric_cols)
        row, col = 1, 1
        
        for col_name in numeric_cols:
            fig.add_trace(
                go.Histogram(x=df[col_name], name=col_name),
                row=row, col=col
            )
            col += 1
            if col > 3:
                row += 1
                col = 1
        
        fig.update_layout(
            title='Distribution Analysis',
            showlegend=False,
            height=800
        )
        
        skewness = df[numeric_cols].skew()
        max_skew = abs(skewness).max()
        
        return {
            'figure': fig.to_dict(),
            'insight': f'Shows data distributions. Maximum skewness: {max_skew:.2f}'
        }
    except Exception as e:
        logger.error(f"Error creating distribution histogram: {str(e)}")
        return None

def create_box_plot_matrix(df, numeric_cols=None):
    """Create box plots for numeric columns"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
        
        fig = go.Figure()
        for col in numeric_cols:
            fig.add_trace(go.Box(y=df[col], name=col))
        
        fig.update_layout(
            title='Box Plot Analysis',
            height=600
        )
        
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        
        max_outliers = max(outliers.values())
        
        return {
            'figure': fig.to_dict(),
            'insight': f'Shows data distribution and outliers. Maximum outliers in a feature: {max_outliers}'
        }
    except Exception as e:
        logger.error(f"Error creating box plot matrix: {str(e)}")
        return None

def create_scatter_plot(df, numeric_cols=None):
    """Create scatter plot matrix for numeric columns"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4]
        
        fig = px.scatter_matrix(
            df[numeric_cols],
            dimensions=numeric_cols,
            title='Feature Relationships'
        )
        
        fig.update_layout(height=800)
        
        return {
            'figure': fig.to_dict(),
            'insight': 'Shows relationships between pairs of numeric features'
        }
    except Exception as e:
        logger.error(f"Error creating scatter plot: {str(e)}")
        return None

def create_parallel_coordinates(df, numeric_cols=None):
    """Create parallel coordinates plot"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
        
        fig = px.parallel_coordinates(
            df[numeric_cols],
            title='Multi-dimensional Feature Analysis'
        )
        
        fig.update_layout(height=600)
        
        return {
            'figure': fig.to_dict(),
            'insight': 'Shows relationships across multiple dimensions simultaneously'
        }
    except Exception as e:
        logger.error(f"Error creating parallel coordinates: {str(e)}")
        return None

def create_violin_plot_matrix(df, numeric_cols=None):
    """Create violin plots for numeric columns"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
        
        fig = go.Figure()
        for col in numeric_cols:
            fig.add_trace(go.Violin(y=df[col], name=col, box_visible=True))
        
        fig.update_layout(
            title='Violin Distribution Analysis',
            height=600
        )
        
        return {
            'figure': fig.to_dict(),
            'insight': 'Shows detailed distribution shape for each feature'
        }
    except Exception as e:
        logger.error(f"Error creating violin plot matrix: {str(e)}")
        return None

def create_kde_plot(df, numeric_cols=None):
    """Create KDE plots for numeric columns"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
        
        fig = go.Figure()
        for col in numeric_cols:
            kde = np.histogram(df[col].dropna(), bins=50, density=True)
            fig.add_trace(go.Scatter(x=kde[1][:-1], y=kde[0], name=col, mode='lines'))
        
        fig.update_layout(
            title='Kernel Density Estimation',
            height=600
        )
        
        return {
            'figure': fig.to_dict(),
            'insight': 'Shows smoothed distribution estimates for each feature'
        }
    except Exception as e:
        logger.error(f"Error creating kde plot: {str(e)}")
        return None

def create_3d_scatter(df, numeric_cols=None):
    """Create 3D scatter plot"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:3]
        
        if len(numeric_cols) >= 3:
            fig = px.scatter_3d(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                z=numeric_cols[2],
                title='3D Feature Analysis'
            )
            
            fig.update_layout(height=800)
            
            return {
                'figure': fig.to_dict(),
                'insight': 'Shows relationships between three numeric features in 3D space'
            }
    except Exception as e:
        logger.error(f"Error creating 3d scatter: {str(e)}")
        return None

def create_joint_plot(df, numeric_cols=None):
    """Create joint distribution plot"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:2]
        
        if len(numeric_cols) >= 2:
            fig = make_subplots(
                rows=2, cols=2,
                row_heights=[0.7, 0.3],
                column_widths=[0.7, 0.3],
                vertical_spacing=0.02,
                horizontal_spacing=0.02
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]], mode='markers'),
                row=1, col=1
            )
            
            # Histogram for x
            fig.add_trace(
                go.Histogram(x=df[numeric_cols[0]]),
                row=2, col=1
            )
            
            # Histogram for y
            fig.add_trace(
                go.Histogram(y=df[numeric_cols[1]]),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Joint Distribution Analysis',
                height=800,
                showlegend=False
            )
            
            return {
                'figure': fig.to_dict(),
                'insight': 'Shows joint and marginal distributions for two features'
            }
    except Exception as e:
        logger.error(f"Error creating joint plot: {str(e)}")
        return None

def create_hexbin_plot(df, numeric_cols=None):
    """Create hexbin plot"""
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:2]
        
        if len(numeric_cols) >= 2:
            fig = px.density_heatmap(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                marginal_x='histogram',
                marginal_y='histogram',
                title='Hexbin Analysis'
            )
            
            fig.update_layout(height=800)
            
            return {
                'figure': fig.to_dict(),
                'insight': 'Shows density of point clusters between two features'
            }
    except Exception as e:
        logger.error(f"Error creating hexbin plot: {str(e)}")
        return None

def create_pie_chart(df, categorical_col):
    """Create pie chart for categorical column"""
    try:
        value_counts = df[categorical_col].value_counts()
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Category Distribution: {categorical_col}'
        )
        
        fig.update_layout(height=600)
        
        return {
            'figure': fig.to_dict(),
            'insight': f'Shows distribution of categories in {categorical_col}'
        }
    except Exception as e:
        logger.error(f"Error creating pie chart: {str(e)}")
        return None

def create_treemap(df, categorical_col):
    """Create treemap for categorical column"""
    try:
        value_counts = df[categorical_col].value_counts()
        fig = px.treemap(
            names=value_counts.index,
            parents=[''] * len(value_counts),
            values=value_counts.values,
            title=f'Hierarchical View: {categorical_col}'
        )
        
        fig.update_layout(height=600)
        
        return {
            'figure': fig.to_dict(),
            'insight': f'Shows hierarchical view of categories in {categorical_col}'
        }
    except Exception as e:
        logger.error(f"Error creating treemap: {str(e)}")
        return None

def create_detailed_column_analysis(df_col):
    """Create detailed analysis visualization for a single column"""
    try:
        fig = make_subplots(rows=2, cols=2)
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df_col, name='Distribution'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df_col, name='Box Plot'),
            row=1, col=2
        )
        
        # KDE
        kde = np.histogram(df_col.dropna(), bins=50, density=True)
        fig.add_trace(
            go.Scatter(x=kde[1][:-1], y=kde[0], name='KDE'),
            row=2, col=1
        )
        
        # QQ plot
        sorted_data = np.sort(df_col.dropna())
        theoretical_quantiles = np.quantile(np.random.normal(size=len(sorted_data)), np.linspace(0, 1, len(sorted_data)))
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Q-Q Plot'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Detailed Analysis of {df_col.name}',
            height=800,
            showlegend=True
        )
        
        stats = {
            'mean': df_col.mean(),
            'std': df_col.std(),
            'skew': df_col.skew(),
            'kurtosis': df_col.kurtosis()
        }
        
        return {
            'figure': fig.to_dict(),
            'insight': f'Detailed analysis shows: Mean={stats["mean"]:.2f}, Std={stats["std"]:.2f}, Skew={stats["skew"]:.2f}'
        }
    except Exception as e:
        logger.error(f"Error creating detailed column analysis: {str(e)}")
        return None 