import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import asyncio
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Set up visualization directory
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "./uploads")
VISUALIZATION_FOLDER = os.path.join(UPLOAD_FOLDER, "visualizations")
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Set Matplotlib style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

async def generate_visualization(
    file_path: str,
    viz_type: str,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate a visualization based on the file and parameters
    
    Args:
        file_path: Path to the data file
        viz_type: Type of visualization to generate
        params: Parameters for the visualization
        
    Returns:
        Dictionary with visualization details
    """
    if params is None:
        params = {}
    
    try:
        # Run in a thread pool to avoid blocking
        return await asyncio.to_thread(
            _generate_visualization_sync,
            file_path,
            viz_type,
            params
        )
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise

def _generate_visualization_sync(
    file_path: str,
    viz_type: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Synchronous implementation of visualization generation"""
    # Load the data
    df = load_data(file_path)
    
    # Generate a unique ID for the visualization
    viz_id = str(uuid.uuid4())
    
    # Generate the visualization
    if viz_type.lower() == "histogram":
        fig, title, description = create_histogram(df, params)
    elif viz_type.lower() == "scatter":
        fig, title, description = create_scatter_plot(df, params)
    elif viz_type.lower() == "bar":
        fig, title, description = create_bar_chart(df, params)
    elif viz_type.lower() == "line":
        fig, title, description = create_line_chart(df, params)
    elif viz_type.lower() == "heatmap":
        fig, title, description = create_heatmap(df, params)
    elif viz_type.lower() == "boxplot":
        fig, title, description = create_boxplot(df, params)
    elif viz_type.lower() == "pie":
        fig, title, description = create_pie_chart(df, params)
    elif viz_type.lower() == "correlation":
        fig, title, description = create_correlation_matrix(df, params)
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")
    
    # Save the visualization
    filename = f"{viz_id}.png"
    filepath = os.path.join(VISUALIZATION_FOLDER, filename)
    
    # Save the figure
    if isinstance(fig, plt.Figure):
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        # Assume it's a plotly figure
        pio.write_image(fig, filepath)
    
    # Return the visualization details
    return {
        "visualization_id": viz_id,
        "visualization_url": f"/api/visualization/visualizations/{filename}",
        "title": title,
        "description": description,
        "viz_type": viz_type,
        "file_path": file_path,
        "created_at": datetime.utcnow().isoformat()
    }

async def generate_dashboard(
    file_path: str,
    title: str = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive dashboard for a dataset
    
    Args:
        file_path: Path to the data file
        title: Title for the dashboard
        
    Returns:
        Dictionary with dashboard details
    """
    try:
        # Run in a thread pool to avoid blocking
        return await asyncio.to_thread(
            _generate_dashboard_sync,
            file_path,
            title
        )
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise

def _generate_dashboard_sync(
    file_path: str,
    title: str = None
) -> Dict[str, Any]:
    """Synchronous implementation of dashboard generation"""
    # Load the data
    df = load_data(file_path)
    
    # Generate a unique ID for the dashboard
    dashboard_id = str(uuid.uuid4())
    
    # Get dataset statistics
    stats = get_dataset_stats(df)
    
    # Generate visualizations
    visualizations = []
    
    # If no title provided, use the filename
    if title is None:
        title = os.path.basename(file_path)
    
    # 1. Distribution of numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for i, col in enumerate(numeric_cols[:5]):  # Limit to first 5 numeric columns
        viz = _generate_visualization_sync(
            file_path,
            "histogram",
            {"column": col, "bins": 20}
        )
        visualizations.append(viz)
    
    # 2. Correlation matrix if there are multiple numeric columns
    if len(numeric_cols) > 1:
        viz = _generate_visualization_sync(
            file_path,
            "correlation",
            {"columns": numeric_cols[:10]}  # Limit to first 10 numeric columns
        )
        visualizations.append(viz)
    
    # 3. Bar charts for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for i, col in enumerate(cat_cols[:3]):  # Limit to first 3 categorical columns
        viz = _generate_visualization_sync(
            file_path,
            "bar",
            {"x": col, "count": True}
        )
        visualizations.append(viz)
    
    # 4. Scatter plot for first two numeric columns if available
    if len(numeric_cols) >= 2:
        viz = _generate_visualization_sync(
            file_path,
            "scatter",
            {"x": numeric_cols[0], "y": numeric_cols[1]}
        )
        visualizations.append(viz)
    
    # 5. Box plots for numeric columns
    for i, col in enumerate(numeric_cols[:3]):  # Limit to first 3 numeric columns
        viz = _generate_visualization_sync(
            file_path,
            "boxplot",
            {"column": col}
        )
        visualizations.append(viz)
    
    # Return the dashboard details
    return {
        "dashboard_id": dashboard_id,
        "title": title,
        "visualizations": visualizations,
        "stats": stats,
        "file_path": file_path,
        "created_at": datetime.utcnow().isoformat()
    }

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a file into a pandas DataFrame"""
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        elif file_ext == '.feather':
            return pd.read_feather(file_path)
        else:
            # Try CSV as a default
            return pd.read_csv(file_path)
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about the dataset"""
    # Basic stats
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=['number']).columns),
        "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
        "missing_values": df.isna().sum().sum(),
        "column_types": {}
    }
    
    # Column types and basic stats
    for col in df.columns:
        col_type = str(df[col].dtype)
        stats["column_types"][col] = {
            "type": col_type,
            "missing": int(df[col].isna().sum())
        }
        
        # Add type-specific stats
        if np.issubdtype(df[col].dtype, np.number):
            stats["column_types"][col].update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
            })
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            stats["column_types"][col].update({
                "unique_values": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict()
            })
    
    return stats

def create_histogram(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a histogram visualization"""
    column = params.get("column")
    if not column or column not in df.columns:
        raise ValueError(f"Invalid column: {column}")
    
    bins = params.get("bins", 20)
    title = params.get("title", f"Distribution of {column}")
    
    # Check if column is numeric
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column {column} is not numeric")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    
    # Generate description
    description = f"Histogram showing the distribution of {column} with {bins} bins."
    
    return fig, title, description

def create_scatter_plot(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a scatter plot visualization"""
    x = params.get("x")
    y = params.get("y")
    
    if not x or x not in df.columns:
        raise ValueError(f"Invalid x column: {x}")
    if not y or y not in df.columns:
        raise ValueError(f"Invalid y column: {y}")
    
    # Check if columns are numeric
    if not np.issubdtype(df[x].dtype, np.number) or not np.issubdtype(df[y].dtype, np.number):
        raise ValueError(f"Columns {x} and {y} must be numeric")
    
    color = params.get("color")
    title = params.get("title", f"{y} vs {x}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color and color in df.columns:
        scatter = sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax)
        # Add legend
        plt.legend(title=color, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        scatter = sns.scatterplot(data=df, x=x, y=y, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    # Add regression line if requested
    if params.get("regression", False):
        sns.regplot(data=df, x=x, y=y, scatter=False, ax=ax, line_kws={"color": "red"})
    
    # Generate description
    description = f"Scatter plot showing the relationship between {x} and {y}."
    if color:
        description += f" Points are colored by {color}."
    if params.get("regression", False):
        description += " A regression line is included."
    
    return fig, title, description

def create_bar_chart(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a bar chart visualization"""
    x = params.get("x")
    y = params.get("y")
    count = params.get("count", False)
    
    if not x or x not in df.columns:
        raise ValueError(f"Invalid x column: {x}")
    
    if count:
        # Count plot
        title = params.get("title", f"Count of {x}")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Limit to top N categories if there are too many
        if df[x].nunique() > 20:
            top_cats = df[x].value_counts().head(20).index
            plot_df = df[df[x].isin(top_cats)].copy()
            sns.countplot(data=plot_df, x=x, ax=ax, order=top_cats)
            ax.set_title(f"{title} (Top 20)")
        else:
            sns.countplot(data=df, x=x, ax=ax)
            ax.set_title(title)
        
        ax.set_xlabel(x)
        ax.set_ylabel("Count")
        
        # Rotate x labels if there are many categories
        if df[x].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
        
        description = f"Bar chart showing the count of each category in {x}."
        if df[x].nunique() > 20:
            description += " Only the top 20 categories are shown."
    
    else:
        # Regular bar chart
        if not y or y not in df.columns:
            raise ValueError(f"Invalid y column: {y}")
        
        title = params.get("title", f"{y} by {x}")
        
        # Aggregate the data if needed
        agg_func = params.get("agg_func", "mean")
        
        # Limit to top N categories if there are too many
        if df[x].nunique() > 20:
            top_cats = df[x].value_counts().head(20).index
            plot_df = df[df[x].isin(top_cats)].copy()
            agg_data = plot_df.groupby(x)[y].agg(agg_func).reset_index()
        else:
            agg_data = df.groupby(x)[y].agg(agg_func).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=agg_data, x=x, y=y, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(f"{agg_func.capitalize()} of {y}")
        
        # Rotate x labels if there are many categories
        if agg_data[x].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
        
        description = f"Bar chart showing the {agg_func} of {y} for each category in {x}."
        if df[x].nunique() > 20:
            description += " Only the top 20 categories are shown."
    
    return fig, title, description

def create_line_chart(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a line chart visualization"""
    x = params.get("x")
    y = params.get("y")
    
    if not x or x not in df.columns:
        raise ValueError(f"Invalid x column: {x}")
    if not y or y not in df.columns:
        raise ValueError(f"Invalid y column: {y}")
    
    title = params.get("title", f"{y} over {x}")
    
    # Sort by x if it's a datetime or numeric column
    if pd.api.types.is_datetime64_any_dtype(df[x]) or np.issubdtype(df[x].dtype, np.number):
        plot_df = df.sort_values(by=x).copy()
    else:
        plot_df = df.copy()
    
    # Group by x if there are duplicates
    if plot_df[x].duplicated().any():
        agg_func = params.get("agg_func", "mean")
        plot_df = plot_df.groupby(x)[y].agg(agg_func).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x=x, y=y, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    # Rotate x labels if needed
    if not np.issubdtype(plot_df[x].dtype, np.number):
        plt.xticks(rotation=45, ha='right')
    
    # Generate description
    description = f"Line chart showing {y} over {x}."
    if plot_df[x].duplicated().any():
        agg_func = params.get("agg_func", "mean")
        description += f" Values are aggregated using {agg_func}."
    
    return fig, title, description

def create_heatmap(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a heatmap visualization"""
    columns = params.get("columns")
    
    # If columns not specified, use numeric columns
    if not columns:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Validate columns
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Invalid column: {col}")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Column {col} is not numeric")
    
    # Limit to 15 columns max for readability
    if len(columns) > 15:
        columns = columns[:15]
    
    title = params.get("title", "Correlation Heatmap")
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=params.get("annot", True),
        cmap=params.get("cmap", "coolwarm"),
        center=0,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(title)
    
    # Generate description
    description = f"Heatmap showing the correlation between {len(columns)} numeric columns."
    
    return fig, title, description

def create_boxplot(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a boxplot visualization"""
    column = params.get("column")
    by = params.get("by")
    
    if not column or column not in df.columns:
        raise ValueError(f"Invalid column: {column}")
    
    # Check if column is numeric
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column {column} is not numeric")
    
    title = params.get("title", f"Distribution of {column}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if by and by in df.columns:
        # If too many categories in 'by', limit to top N
        if df[by].nunique() > 10:
            top_cats = df[by].value_counts().head(10).index
            plot_df = df[df[by].isin(top_cats)].copy()
            sns.boxplot(data=plot_df, x=by, y=column, ax=ax)
            title = f"Distribution of {column} by {by} (Top 10)"
        else:
            sns.boxplot(data=df, x=by, y=column, ax=ax)
            title = f"Distribution of {column} by {by}"
        
        # Rotate x labels if there are many categories
        if df[by].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
    else:
        sns.boxplot(data=df, y=column, ax=ax)
    
    ax.set_title(title)
    
    # Generate description
    description = f"Box plot showing the distribution of {column}."
    if by:
        description += f" Grouped by {by}."
        if df[by].nunique() > 10:
            description += " Only the top 10 categories are shown."
    
    return fig, title, description

def create_pie_chart(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a pie chart visualization"""
    column = params.get("column")
    
    if not column or column not in df.columns:
        raise ValueError(f"Invalid column: {column}")
    
    title = params.get("title", f"Distribution of {column}")
    
    # Get value counts
    value_counts = df[column].value_counts()
    
    # If too many categories, group smaller ones into "Other"
    if len(value_counts) > 10:
        top_n = value_counts.head(9)
        other_count = value_counts[9:].sum()
        
        # Create a new series with top N and "Other"
        values = pd.Series(list(top_n) + [other_count], 
                          index=list(top_n.index) + ["Other"])
    else:
        values = value_counts
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(
        values,
        labels=values.index,
        autopct='%1.1f%%',
        startangle=90,
        shadow=params.get("shadow", False),
        explode=[0.05] * len(values) if params.get("explode", False) else None
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    ax.set_title(title)
    
    # Generate description
    description = f"Pie chart showing the distribution of categories in {column}."
    if len(value_counts) > 10:
        description += " The top 9 categories are shown, with remaining categories grouped as 'Other'."
    
    return fig, title, description

def create_correlation_matrix(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Union[plt.Figure, go.Figure], str, str]:
    """Create a correlation matrix visualization"""
    columns = params.get("columns")
    
    # If columns not specified, use numeric columns
    if not columns:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Validate columns
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Invalid column: {col}")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Column {col} is not numeric")
    
    # Limit to 15 columns max for readability
    if len(columns) > 15:
        columns = columns[:15]
    
    title = params.get("title", "Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(title)
    
    # Generate description
    description = f"Correlation matrix showing the Pearson correlation coefficients between {len(columns)} numeric columns."
    
    return fig, title, description 