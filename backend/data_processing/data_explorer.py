"""
Data Explorer Module

This module handles exploring and understanding the data, including:
- Data types
- Missing values
- Summary statistics
- Outliers
- Distributions
- Correlations
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
import json

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/system_logs/data_explorer.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DataExplorer:
    """
    Class for exploring and understanding data.
    """
    
    def __init__(self):
        """Initialize the DataExplorer class."""
        pass
    
    def explore_data(self, df):
        """
        Explore the data and return a summary of the findings.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: A summary of the findings.
        """
        try:
            logger.info("Starting data exploration")
            
            # Check if df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error("Input is not a pandas DataFrame")
                raise ValueError("Input is not a pandas DataFrame")
            
            # Get basic information
            basic_info = self._get_basic_info(df)
            
            # Get data types
            data_types = self._get_data_types(df)
            
            # Get missing values
            missing_values = self._get_missing_values(df)
            
            # Get summary statistics
            summary_stats = self._get_summary_statistics(df)
            
            # Get outliers
            outliers = self._get_outliers(df)
            
            # Get distributions
            distributions = self._get_distributions(df)
            
            # Get correlations
            correlations = self._get_correlations(df)
            
            # Combine all findings
            findings = {
                "basic_info": basic_info,
                "data_types": data_types,
                "missing_values": missing_values,
                "summary_statistics": summary_stats,
                "outliers": outliers,
                "distributions": distributions,
                "correlations": correlations
            }
            
            logger.info("Data exploration completed successfully")
            
            return findings
        
        except Exception as e:
            logger.error(f"Error exploring data: {str(e)}")
            raise
    
    def _get_basic_info(self, df):
        """
        Get basic information about the data.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Basic information about the data.
        """
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
        }
    
    def _get_data_types(self, df):
        """
        Get the data types of each column.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Data types of each column.
        """
        # Get pandas dtypes
        dtypes = df.dtypes.astype(str).to_dict()
        
        # Categorize columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()
        
        return {
            "dtypes": dtypes,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "boolean_columns": boolean_columns,
        }
    
    def _get_missing_values(self, df):
        """
        Get information about missing values.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Information about missing values.
        """
        # Count missing values
        missing_count = df.isnull().sum().to_dict()
        
        # Calculate percentage of missing values
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Get columns with missing values
        columns_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
        
        return {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "columns_with_missing": columns_with_missing,
            "total_missing_cells": df.isnull().sum().sum(),
            "total_missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        }
    
    def _get_summary_statistics(self, df):
        """
        Get summary statistics for numeric columns.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Summary statistics for numeric columns.
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        # Calculate summary statistics
        summary = numeric_df.describe().to_dict()
        
        # Calculate additional statistics
        skewness = numeric_df.skew().to_dict()
        kurtosis = numeric_df.kurtosis().to_dict()
        
        # Calculate IQR
        q1 = numeric_df.quantile(0.25).to_dict()
        q3 = numeric_df.quantile(0.75).to_dict()
        iqr = {col: q3[col] - q1[col] for col in q1.keys()}
        
        # Get categorical columns
        categorical_df = df.select_dtypes(include=['object', 'category'])
        categorical_summary = {}
        
        for col in categorical_df.columns:
            value_counts = categorical_df[col].value_counts().to_dict()
            unique_count = len(value_counts)
            top_values = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            
            categorical_summary[col] = {
                "unique_count": unique_count,
                "top_values": top_values,
            }
        
        return {
            "numeric": {
                "summary": summary,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "iqr": iqr,
            },
            "categorical": categorical_summary,
        }
    
    def _get_outliers(self, df):
        """
        Detect outliers in numeric columns.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Information about outliers.
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        outliers = {}
        
        for col in numeric_df.columns:
            # Calculate IQR
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outlier_count = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            outlier_percentage = outlier_count / len(numeric_df) * 100
            
            # Get outlier values
            outlier_values = numeric_df[col][(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)].tolist()
            
            # Limit the number of outlier values to display
            if len(outlier_values) > 10:
                outlier_values = outlier_values[:10]
            
            outliers[col] = {
                "outlier_count": int(outlier_count),
                "outlier_percentage": float(outlier_percentage),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": outlier_values,
            }
        
        return outliers
    
    def _get_distributions(self, df):
        """
        Get distribution information for numeric columns.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Distribution information for numeric columns.
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        distributions = {}
        
        for col in numeric_df.columns:
            # Calculate basic statistics
            mean = float(numeric_df[col].mean())
            median = float(numeric_df[col].median())
            mode = float(numeric_df[col].mode().iloc[0]) if not numeric_df[col].mode().empty else None
            std = float(numeric_df[col].std())
            min_val = float(numeric_df[col].min())
            max_val = float(numeric_df[col].max())
            
            # Calculate percentiles
            percentiles = {
                "5%": float(numeric_df[col].quantile(0.05)),
                "25%": float(numeric_df[col].quantile(0.25)),
                "50%": float(numeric_df[col].quantile(0.50)),
                "75%": float(numeric_df[col].quantile(0.75)),
                "95%": float(numeric_df[col].quantile(0.95)),
            }
            
            # Test for normality
            if len(numeric_df[col].dropna()) >= 8:  # Need at least 8 observations for shapiro test
                shapiro_test = stats.shapiro(numeric_df[col].dropna())
                normality_test = {
                    "test": "Shapiro-Wilk",
                    "statistic": float(shapiro_test[0]),
                    "p_value": float(shapiro_test[1]),
                    "is_normal": shapiro_test[1] > 0.05,
                }
            else:
                normality_test = {
                    "test": "Shapiro-Wilk",
                    "message": "Not enough observations for normality test",
                }
            
            distributions[col] = {
                "mean": mean,
                "median": median,
                "mode": mode,
                "std": std,
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "percentiles": percentiles,
                "normality_test": normality_test,
            }
        
        return distributions
    
    def _get_correlations(self, df):
        """
        Calculate correlations between numeric columns.
        
        Args:
            df (pandas.DataFrame): The data to explore.
            
        Returns:
            dict: Correlations between numeric columns.
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}
        
        # Calculate correlations
        pearson_corr = numeric_df.corr(method='pearson').to_dict()
        spearman_corr = numeric_df.corr(method='spearman').to_dict()
        
        # Find strong correlations
        strong_correlations = []
        
        for col1 in pearson_corr:
            for col2 in pearson_corr[col1]:
                if col1 != col2 and abs(pearson_corr[col1][col2]) > 0.7:
                    strong_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "pearson_correlation": pearson_corr[col1][col2],
                        "spearman_correlation": spearman_corr[col1][col2],
                    })
        
        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "strong_correlations": strong_correlations,
        } 