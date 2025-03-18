"""
Data Cleaner Module

This module handles cleaning and preprocessing data, including:
- Handling missing values
- Removing duplicates
- Handling outliers
- Converting data types
- Normalizing and standardizing data
"""

import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/system_logs/data_cleaner.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DataCleaner:
    """
    Class for cleaning and preprocessing data.
    """
    
    def __init__(self):
        """Initialize the DataCleaner class."""
        pass
    
    def clean_data(self, df, config=None):
        """
        Clean and preprocess the data.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            config (dict, optional): Configuration for cleaning operations.
                If None, default configurations will be used.
            
        Returns:
            pandas.DataFrame: Cleaned data.
        """
        try:
            logger.info("Starting data cleaning")
            
            # Check if df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error("Input is not a pandas DataFrame")
                raise ValueError("Input is not a pandas DataFrame")
            
            # Make a copy of the DataFrame to avoid modifying the original
            cleaned_df = df.copy()
            
            # Set default configuration if not provided
            if config is None:
                config = self._get_default_config(cleaned_df)
            
            # Handle missing values
            if config.get("handle_missing", True):
                cleaned_df = self._handle_missing_values(cleaned_df, config.get("missing_config", {}))
            
            # Remove duplicates
            if config.get("remove_duplicates", True):
                cleaned_df = self._remove_duplicates(cleaned_df, config.get("duplicate_config", {}))
            
            # Handle outliers
            if config.get("handle_outliers", True):
                cleaned_df = self._handle_outliers(cleaned_df, config.get("outlier_config", {}))
            
            # Convert data types
            if config.get("convert_types", True):
                cleaned_df = self._convert_data_types(cleaned_df, config.get("type_config", {}))
            
            # Normalize/standardize data
            if config.get("normalize", False):
                cleaned_df = self._normalize_data(cleaned_df, config.get("normalize_config", {}))
            
            logger.info("Data cleaning completed successfully")
            
            return cleaned_df
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def _get_default_config(self, df):
        """
        Get default configuration for data cleaning.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            
        Returns:
            dict: Default configuration.
        """
        # Get numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Default configuration
        config = {
            "handle_missing": True,
            "missing_config": {
                "numeric_strategy": "mean",
                "categorical_strategy": "most_frequent",
                "max_missing_ratio": 0.5,  # Drop columns with more than 50% missing values
            },
            "remove_duplicates": True,
            "duplicate_config": {
                "subset": None,  # Consider all columns
                "keep": "first",  # Keep first occurrence
            },
            "handle_outliers": True,
            "outlier_config": {
                "method": "iqr",
                "threshold": 1.5,
                "strategy": "clip",  # 'clip', 'remove', or 'replace'
            },
            "convert_types": True,
            "type_config": {
                "infer_datetime": True,
                "categorical_threshold": 10,  # Convert columns with less than 10 unique values to categorical
            },
            "normalize": False,
            "normalize_config": {
                "method": "standard",  # 'standard', 'minmax', or 'robust'
                "columns": numeric_columns,
            },
        }
        
        return config
    
    def _handle_missing_values(self, df, config):
        """
        Handle missing values in the data.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            config (dict): Configuration for handling missing values.
            
        Returns:
            pandas.DataFrame: Data with missing values handled.
        """
        # Get configuration
        numeric_strategy = config.get("numeric_strategy", "mean")
        categorical_strategy = config.get("categorical_strategy", "most_frequent")
        max_missing_ratio = config.get("max_missing_ratio", 0.5)
        
        # Get columns with missing values
        missing_columns = df.columns[df.isnull().any()].tolist()
        
        if not missing_columns:
            logger.info("No missing values found")
            return df
        
        # Drop columns with too many missing values
        for col in missing_columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                logger.info(f"Dropping column {col} with {missing_ratio:.2%} missing values")
                df = df.drop(columns=[col])
        
        # Get numeric and categorical columns with missing values
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_missing = [col for col in numeric_columns if df[col].isnull().any()]
        categorical_missing = [col for col in categorical_columns if df[col].isnull().any()]
        
        # Handle missing values in numeric columns
        if numeric_missing:
            if numeric_strategy == "knn":
                # Use KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_missing] = imputer.fit_transform(df[numeric_missing])
            else:
                # Use simple imputation
                imputer = SimpleImputer(strategy=numeric_strategy)
                df[numeric_missing] = imputer.fit_transform(df[numeric_missing])
        
        # Handle missing values in categorical columns
        if categorical_missing:
            for col in categorical_missing:
                if categorical_strategy == "most_frequent":
                    # Fill with most frequent value
                    most_frequent = df[col].mode()[0]
                    df[col] = df[col].fillna(most_frequent)
                elif categorical_strategy == "new_category":
                    # Fill with a new category
                    df[col] = df[col].fillna("Unknown")
        
        return df
    
    def _remove_duplicates(self, df, config):
        """
        Remove duplicate rows from the data.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            config (dict): Configuration for removing duplicates.
            
        Returns:
            pandas.DataFrame: Data with duplicates removed.
        """
        # Get configuration
        subset = config.get("subset", None)
        keep = config.get("keep", "first")
        
        # Check for duplicates
        duplicate_count = df.duplicated(subset=subset).sum()
        
        if duplicate_count == 0:
            logger.info("No duplicate rows found")
            return df
        
        # Remove duplicates
        logger.info(f"Removing {duplicate_count} duplicate rows")
        df = df.drop_duplicates(subset=subset, keep=keep)
        
        return df
    
    def _handle_outliers(self, df, config):
        """
        Handle outliers in numeric columns.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            config (dict): Configuration for handling outliers.
            
        Returns:
            pandas.DataFrame: Data with outliers handled.
        """
        # Get configuration
        method = config.get("method", "iqr")
        threshold = config.get("threshold", 1.5)
        strategy = config.get("strategy", "clip")
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns:
            logger.info("No numeric columns found for outlier handling")
            return df
        
        # Handle outliers in each numeric column
        for col in numeric_columns:
            if method == "iqr":
                # Calculate IQR
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            elif method == "zscore":
                # Calculate Z-score
                mean = df[col].mean()
                std = df[col].std()
                
                # Define outlier bounds
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            # Count outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count == 0:
                continue
            
            logger.info(f"Found {outlier_count} outliers in column {col}")
            
            # Handle outliers based on strategy
            if strategy == "clip":
                # Clip values to the bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif strategy == "remove":
                # Remove rows with outliers
                df = df[~outlier_mask]
            elif strategy == "replace":
                # Replace outliers with bounds
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
    
    def _convert_data_types(self, df, config):
        """
        Convert data types of columns.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            config (dict): Configuration for converting data types.
            
        Returns:
            pandas.DataFrame: Data with converted data types.
        """
        # Get configuration
        infer_datetime = config.get("infer_datetime", True)
        categorical_threshold = config.get("categorical_threshold", 10)
        
        # Infer datetime columns
        if infer_datetime:
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Try to convert to datetime
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Converted column {col} to datetime")
                except:
                    pass
        
        # Convert columns with few unique values to categorical
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() <= categorical_threshold:
                df[col] = df[col].astype('category')
                logger.info(f"Converted column {col} to categorical")
        
        return df
    
    def _normalize_data(self, df, config):
        """
        Normalize or standardize numeric columns.
        
        Args:
            df (pandas.DataFrame): The data to clean.
            config (dict): Configuration for normalization.
            
        Returns:
            pandas.DataFrame: Data with normalized columns.
        """
        # Get configuration
        method = config.get("method", "standard")
        columns = config.get("columns", df.select_dtypes(include=['number']).columns.tolist())
        
        if not columns:
            logger.info("No columns to normalize")
            return df
        
        # Create a copy of the DataFrame
        normalized_df = df.copy()
        
        # Select the scaler based on the method
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return df
        
        # Apply the scaler
        normalized_df[columns] = scaler.fit_transform(normalized_df[columns])
        
        return normalized_df 