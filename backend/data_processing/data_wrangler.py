"""
Data Wrangler Module

This module handles feature engineering and data transformation, including:
- Creating new features
- Transforming existing features
- Encoding categorical variables
- Feature selection
- Dimensionality reduction
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import category_encoders as ce

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/system_logs/data_wrangler.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DataWrangler:
    """
    Class for feature engineering and data transformation.
    """
    
    def __init__(self):
        """Initialize the DataWrangler class."""
        pass
    
    def wrangle_data(self, df, config=None):
        """
        Perform feature engineering and data transformation.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            config (dict, optional): Configuration for wrangling operations.
                If None, default configurations will be used.
            
        Returns:
            pandas.DataFrame: Wrangled data.
        """
        try:
            logger.info("Starting data wrangling")
            
            # Check if df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error("Input is not a pandas DataFrame")
                raise ValueError("Input is not a pandas DataFrame")
            
            # Make a copy of the DataFrame to avoid modifying the original
            wrangled_df = df.copy()
            
            # Set default configuration if not provided
            if config is None:
                config = self._get_default_config(wrangled_df)
            
            # Create new features
            if config.get("create_features", True):
                wrangled_df = self._create_features(wrangled_df, config.get("feature_config", {}))
            
            # Transform features
            if config.get("transform_features", True):
                wrangled_df = self._transform_features(wrangled_df, config.get("transform_config", {}))
            
            # Encode categorical variables
            if config.get("encode_categorical", True):
                wrangled_df = self._encode_categorical(wrangled_df, config.get("encoding_config", {}))
            
            # Select features
            if config.get("select_features", False):
                wrangled_df = self._select_features(wrangled_df, config.get("selection_config", {}))
            
            # Reduce dimensionality
            if config.get("reduce_dimensions", False):
                wrangled_df = self._reduce_dimensions(wrangled_df, config.get("dimension_config", {}))
            
            logger.info("Data wrangling completed successfully")
            
            return wrangled_df
        
        except Exception as e:
            logger.error(f"Error wrangling data: {str(e)}")
            raise
    
    def _get_default_config(self, df):
        """
        Get default configuration for data wrangling.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            
        Returns:
            dict: Default configuration.
        """
        # Get numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Default configuration
        config = {
            "create_features": True,
            "feature_config": {
                "interaction_terms": False,
                "polynomial_features": False,
                "date_features": True,
                "text_features": True,
            },
            "transform_features": True,
            "transform_config": {
                "log_transform": [],
                "sqrt_transform": [],
                "box_cox_transform": [],
                "yeo_johnson_transform": [],
            },
            "encode_categorical": True,
            "encoding_config": {
                "method": "one_hot",  # 'one_hot', 'label', 'target', 'binary', 'frequency', 'ordinal'
                "drop_first": True,
                "handle_unknown": "ignore",
            },
            "select_features": False,
            "selection_config": {
                "method": "k_best",  # 'k_best', 'mutual_info'
                "k": min(10, len(numeric_columns)) if numeric_columns else 0,
                "target_column": None,
            },
            "reduce_dimensions": False,
            "dimension_config": {
                "method": "pca",  # 'pca', 't-sne', 'umap'
                "n_components": min(5, len(numeric_columns)) if numeric_columns else 0,
            },
        }
        
        return config
    
    def _create_features(self, df, config):
        """
        Create new features from existing ones.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            config (dict): Configuration for feature creation.
            
        Returns:
            pandas.DataFrame: Data with new features.
        """
        # Get configuration
        interaction_terms = config.get("interaction_terms", False)
        polynomial_features = config.get("polynomial_features", False)
        date_features = config.get("date_features", True)
        text_features = config.get("text_features", True)
        
        # Create interaction terms
        if interaction_terms:
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_columns) >= 2:
                logger.info("Creating interaction terms")
                
                # Create interaction terms for pairs of numeric columns
                for i in range(len(numeric_columns)):
                    for j in range(i + 1, len(numeric_columns)):
                        col1 = numeric_columns[i]
                        col2 = numeric_columns[j]
                        
                        # Multiplication
                        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                        
                        # Division (avoid division by zero)
                        df[f"{col1}_div_{col2}"] = df[col1] / df[col2].replace(0, np.nan)
                        df[f"{col2}_div_{col1}"] = df[col2] / df[col1].replace(0, np.nan)
        
        # Create polynomial features
        if polynomial_features:
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_columns:
                logger.info("Creating polynomial features")
                
                # Create polynomial features
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                poly_features = poly.fit_transform(df[numeric_columns])
                
                # Create feature names
                feature_names = poly.get_feature_names_out(numeric_columns)
                
                # Add polynomial features to the DataFrame
                for i, name in enumerate(feature_names):
                    if name not in numeric_columns:  # Skip original features
                        df[name] = poly_features[:, i]
        
        # Create date features
        if date_features:
            datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
            
            for col in datetime_columns:
                logger.info(f"Creating date features for column {col}")
                
                # Extract date components
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                
                # Create cyclical features for month, day of week, etc.
                df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
                df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
                df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
                df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
        
        # Create text features
        if text_features:
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            for col in text_columns:
                # Check if the column contains text data (more than just a few words)
                if df[col].str.count(' ').mean() > 3:
                    logger.info(f"Creating text features for column {col}")
                    
                    # Create basic text features
                    df[f"{col}_length"] = df[col].str.len()
                    df[f"{col}_word_count"] = df[col].str.split().str.len()
                    df[f"{col}_char_per_word"] = df[f"{col}_length"] / df[f"{col}_word_count"].replace(0, np.nan)
        
        return df
    
    def _transform_features(self, df, config):
        """
        Transform features using various methods.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            config (dict): Configuration for feature transformation.
            
        Returns:
            pandas.DataFrame: Data with transformed features.
        """
        # Get configuration
        log_transform = config.get("log_transform", [])
        sqrt_transform = config.get("sqrt_transform", [])
        box_cox_transform = config.get("box_cox_transform", [])
        yeo_johnson_transform = config.get("yeo_johnson_transform", [])
        
        # Apply log transformation
        for col in log_transform:
            if col in df.columns and df[col].dtype.kind in 'if':  # Check if column is numeric
                logger.info(f"Applying log transformation to column {col}")
                
                # Add a small constant to avoid log(0)
                min_val = df[col].min()
                if min_val <= 0:
                    df[f"{col}_log"] = np.log(df[col] - min_val + 1)
                else:
                    df[f"{col}_log"] = np.log(df[col])
        
        # Apply square root transformation
        for col in sqrt_transform:
            if col in df.columns and df[col].dtype.kind in 'if':  # Check if column is numeric
                logger.info(f"Applying square root transformation to column {col}")
                
                # Ensure values are non-negative
                min_val = df[col].min()
                if min_val < 0:
                    df[f"{col}_sqrt"] = np.sqrt(df[col] - min_val)
                else:
                    df[f"{col}_sqrt"] = np.sqrt(df[col])
        
        # Apply Box-Cox transformation
        for col in box_cox_transform:
            if col in df.columns and df[col].dtype.kind in 'if':  # Check if column is numeric
                try:
                    from scipy import stats
                    
                    logger.info(f"Applying Box-Cox transformation to column {col}")
                    
                    # Ensure values are positive
                    min_val = df[col].min()
                    if min_val <= 0:
                        shifted_col = df[col] - min_val + 1
                    else:
                        shifted_col = df[col]
                    
                    # Apply Box-Cox transformation
                    transformed_col, lambda_val = stats.boxcox(shifted_col)
                    df[f"{col}_boxcox"] = transformed_col
                except Exception as e:
                    logger.warning(f"Error applying Box-Cox transformation to column {col}: {str(e)}")
        
        # Apply Yeo-Johnson transformation
        for col in yeo_johnson_transform:
            if col in df.columns and df[col].dtype.kind in 'if':  # Check if column is numeric
                try:
                    from sklearn.preprocessing import PowerTransformer
                    
                    logger.info(f"Applying Yeo-Johnson transformation to column {col}")
                    
                    # Apply Yeo-Johnson transformation
                    pt = PowerTransformer(method='yeo-johnson')
                    df[f"{col}_yeojohnson"] = pt.fit_transform(df[[col]])
                except Exception as e:
                    logger.warning(f"Error applying Yeo-Johnson transformation to column {col}: {str(e)}")
        
        return df
    
    def _encode_categorical(self, df, config):
        """
        Encode categorical variables.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            config (dict): Configuration for categorical encoding.
            
        Returns:
            pandas.DataFrame: Data with encoded categorical variables.
        """
        # Get configuration
        method = config.get("method", "one_hot")
        drop_first = config.get("drop_first", True)
        handle_unknown = config.get("handle_unknown", "ignore")
        target_column = config.get("target_column", None)
        
        # Get categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            logger.info("No categorical columns to encode")
            return df
        
        # Encode categorical variables
        if method == "one_hot":
            logger.info("Applying one-hot encoding")
            
            # Apply one-hot encoding
            encoder = OneHotEncoder(drop='first' if drop_first else None, handle_unknown=handle_unknown, sparse_output=False)
            encoded = encoder.fit_transform(df[categorical_columns])
            
            # Create feature names
            feature_names = encoder.get_feature_names_out(categorical_columns)
            
            # Create a DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            
            # Concatenate with original DataFrame, dropping original categorical columns
            df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        
        elif method == "label":
            logger.info("Applying label encoding")
            
            # Apply label encoding
            for col in categorical_columns:
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
            
            # Drop original categorical columns
            df = df.drop(columns=categorical_columns)
        
        elif method == "target":
            if target_column is None or target_column not in df.columns:
                logger.warning("Target column not specified or not found, falling back to one-hot encoding")
                return self._encode_categorical(df, {"method": "one_hot", "drop_first": drop_first, "handle_unknown": handle_unknown})
            
            logger.info("Applying target encoding")
            
            # Apply target encoding
            encoder = ce.TargetEncoder()
            df[categorical_columns + "_encoded"] = encoder.fit_transform(df[categorical_columns], df[target_column])
            
            # Drop original categorical columns
            df = df.drop(columns=categorical_columns)
        
        elif method == "binary":
            logger.info("Applying binary encoding")
            
            # Apply binary encoding
            encoder = ce.BinaryEncoder()
            encoded = encoder.fit_transform(df[categorical_columns])
            
            # Concatenate with original DataFrame, dropping original categorical columns
            df = pd.concat([df.drop(columns=categorical_columns), encoded], axis=1)
        
        elif method == "frequency":
            logger.info("Applying frequency encoding")
            
            # Apply frequency encoding
            for col in categorical_columns:
                frequency = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq"] = df[col].map(frequency)
            
            # Drop original categorical columns
            df = df.drop(columns=categorical_columns)
        
        elif method == "ordinal":
            logger.info("Applying ordinal encoding")
            
            # Apply ordinal encoding
            encoder = ce.OrdinalEncoder()
            encoded = encoder.fit_transform(df[categorical_columns])
            
            # Concatenate with original DataFrame, dropping original categorical columns
            df = pd.concat([df.drop(columns=categorical_columns), encoded], axis=1)
        
        else:
            logger.warning(f"Unknown encoding method: {method}, falling back to one-hot encoding")
            return self._encode_categorical(df, {"method": "one_hot", "drop_first": drop_first, "handle_unknown": handle_unknown})
        
        return df
    
    def _select_features(self, df, config):
        """
        Select features based on various methods.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            config (dict): Configuration for feature selection.
            
        Returns:
            pandas.DataFrame: Data with selected features.
        """
        # Get configuration
        method = config.get("method", "k_best")
        k = config.get("k", 10)
        target_column = config.get("target_column", None)
        
        # Check if target column is specified
        if target_column is None or target_column not in df.columns:
            logger.warning("Target column not specified or not found, skipping feature selection")
            return df
        
        # Get numeric columns (excluding target)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != target_column]
        
        if not numeric_columns:
            logger.info("No numeric columns for feature selection")
            return df
        
        # Select features
        if method == "k_best":
            logger.info(f"Selecting {k} best features using f_regression")
            
            # Select k best features
            selector = SelectKBest(f_regression, k=k)
            selected = selector.fit_transform(df[numeric_columns], df[target_column])
            
            # Get selected feature names
            selected_columns = [numeric_columns[i] for i in range(len(numeric_columns)) if selector.get_support()[i]]
            
            # Keep only selected features and non-numeric columns
            non_numeric_columns = [col for col in df.columns if col not in numeric_columns and col != target_column]
            df = df[selected_columns + non_numeric_columns + [target_column]]
        
        elif method == "mutual_info":
            logger.info(f"Selecting {k} best features using mutual information")
            
            # Select k best features
            selector = SelectKBest(mutual_info_regression, k=k)
            selected = selector.fit_transform(df[numeric_columns], df[target_column])
            
            # Get selected feature names
            selected_columns = [numeric_columns[i] for i in range(len(numeric_columns)) if selector.get_support()[i]]
            
            # Keep only selected features and non-numeric columns
            non_numeric_columns = [col for col in df.columns if col not in numeric_columns and col != target_column]
            df = df[selected_columns + non_numeric_columns + [target_column]]
        
        else:
            logger.warning(f"Unknown feature selection method: {method}")
        
        return df
    
    def _reduce_dimensions(self, df, config):
        """
        Reduce dimensionality of the data.
        
        Args:
            df (pandas.DataFrame): The data to wrangle.
            config (dict): Configuration for dimensionality reduction.
            
        Returns:
            pandas.DataFrame: Data with reduced dimensions.
        """
        # Get configuration
        method = config.get("method", "pca")
        n_components = config.get("n_components", 5)
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns or len(numeric_columns) <= n_components:
            logger.info("Not enough numeric columns for dimensionality reduction")
            return df
        
        # Reduce dimensions
        if method == "pca":
            logger.info(f"Reducing dimensions to {n_components} components using PCA")
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(df[numeric_columns])
            
            # Create feature names
            feature_names = [f"pca_component_{i+1}" for i in range(n_components)]
            
            # Create a DataFrame with reduced features
            reduced_df = pd.DataFrame(reduced, columns=feature_names, index=df.index)
            
            # Concatenate with non-numeric columns
            non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
            df = pd.concat([reduced_df, df[non_numeric_columns]], axis=1)
        
        elif method == "t-sne":
            try:
                from sklearn.manifold import TSNE
                
                logger.info(f"Reducing dimensions to {n_components} components using t-SNE")
                
                # Apply t-SNE
                tsne = TSNE(n_components=n_components, random_state=42)
                reduced = tsne.fit_transform(df[numeric_columns])
                
                # Create feature names
                feature_names = [f"tsne_component_{i+1}" for i in range(n_components)]
                
                # Create a DataFrame with reduced features
                reduced_df = pd.DataFrame(reduced, columns=feature_names, index=df.index)
                
                # Concatenate with non-numeric columns
                non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
                df = pd.concat([reduced_df, df[non_numeric_columns]], axis=1)
            except Exception as e:
                logger.warning(f"Error applying t-SNE: {str(e)}")
        
        elif method == "umap":
            try:
                import umap
                
                logger.info(f"Reducing dimensions to {n_components} components using UMAP")
                
                # Apply UMAP
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(df[numeric_columns])
                
                # Create feature names
                feature_names = [f"umap_component_{i+1}" for i in range(n_components)]
                
                # Create a DataFrame with reduced features
                reduced_df = pd.DataFrame(reduced, columns=feature_names, index=df.index)
                
                # Concatenate with non-numeric columns
                non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
                df = pd.concat([reduced_df, df[non_numeric_columns]], axis=1)
            except Exception as e:
                logger.warning(f"Error applying UMAP: {str(e)}")
        
        else:
            logger.warning(f"Unknown dimensionality reduction method: {method}")
        
        return df 