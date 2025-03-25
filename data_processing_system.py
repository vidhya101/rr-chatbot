import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.datasets = {}  # Store dataframes with their names
        self.transformations = {}  # Store applied transformations
        self.models = {}  # Store trained models
        self.logs = []  # Store processing logs and code snippets
        self.validation_rules = {}  # Store data validation rules
        self.preprocessing_pipelines = {}  # Store preprocessing pipelines
        
    def add_validation_rule(self, rule_name: str, rule_function: callable, description: str) -> None:
        """
        Add a custom validation rule to the processor
        
        Args:
            rule_name: Name of the validation rule
            rule_function: Function that implements the validation logic
            description: Description of what the rule checks
        """
        self.validation_rules[rule_name] = {
            "function": rule_function,
            "description": description
        }
        
    def validate_dataset(self, dataset_id: str, rules: List[str] = None) -> Dict[str, Any]:
        """
        Validate a dataset against specified rules
        
        Args:
            dataset_id: ID of the dataset to validate
            rules: List of rule names to apply (None for all rules)
            
        Returns:
            Dict with validation results
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        df = self.datasets[dataset_id]
        results = {
            "dataset_id": dataset_id,
            "validation_results": {},
            "is_valid": True
        }
        
        # If no rules specified, use all available rules
        rules_to_check = rules if rules else list(self.validation_rules.keys())
        
        for rule_name in rules_to_check:
            if rule_name not in self.validation_rules:
                logger.warning(f"Rule {rule_name} not found")
                continue
                
            rule = self.validation_rules[rule_name]
            try:
                validation_result = rule["function"](df)
                results["validation_results"][rule_name] = {
                    "description": rule["description"],
                    "passed": validation_result["passed"],
                    "details": validation_result.get("details", {})
                }
                if not validation_result["passed"]:
                    results["is_valid"] = False
            except Exception as e:
                logger.error(f"Error applying rule {rule_name}: {str(e)}")
                results["validation_results"][rule_name] = {
                    "description": rule["description"],
                    "passed": False,
                    "error": str(e)
                }
                results["is_valid"] = False
        
        return results
        
    def create_preprocessing_pipeline(self, pipeline_name: str, steps: List[Dict[str, Any]]) -> None:
        """
        Create a preprocessing pipeline with multiple steps
        
        Args:
            pipeline_name: Name of the pipeline
            steps: List of preprocessing steps with their parameters
        """
        self.preprocessing_pipelines[pipeline_name] = steps
        
    def apply_preprocessing_pipeline(self, dataset_id: str, pipeline_name: str) -> str:
        """
        Apply a preprocessing pipeline to a dataset
        
        Args:
            dataset_id: ID of the dataset to process
            pipeline_name: Name of the pipeline to apply
            
        Returns:
            str: ID of the processed dataset
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        if pipeline_name not in self.preprocessing_pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
            
        df = self.datasets[dataset_id].copy()
        pipeline = self.preprocessing_pipelines[pipeline_name]
        code_lines = []
        
        for step in pipeline:
            step_type = step["type"]
            params = step.get("params", {})
            
            if step_type == "handle_missing":
                df = self._handle_missing_values(df, params)
                code_lines.extend(self._generate_missing_values_code(params))
                
            elif step_type == "scale":
                df = self._scale_features(df, params)
                code_lines.extend(self._generate_scaling_code(params))
                
            elif step_type == "encode":
                df = self._encode_features(df, params)
                code_lines.extend(self._generate_encoding_code(params))
                
            elif step_type == "select_features":
                df = self._select_features(df, params)
                code_lines.extend(self._generate_feature_selection_code(params))
                
            elif step_type == "create_features":
                df = self._create_features(df, params)
                code_lines.extend(self._generate_feature_creation_code(params))
        
        # Create new dataset ID
        processed_dataset_id = f"{dataset_id}_{pipeline_name}"
        self.datasets[processed_dataset_id] = df
        
        # Log the operation
        code_snippet = "\n".join(code_lines)
        self.logs.append({
            "operation": "apply_preprocessing_pipeline",
            "dataset_id": dataset_id,
            "pipeline_name": pipeline_name,
            "processed_dataset_id": processed_dataset_id,
            "code_snippet": code_snippet
        })
        
        return processed_dataset_id
        
    def _handle_missing_values(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values based on specified parameters"""
        strategy = params.get("strategy", "mean")
        columns = params.get("columns", df.columns)
        
        if strategy == "mean":
            df[columns] = df[columns].fillna(df[columns].mean())
        elif strategy == "median":
            df[columns] = df[columns].fillna(df[columns].median())
        elif strategy == "mode":
            df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
        elif strategy == "drop":
            df = df.dropna(subset=columns)
        elif strategy == "constant":
            value = params.get("value", 0)
            df[columns] = df[columns].fillna(value)
            
        return df
        
    def _scale_features(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Scale features based on specified parameters"""
        method = params.get("method", "standard")
        columns = params.get("columns", df.select_dtypes(include=['int64', 'float64']).columns)
        
        if method == "standard":
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "minmax":
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "robust":
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
            
        return df
        
    def _encode_features(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical features based on specified parameters"""
        method = params.get("method", "onehot")
        columns = params.get("columns", df.select_dtypes(include=['object', 'category']).columns)
        
        if method == "onehot":
            df = pd.get_dummies(df, columns=columns, prefix=columns)
        elif method == "label":
            for col in columns:
                df[col] = df[col].astype('category').cat.codes
        elif method == "binary":
            for col in columns:
                df[col] = (df[col] == params.get("positive_value", df[col].mode().iloc[0])).astype(int)
                
        return df
        
    def _select_features(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Select features based on specified parameters"""
        method = params.get("method", "correlation")
        
        if method == "correlation":
            threshold = params.get("threshold", 0.95)
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            df = df.drop(columns=to_drop)
            
        elif method == "variance":
            threshold = params.get("threshold", 0.01)
            selector = VarianceThreshold(threshold=threshold)
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            selector.fit(numeric_df)
            selected_features = numeric_df.columns[selector.get_support()].tolist()
            df = df[selected_features + [col for col in df.columns if col not in numeric_df.columns]]
            
        return df
        
    def _create_features(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Create new features based on specified parameters"""
        operations = params.get("operations", [])
        
        for operation in operations:
            op_type = operation["type"]
            
            if op_type == "interaction":
                col1, col2 = operation["columns"]
                df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
                
            elif op_type == "polynomial":
                col = operation["column"]
                degree = operation.get("degree", 2)
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d
                    
            elif op_type == "binning":
                col = operation["column"]
                bins = operation["bins"]
                labels = operation.get("labels", None)
                df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels)
                
            elif op_type == "datetime":
                col = operation["column"]
                features = operation.get("features", ["year", "month", "day"])
                df[col] = pd.to_datetime(df[col])
                for feature in features:
                    df[f"{col}_{feature}"] = getattr(df[col].dt, feature)
                    
        return df
        
    def _generate_missing_values_code(self, params: Dict[str, Any]) -> List[str]:
        """Generate code for handling missing values"""
        code_lines = []
        strategy = params.get("strategy", "mean")
        columns = params.get("columns", "df.columns")
        
        if strategy == "mean":
            code_lines.append(f"df[{columns}] = df[{columns}].fillna(df[{columns}].mean())")
        elif strategy == "median":
            code_lines.append(f"df[{columns}] = df[{columns}].fillna(df[{columns}].median())")
        elif strategy == "mode":
            code_lines.append(f"df[{columns}] = df[{columns}].fillna(df[{columns}].mode().iloc[0])")
        elif strategy == "drop":
            code_lines.append(f"df = df.dropna(subset={columns})")
        elif strategy == "constant":
            value = params.get("value", 0)
            code_lines.append(f"df[{columns}] = df[{columns}].fillna({value})")
            
        return code_lines
        
    def _generate_scaling_code(self, params: Dict[str, Any]) -> List[str]:
        """Generate code for feature scaling"""
        code_lines = []
        method = params.get("method", "standard")
        columns = params.get("columns", "df.select_dtypes(include=['int64', 'float64']).columns")
        
        if method == "standard":
            code_lines.append("from sklearn.preprocessing import StandardScaler")
            code_lines.append("scaler = StandardScaler()")
            code_lines.append(f"df[{columns}] = scaler.fit_transform(df[{columns}])")
        elif method == "minmax":
            code_lines.append("from sklearn.preprocessing import MinMaxScaler")
            code_lines.append("scaler = MinMaxScaler()")
            code_lines.append(f"df[{columns}] = scaler.fit_transform(df[{columns}])")
        elif method == "robust":
            code_lines.append("from sklearn.preprocessing import RobustScaler")
            code_lines.append("scaler = RobustScaler()")
            code_lines.append(f"df[{columns}] = scaler.fit_transform(df[{columns}])")
            
        return code_lines
        
    def _generate_encoding_code(self, params: Dict[str, Any]) -> List[str]:
        """Generate code for feature encoding"""
        code_lines = []
        method = params.get("method", "onehot")
        columns = params.get("columns", "df.select_dtypes(include=['object', 'category']).columns")
        
        if method == "onehot":
            code_lines.append(f"df = pd.get_dummies(df, columns={columns}, prefix={columns})")
        elif method == "label":
            code_lines.append(f"for col in {columns}:")
            code_lines.append("    df[col] = df[col].astype('category').cat.codes")
        elif method == "binary":
            positive_value = params.get("positive_value", "df[col].mode().iloc[0]")
            code_lines.append(f"for col in {columns}:")
            code_lines.append(f"    df[col] = (df[col] == {positive_value}).astype(int)")
            
        return code_lines
        
    def _generate_feature_selection_code(self, params: Dict[str, Any]) -> List[str]:
        """Generate code for feature selection"""
        code_lines = []
        method = params.get("method", "correlation")
        
        if method == "correlation":
            threshold = params.get("threshold", 0.95)
            code_lines.append("numeric_df = df.select_dtypes(include=['int64', 'float64'])")
            code_lines.append("corr_matrix = numeric_df.corr().abs()")
            code_lines.append("upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))")
            code_lines.append(f"to_drop = [column for column in upper.columns if any(upper[column] > {threshold})]")
            code_lines.append("df = df.drop(columns=to_drop)")
            
        elif method == "variance":
            threshold = params.get("threshold", 0.01)
            code_lines.append("from sklearn.feature_selection import VarianceThreshold")
            code_lines.append("numeric_df = df.select_dtypes(include=['int64', 'float64'])")
            code_lines.append(f"selector = VarianceThreshold(threshold={threshold})")
            code_lines.append("selector.fit(numeric_df)")
            code_lines.append("selected_features = numeric_df.columns[selector.get_support()].tolist()")
            code_lines.append("df = df[selected_features + [col for col in df.columns if col not in numeric_df.columns]]")
            
        return code_lines
        
    def _generate_feature_creation_code(self, params: Dict[str, Any]) -> List[str]:
        """Generate code for feature creation"""
        code_lines = []
        operations = params.get("operations", [])
        
        for operation in operations:
            op_type = operation["type"]
            
            if op_type == "interaction":
                col1, col2 = operation["columns"]
                code_lines.append(f"df['{col1}_{col2}_interaction'] = df['{col1}'] * df['{col2}']")
                
            elif op_type == "polynomial":
                col = operation["column"]
                degree = operation.get("degree", 2)
                for d in range(2, degree + 1):
                    code_lines.append(f"df['{col}_pow{d}'] = df['{col}'] ** {d}")
                    
            elif op_type == "binning":
                col = operation["column"]
                bins = operation["bins"]
                labels = operation.get("labels", None)
                code_lines.append(f"df['{col}_binned'] = pd.cut(df['{col}'], bins={bins}, labels={labels})")
                
            elif op_type == "datetime":
                col = operation["column"]
                features = operation.get("features", ["year", "month", "day"])
                code_lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'])")
                for feature in features:
                    code_lines.append(f"df['{col}_{feature}'] = df['{col}'].dt.{feature}")
                    
        return code_lines
        
    def analyze_data(self, dataset_id: str, analysis_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform advanced data analysis on a dataset
        
        Args:
            dataset_id: ID of the dataset to analyze
            analysis_type: Type of analysis to perform
            params: Additional parameters for the analysis
            
        Returns:
            Dict containing analysis results
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        df = self.datasets[dataset_id]
        results = {}
        
        if analysis_type == "descriptive_stats":
            results = self._calculate_descriptive_stats(df)
            
        elif analysis_type == "correlation_analysis":
            results = self._perform_correlation_analysis(df, params)
            
        elif analysis_type == "time_series_analysis":
            results = self._perform_time_series_analysis(df, params)
            
        elif analysis_type == "group_analysis":
            results = self._perform_group_analysis(df, params)
            
        elif analysis_type == "outlier_detection":
            results = self._detect_outliers(df, params)
            
        # Log the analysis
        self.logs.append({
            "operation": "analyze_data",
            "dataset_id": dataset_id,
            "analysis_type": analysis_type,
            "params": params,
            "results": results
        })
        
        return results
        
    def _calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics for the dataset"""
        stats = {
            "numeric_stats": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
        return stats
        
    def _perform_correlation_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis on numeric columns"""
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        method = params.get("method", "pearson")
        
        results = {
            "correlation_matrix": numeric_df.corr(method=method).to_dict(),
            "high_correlations": self._find_high_correlations(numeric_df, params.get("threshold", 0.7)),
            "method": method
        }
        return results
        
    def _perform_time_series_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform time series analysis on datetime columns"""
        time_col = params.get("time_column")
        if not time_col or time_col not in df.columns:
            raise ValueError("Time column not specified or not found")
            
        df[time_col] = pd.to_datetime(df[time_col])
        results = {
            "time_range": {
                "start": df[time_col].min().isoformat(),
                "end": df[time_col].max().isoformat()
            },
            "time_gaps": self._find_time_gaps(df, time_col),
            "seasonality": self._analyze_seasonality(df, time_col, params)
        }
        return results
        
    def _perform_group_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform group analysis on categorical columns"""
        group_cols = params.get("group_columns", [])
        agg_cols = params.get("aggregate_columns", df.select_dtypes(include=['int64', 'float64']).columns)
        agg_funcs = params.get("aggregate_functions", ["mean", "count", "sum"])
        
        if not group_cols:
            raise ValueError("No group columns specified")
            
        results = {
            "group_stats": df.groupby(group_cols)[agg_cols].agg(agg_funcs).to_dict(),
            "group_sizes": df.groupby(group_cols).size().to_dict(),
            "group_columns": group_cols,
            "aggregate_columns": agg_cols.tolist(),
            "aggregate_functions": agg_funcs
        }
        return results
        
    def _detect_outliers(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        method = params.get("method", "zscore")
        threshold = params.get("threshold", 3)
        columns = params.get("columns", df.select_dtypes(include=['int64', 'float64']).columns)
        
        results = {}
        for col in columns:
            if method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
            elif method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                
            results[col] = {
                "outlier_indices": outliers.index.tolist(),
                "outlier_values": outliers[col].tolist(),
                "method": method,
                "threshold": threshold
            }
            
        return results
        
    def _find_high_correlations(self, df: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Find highly correlated columns"""
        corr_matrix = df.corr().abs()
        high_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_correlations.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })
                    
        return high_correlations
        
    def _find_time_gaps(self, df: pd.DataFrame, time_col: str) -> List[Dict[str, Any]]:
        """Find gaps in time series data"""
        df = df.sort_values(time_col)
        time_diff = df[time_col].diff()
        gaps = df[time_diff > time_diff.mean() + 2 * time_diff.std()]
        
        return [{
            "start": row[time_col].isoformat(),
            "end": next_row[time_col].isoformat(),
            "gap_duration": (next_row[time_col] - row[time_col]).total_seconds()
        } for row, next_row in zip(gaps.itertuples(), gaps.iloc[1:].itertuples())]
        
    def _analyze_seasonality(self, df: pd.DataFrame, time_col: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonality in time series data"""
        value_col = params.get("value_column")
        if not value_col or value_col not in df.columns:
            raise ValueError("Value column not specified or not found")
            
        df = df.set_index(time_col)
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Calculate seasonal statistics
        results = {
            "hourly_stats": df.groupby('hour')[value_col].mean().to_dict(),
            "daily_stats": df.groupby('day_of_week')[value_col].mean().to_dict(),
            "monthly_stats": df.groupby('month')[value_col].mean().to_dict(),
            "quarterly_stats": df.groupby('quarter')[value_col].mean().to_dict()
        }
        
        return results
        
    def create_visualization(self, dataset_id: str, viz_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a visualization for a dataset
        
        Args:
            dataset_id: ID of the dataset to visualize
            viz_type: Type of visualization to create
            params: Parameters for the visualization
            
        Returns:
            Dict containing visualization data and metadata
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        df = self.datasets[dataset_id]
        viz_data = {}
        
        if viz_type == "scatter_plot":
            viz_data = self._create_scatter_plot(df, params)
            
        elif viz_type == "line_plot":
            viz_data = self._create_line_plot(df, params)
            
        elif viz_type == "bar_plot":
            viz_data = self._create_bar_plot(df, params)
            
        elif viz_type == "histogram":
            viz_data = self._create_histogram(df, params)
            
        elif viz_type == "box_plot":
            viz_data = self._create_box_plot(df, params)
            
        elif viz_type == "heatmap":
            viz_data = self._create_heatmap(df, params)
            
        elif viz_type == "pie_chart":
            viz_data = self._create_pie_chart(df, params)
            
        # Log the visualization
        self.logs.append({
            "operation": "create_visualization",
            "dataset_id": dataset_id,
            "viz_type": viz_type,
            "params": params,
            "viz_data": viz_data
        })
        
        return viz_data
        
    def _create_scatter_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a scatter plot visualization"""
        x_col = params.get("x_column")
        y_col = params.get("y_column")
        color_col = params.get("color_column")
        size_col = params.get("size_column")
        
        if not x_col or not y_col:
            raise ValueError("X and Y columns must be specified")
            
        viz_data = {
            "type": "scatter",
            "data": {
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "mode": "markers",
                "marker": {
                    "size": df[size_col].tolist() if size_col else 10,
                    "color": df[color_col].tolist() if color_col else None,
                    "colorscale": params.get("colorscale", "Viridis"),
                    "showscale": bool(color_col)
                }
            },
            "layout": {
                "title": params.get("title", f"{y_col} vs {x_col}"),
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col},
                "showlegend": bool(color_col)
            }
        }
        
        return viz_data
        
    def _create_line_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a line plot visualization"""
        x_col = params.get("x_column")
        y_cols = params.get("y_columns", [])
        
        if not x_col or not y_cols:
            raise ValueError("X column and at least one Y column must be specified")
            
        traces = []
        for y_col in y_cols:
            trace = {
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "name": y_col,
                "type": "scatter",
                "mode": "lines+markers"
            }
            traces.append(trace)
            
        viz_data = {
            "type": "line",
            "data": traces,
            "layout": {
                "title": params.get("title", f"Line Plot: {', '.join(y_cols)} vs {x_col}"),
                "xaxis": {"title": x_col},
                "yaxis": {"title": "Value"},
                "showlegend": True
            }
        }
        
        return viz_data
        
    def _create_bar_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a bar plot visualization"""
        x_col = params.get("x_column")
        y_col = params.get("y_column")
        group_col = params.get("group_column")
        
        if not x_col or not y_col:
            raise ValueError("X and Y columns must be specified")
            
        if group_col:
            traces = []
            for group in df[group_col].unique():
                group_df = df[df[group_col] == group]
                trace = {
                    "x": group_df[x_col].tolist(),
                    "y": group_df[y_col].tolist(),
                    "name": str(group),
                    "type": "bar"
                }
                traces.append(trace)
        else:
            traces = [{
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "type": "bar"
            }]
            
        viz_data = {
            "type": "bar",
            "data": traces,
            "layout": {
                "title": params.get("title", f"Bar Plot: {y_col} by {x_col}"),
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col},
                "barmode": "group" if group_col else "relative",
                "showlegend": bool(group_col)
            }
        }
        
        return viz_data
        
    def _create_histogram(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a histogram visualization"""
        column = params.get("column")
        bins = params.get("bins", 30)
        group_col = params.get("group_column")
        
        if not column:
            raise ValueError("Column must be specified")
            
        if group_col:
            traces = []
            for group in df[group_col].unique():
                group_df = df[df[group_col] == group]
                trace = {
                    "x": group_df[column].tolist(),
                    "name": str(group),
                    "type": "histogram",
                    "nbinsx": bins,
                    "opacity": 0.7
                }
                traces.append(trace)
        else:
            traces = [{
                "x": df[column].tolist(),
                "type": "histogram",
                "nbinsx": bins
            }]
            
        viz_data = {
            "type": "histogram",
            "data": traces,
            "layout": {
                "title": params.get("title", f"Histogram of {column}"),
                "xaxis": {"title": column},
                "yaxis": {"title": "Count"},
                "barmode": "overlay" if group_col else "relative",
                "showlegend": bool(group_col)
            }
        }
        
        return viz_data
        
    def _create_box_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a box plot visualization"""
        column = params.get("column")
        group_col = params.get("group_column")
        
        if not column:
            raise ValueError("Column must be specified")
            
        if group_col:
            traces = []
            for group in df[group_col].unique():
                group_df = df[df[group_col] == group]
                trace = {
                    "y": group_df[column].tolist(),
                    "name": str(group),
                    "type": "box"
                }
                traces.append(trace)
        else:
            traces = [{
                "y": df[column].tolist(),
                "type": "box"
            }]
            
        viz_data = {
            "type": "box",
            "data": traces,
            "layout": {
                "title": params.get("title", f"Box Plot of {column}"),
                "yaxis": {"title": column},
                "showlegend": bool(group_col)
            }
        }
        
        return viz_data
        
    def _create_heatmap(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a heatmap visualization"""
        columns = params.get("columns", df.select_dtypes(include=['int64', 'float64']).columns)
        
        if len(columns) < 2:
            raise ValueError("At least two numeric columns must be specified")
            
        corr_matrix = df[columns].corr()
        
        viz_data = {
            "type": "heatmap",
            "data": [{
                "z": corr_matrix.values.tolist(),
                "x": corr_matrix.columns.tolist(),
                "y": corr_matrix.index.tolist(),
                "type": "heatmap",
                "colorscale": params.get("colorscale", "RdBu")
            }],
            "layout": {
                "title": params.get("title", "Correlation Heatmap"),
                "xaxis": {"title": "Features"},
                "yaxis": {"title": "Features"}
            }
        }
        
        return viz_data
        
    def _create_pie_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pie chart visualization"""
        column = params.get("column")
        
        if not column:
            raise ValueError("Column must be specified")
            
        value_counts = df[column].value_counts()
        
        viz_data = {
            "type": "pie",
            "data": [{
                "values": value_counts.values.tolist(),
                "labels": value_counts.index.tolist(),
                "type": "pie",
                "hole": params.get("hole", 0)
            }],
            "layout": {
                "title": params.get("title", f"Distribution of {column}")
            }
        }
        
        return viz_data 