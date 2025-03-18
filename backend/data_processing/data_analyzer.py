"""
Data Analyzer Module

This module handles data analysis and insights generation, including:
- Statistical analysis
- Correlation analysis
- Trend analysis
- Predictive modeling
- Insights generation
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/system_logs/data_analyzer.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DataAnalyzer:
    """
    Class for analyzing data and generating insights.
    """
    
    def __init__(self):
        """Initialize the DataAnalyzer class."""
        self.models = {}
    
    def analyze_data(self, df):
        """
        Analyze the data and return insights.
        
        Args:
            df (pandas.DataFrame): The data to analyze.
            
        Returns:
            dict: Analysis results and insights.
        """
        try:
            logger.info("Starting data analysis")
            
            # Check if df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error("Input is not a pandas DataFrame")
                raise ValueError("Input is not a pandas DataFrame")
            
            # Get statistical analysis
            statistical_analysis = self._perform_statistical_analysis(df)
            
            # Get correlation analysis
            correlation_analysis = self._perform_correlation_analysis(df)
            
            # Get trend analysis
            trend_analysis = self._perform_trend_analysis(df)
            
            # Get predictive analysis
            predictive_analysis = self._perform_predictive_analysis(df)
            
            # Generate insights
            insights = self._generate_insights(df, statistical_analysis, correlation_analysis, trend_analysis, predictive_analysis)
            
            # Combine all analyses
            analysis_results = {
                "statistical_analysis": statistical_analysis,
                "correlation_analysis": correlation_analysis,
                "trend_analysis": trend_analysis,
                "predictive_analysis": predictive_analysis,
                "insights": insights
            }
            
            logger.info("Data analysis completed successfully")
            
            return analysis_results
        
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            raise
    
    def _perform_statistical_analysis(self, df):
        """
        Perform statistical analysis on the data.
        
        Args:
            df (pandas.DataFrame): The data to analyze.
            
        Returns:
            dict: Statistical analysis results.
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found for statistical analysis"}
        
        # Calculate basic statistics
        basic_stats = numeric_df.describe().to_dict()
        
        # Calculate additional statistics
        skewness = numeric_df.skew().to_dict()
        kurtosis = numeric_df.kurtosis().to_dict()
        
        # Perform normality tests
        normality_tests = {}
        for col in numeric_df.columns:
            if len(numeric_df[col].dropna()) >= 8:  # Need at least 8 observations for shapiro test
                shapiro_test = stats.shapiro(numeric_df[col].dropna())
                normality_tests[col] = {
                    "test": "Shapiro-Wilk",
                    "statistic": float(shapiro_test[0]),
                    "p_value": float(shapiro_test[1]),
                    "is_normal": shapiro_test[1] > 0.05,
                }
        
        # Perform t-tests for each numeric column (comparing to the mean)
        t_tests = {}
        for col in numeric_df.columns:
            if len(numeric_df[col].dropna()) >= 8:
                t_test = stats.ttest_1samp(numeric_df[col].dropna(), numeric_df[col].mean())
                t_tests[col] = {
                    "test": "One-sample t-test",
                    "statistic": float(t_test[0]),
                    "p_value": float(t_test[1]),
                    "significant": t_test[1] < 0.05,
                }
        
        return {
            "basic_stats": basic_stats,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "normality_tests": normality_tests,
            "t_tests": t_tests,
        }
    
    def _perform_correlation_analysis(self, df):
        """
        Perform correlation analysis on the data.
        
        Args:
            df (pandas.DataFrame): The data to analyze.
            
        Returns:
            dict: Correlation analysis results.
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
        
        # Sort strong correlations by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x["pearson_correlation"]), reverse=True)
        
        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "strong_correlations": strong_correlations,
        }
    
    def _perform_trend_analysis(self, df):
        """
        Perform trend analysis on the data.
        
        Args:
            df (pandas.DataFrame): The data to analyze.
            
        Returns:
            dict: Trend analysis results.
        """
        # Check if there are datetime columns
        datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if not datetime_columns:
            return {"message": "No datetime columns found for trend analysis"}
        
        trend_results = {}
        
        # For each datetime column, analyze trends in numeric columns
        for date_col in datetime_columns:
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_columns:
                trend_results[date_col] = {"message": "No numeric columns found for trend analysis"}
                continue
            
            # Group by date and calculate statistics
            date_trends = {}
            
            # Try different time periods (year, month, day)
            for period in ['year', 'month', 'day']:
                try:
                    if period == 'year':
                        grouped = df.groupby(df[date_col].dt.year)
                    elif period == 'month':
                        grouped = df.groupby([df[date_col].dt.year, df[date_col].dt.month])
                    else:  # day
                        grouped = df.groupby([df[date_col].dt.year, df[date_col].dt.month, df[date_col].dt.day])
                    
                    # Calculate statistics for each numeric column
                    period_trends = {}
                    for col in numeric_columns:
                        stats = grouped[col].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
                        
                        # Convert to dict for JSON serialization
                        period_trends[col] = stats.to_dict(orient='records')
                    
                    date_trends[period] = period_trends
                except Exception as e:
                    logger.warning(f"Error calculating {period} trends for {date_col}: {str(e)}")
            
            trend_results[date_col] = date_trends
        
        return trend_results
    
    def _perform_predictive_analysis(self, df):
        """
        Perform predictive analysis on the data.
        
        Args:
            df (pandas.DataFrame): The data to analyze.
            
        Returns:
            dict: Predictive analysis results.
        """
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_columns) < 2:
            return {"message": "Not enough numeric columns for predictive analysis"}
        
        predictive_results = {}
        
        # Try to predict each numeric column using other numeric columns
        for target_col in numeric_columns:
            # Skip if too many missing values
            if df[target_col].isnull().sum() / len(df) > 0.2:
                predictive_results[target_col] = {"message": "Too many missing values for prediction"}
                continue
            
            # Get feature columns (excluding target)
            feature_cols = [col for col in numeric_columns if col != target_col]
            
            # Skip if no features
            if not feature_cols:
                predictive_results[target_col] = {"message": "No feature columns for prediction"}
                continue
            
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train a simple linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Get feature importance
                feature_importance = dict(zip(feature_cols, model.coef_))
                
                # Store model
                self.models[target_col] = model
                
                # Store results
                predictive_results[target_col] = {
                    "model_type": "Linear Regression",
                    "mse": float(mse),
                    "r2": float(r2),
                    "feature_importance": feature_importance,
                }
            except Exception as e:
                logger.warning(f"Error performing predictive analysis for {target_col}: {str(e)}")
                predictive_results[target_col] = {"message": f"Error in prediction: {str(e)}"}
        
        return predictive_results
    
    def _generate_insights(self, df, statistical_analysis, correlation_analysis, trend_analysis, predictive_analysis):
        """
        Generate insights from the analysis results.
        
        Args:
            df (pandas.DataFrame): The data.
            statistical_analysis (dict): Statistical analysis results.
            correlation_analysis (dict): Correlation analysis results.
            trend_analysis (dict): Trend analysis results.
            predictive_analysis (dict): Predictive analysis results.
            
        Returns:
            list: Insights generated from the analysis.
        """
        insights = []
        
        # Generate insights from statistical analysis
        if "basic_stats" in statistical_analysis:
            for col, stats in statistical_analysis["basic_stats"].items():
                if "mean" in stats and "50%" in stats and abs(stats["mean"] - stats["50%"]) > 0.5 * stats["std"]:
                    insights.append({
                        "type": "statistical",
                        "column": col,
                        "insight": f"The distribution of {col} is skewed (mean: {stats['mean']:.2f}, median: {stats['50%']:.2f}).",
                        "importance": "medium"
                    })
        
        # Generate insights from correlation analysis
        if "strong_correlations" in correlation_analysis:
            for corr in correlation_analysis["strong_correlations"][:5]:  # Top 5 correlations
                col1 = corr["column1"]
                col2 = corr["column2"]
                corr_value = corr["pearson_correlation"]
                
                if abs(corr_value) > 0.9:
                    importance = "high"
                elif abs(corr_value) > 0.7:
                    importance = "medium"
                else:
                    importance = "low"
                
                insights.append({
                    "type": "correlation",
                    "columns": [col1, col2],
                    "insight": f"Strong {'positive' if corr_value > 0 else 'negative'} correlation ({corr_value:.2f}) between {col1} and {col2}.",
                    "importance": importance
                })
        
        # Generate insights from trend analysis
        for date_col, trends in trend_analysis.items():
            if isinstance(trends, dict) and "year" in trends:
                for col, year_data in trends["year"].items():
                    if len(year_data) > 1:  # At least 2 years of data
                        first_year = year_data[0]
                        last_year = year_data[-1]
                        
                        if "mean" in first_year and "mean" in last_year:
                            change = last_year["mean"] - first_year["mean"]
                            percent_change = (change / first_year["mean"]) * 100 if first_year["mean"] != 0 else float('inf')
                            
                            if abs(percent_change) > 50:
                                importance = "high"
                            elif abs(percent_change) > 20:
                                importance = "medium"
                            else:
                                importance = "low"
                            
                            insights.append({
                                "type": "trend",
                                "columns": [date_col, col],
                                "insight": f"{col} has {'increased' if change > 0 else 'decreased'} by {abs(percent_change):.1f}% from {first_year['year']} to {last_year['year']}.",
                                "importance": importance
                            })
        
        # Generate insights from predictive analysis
        for col, pred_results in predictive_analysis.items():
            if isinstance(pred_results, dict) and "r2" in pred_results:
                r2 = pred_results["r2"]
                
                if r2 > 0.8:
                    importance = "high"
                elif r2 > 0.5:
                    importance = "medium"
                else:
                    importance = "low"
                
                # Get top features
                if "feature_importance" in pred_results:
                    feature_importance = pred_results["feature_importance"]
                    top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    top_features_str = ", ".join([f"{feat} ({imp:.2f})" for feat, imp in top_features])
                    
                    insights.append({
                        "type": "predictive",
                        "column": col,
                        "insight": f"{col} can be predicted with {r2:.2f} R² score. Top predictors: {top_features_str}.",
                        "importance": importance
                    })
        
        # Sort insights by importance
        importance_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: importance_order.get(x["importance"], 3))
        
        return insights
    
    def get_summary(self, df):
        """
        Get a summary of the data.
        
        Args:
            df (pandas.DataFrame): The data to summarize.
            
        Returns:
            dict: Summary of the data.
        """
        try:
            # Check if df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error("Input is not a pandas DataFrame")
                raise ValueError("Input is not a pandas DataFrame")
            
            # Get basic information
            basic_info = {
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            }
            
            # Get data types
            dtypes = df.dtypes.astype(str).to_dict()
            
            # Get missing values
            missing_values = {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_percentage": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                "columns_with_missing": [col for col in df.columns if df[col].isnull().any()],
            }
            
            # Get numeric summary
            numeric_df = df.select_dtypes(include=['number'])
            numeric_summary = {}
            
            if not numeric_df.empty:
                numeric_summary = numeric_df.describe().to_dict()
            
            # Get categorical summary
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
            
            # Combine all summaries
            summary = {
                "basic_info": basic_info,
                "dtypes": dtypes,
                "missing_values": missing_values,
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary,
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}")
            raise
    
    def predict(self, df, target_column):
        """
        Make predictions using a trained model.
        
        Args:
            df (pandas.DataFrame): The data to make predictions on.
            target_column (str): The column to predict.
            
        Returns:
            numpy.ndarray: Predictions.
        """
        try:
            # Check if model exists
            if target_column not in self.models:
                logger.error(f"No model found for {target_column}")
                raise ValueError(f"No model found for {target_column}")
            
            # Get the model
            model = self.models[target_column]
            
            # Get feature columns
            feature_cols = [col for col in df.columns if col != target_column and df[col].dtype.kind in 'if']
            
            # Prepare data
            X = df[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Make predictions
            predictions = model.predict(X)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise 