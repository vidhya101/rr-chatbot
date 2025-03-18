# dataviz_system.py - Main integration module for the data visualization system

import os
import logging
import traceback
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

from data_processor import DataProcessor
from analysis_engine import AnalysisEngine
from nlp_processor import NLPProcessor
from visualization_generator import VisualizationGenerator
from dashboard_generator import DashboardGenerator
from backend.data_processing.clustering_service import ClusteringService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataviz_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataVizSystem")


class DataVizSystem:
    """
    Main integration class for the data visualization system.
    Coordinates all components and provides a unified interface.
    
    This class brings together:
    - Data processing capabilities
    - Advanced analysis and machine learning
    - Natural language query understanding
    - Visualization generation
    - Interactive dashboard creation
    
    Attributes:
        data_processor: Handles data loading, cleaning, and transformation
        analysis_engine: Performs advanced analysis and machine learning
        nlp_processor: Processes natural language queries
        viz_generator: Creates data visualizations
        dashboard_generator: Creates interactive dashboards
    """
    
    def __init__(self):
        """Initialize the data visualization system."""
        # Create system components
        self.data_processor = DataProcessor()
        self.analysis_engine = AnalysisEngine()
        self.nlp_processor = NLPProcessor()
        self.viz_generator = VisualizationGenerator()
        self.dashboard_generator = DashboardGenerator()
        self.clustering_service = ClusteringService()
        
        # Connect components
        self.analysis_engine.set_data_processor(self.data_processor)
        self.viz_generator.set_data_processor(self.data_processor)
        self.dashboard_generator.set_data_processor(self.data_processor)
        self.dashboard_generator.set_viz_generator(self.viz_generator)
        
        # Keep track of history
        self.query_history = []
        self.viz_history = []
        
        # Initialize cache for optimization
        self.cache = {
            'insights': None,
            'analysis_summary': None,
            'last_data_update': None,
            'clustering_results': None
        }
        
        logger.info("DataVizSystem initialized")
    
    def load_data(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Load data from a source (file or DataFrame).
        
        Args:
            data_source: Path to file or DataFrame object
            **kwargs: Additional options for loading data
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If data source cannot be loaded
        """
        try:
            logger.info(f"Loading data from {data_source}")
            result = self.data_processor.load_data(data_source, **kwargs)
            
            # Update cache timestamp
            self.cache['last_data_update'] = pd.Timestamp.now()
            # Clear cached results
            self.cache['insights'] = None
            self.cache['analysis_summary'] = None
            
            logger.info(f"Data loaded successfully: {result.shape[0]} rows, {result.shape[1]} columns")
            return result
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def clean_data(self, **kwargs) -> pd.DataFrame:
        """
        Clean the loaded data.
        
        Args:
            **kwargs: Cleaning options (handle_missing, handle_outliers, etc.)
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ValueError: If data has not been loaded
        """
        try:
            logger.info("Cleaning data")
            result = self.data_processor.clean_data(**kwargs)
            
            # Update cache timestamp
            self.cache['last_data_update'] = pd.Timestamp.now()
            # Clear cached results
            self.cache['insights'] = None
            self.cache['analysis_summary'] = None
            
            logger.info("Data cleaned successfully")
            return result
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to clean data: {str(e)}")
    
    def engineer_features(self, feature_configs: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Engineer features from the data.
        
        Args:
            feature_configs: List of feature engineering configurations
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            ValueError: If data has not been loaded
        """
        try:
            logger.info("Engineering features")
            result = self.data_processor.engineer_features(feature_configs)
            
            # Update cache timestamp
            self.cache['last_data_update'] = pd.Timestamp.now()
            # Clear cached results
            self.cache['insights'] = None
            self.cache['analysis_summary'] = None
            
            logger.info("Features engineered successfully")
            return result
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to engineer features: {str(e)}")
    
    def analyze_data(self, analysis_type: str, **kwargs) -> Any:
        """
        Perform data analysis.
        
        Args:
            analysis_type: Type of analysis to perform
            **kwargs: Analysis parameters
            
        Returns:
            Analysis results
            
        Raises:
            ValueError: If analysis type is not supported or data is not available
        """
        try:
            logger.info(f"Performing {analysis_type} analysis")
            
            # Clear cached insights as new analysis might affect them
            self.cache['insights'] = None
            
            if analysis_type == 'outliers':
                return self.analysis_engine.detect_outliers(**kwargs)
            elif analysis_type == 'clustering':
                # Use the new clustering service
                df = self.data_processor.get_current_data()
                if df is None:
                    raise ValueError("No data available for clustering")
                    
                # Get clustering parameters
                method = kwargs.get('method', 'kmeans')
                features = kwargs.get('features', None)
                
                # Perform clustering
                results = self.clustering_service.cluster_data(
                    df=df,
                    method=method,
                    features=features,
                    **kwargs
                )
                
                # Cache the results
                self.cache['clustering_results'] = results
                return results
            elif analysis_type == 'dimensionality_reduction':
                return self.analysis_engine.reduce_dimensions(**kwargs)
            elif analysis_type == 'prediction':
                return self.analysis_engine.train_prediction_model(**kwargs)
            elif analysis_type == 'time_series':
                return self.analysis_engine.analyze_time_series(**kwargs)
            elif analysis_type == 'correlations':
                return self.analysis_engine.find_correlations(**kwargs)
            elif analysis_type == 'distribution':
                return self.analysis_engine.analyze_distribution(**kwargs)
            elif analysis_type == 'feature_importance':
                return self.analysis_engine.get_feature_importance(**kwargs)
            else:
                logger.error(f"Unsupported analysis type: {analysis_type}")
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Analysis failed: {str(e)}")
    
    def get_insights(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get insights from the data.
        
        Args:
            force_refresh: Whether to force recalculation of insights
            
        Returns:
            List of insights
            
        Raises:
            ValueError: If data has not been loaded
        """
        try:
            # Check cache first
            if not force_refresh and self.cache['insights'] is not None:
                logger.info("Returning cached insights")
                return self.cache['insights']
            
            logger.info("Retrieving insights")
            insights = self.data_processor.get_insights()
            
            # Add insights from analysis engine
            if hasattr(self.analysis_engine, 'get_analysis_summary'):
                analysis_summary = self.analysis_engine.get_analysis_summary()
                self.cache['analysis_summary'] = analysis_summary
                
                # Convert analysis summary to insights
                for analysis_type, analysis in analysis_summary.get('analyses', {}).items():
                    if analysis_type == 'correlations' and 'top_correlation' in analysis and analysis['top_correlation']:
                        top_corr = analysis['top_correlation']
                        insights.append({
                            'type': 'correlation',
                            'title': f"Strong {top_corr['correlation_type']} correlation detected",
                            'description': f"Found a {top_corr['correlation_strength']} {top_corr['correlation_type']} correlation ({top_corr['correlation']:.2f}) between {top_corr['column1']} and {top_corr['column2']}",
                            'details': top_corr
                        })
                    
                    elif analysis_type == 'outlier_detection' and analysis.get('outliers_found', 0) > 0:
                        insights.append({
                            'type': 'outlier',
                            'title': "Outliers detected in data",
                            'description': f"Found {analysis['outliers_found']} outliers ({analysis['outlier_percentage']:.1f}%) using {analysis['method']} method",
                            'details': analysis
                        })
                    
                    elif analysis_type == 'clustering':
                        # Get enhanced clustering insights from cache
                        clustering_results = self.cache.get('clustering_results')
                        if clustering_results:
                            # Add main clustering insight
                            insights.append({
                                'type': 'clustering',
                                'title': f"Data clustered into {clustering_results['n_clusters']} groups",
                                'description': f"Found {clustering_results['n_clusters']} natural clusters using {clustering_results['method']} clustering",
                                'details': clustering_results
                            })
                            
                            # Add cluster statistics insights
                            for cluster_id, stats in clustering_results.get('cluster_stats', {}).items():
                                insights.append({
                                    'type': 'cluster_stats',
                                    'title': f"Cluster {cluster_id} characteristics",
                                    'description': f"Cluster contains {stats['size']} points ({stats['percentage']:.1f}% of data)",
                                    'details': stats
                                })
                            
                            # Add validation score insights
                            if clustering_results.get('validation_scores'):
                                scores = clustering_results['validation_scores']
                                if isinstance(scores, dict) and 'silhouette' in scores:
                                    quality = 'high' if scores['silhouette'] > 0.5 else 'moderate' if scores['silhouette'] > 0.25 else 'low'
                                    insights.append({
                                        'type': 'cluster_validation',
                                        'title': 'Cluster quality assessment',
                                        'description': f"Clustering shows {quality} quality (silhouette score: {scores['silhouette']:.2f})",
                                        'details': scores
                                    })
                            
                            # Add feature importance insights
                            if clustering_results.get('feature_importance'):
                                importance = clustering_results['feature_importance']
                                if isinstance(importance, dict):
                                    # Get top 3 most important features
                                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                                    features_str = ', '.join(f"{feat} ({imp:.2f})" for feat, imp in top_features)
                                    insights.append({
                                        'type': 'cluster_features',
                                        'title': 'Key clustering features identified',
                                        'description': f"Top features for clustering: {features_str}",
                                        'details': {'top_features': dict(top_features)}
                                    })
                        else:
                            insights.append({
                                'type': 'clustering',
                                'title': f"Data clustered into {analysis['clusters_found']} groups",
                                'description': f"Found {analysis['clusters_found']} natural clusters in the data using {analysis['method']} clustering",
                                'details': analysis
                            })
                        
                    elif analysis_type == 'distribution':
                        dist_info = analysis.get('statistics', {})
                        if dist_info.get('skewness') is not None:
                            skew_type = 'positive' if dist_info['skewness'] > 0 else 'negative'
                            insights.append({
                                'type': 'distribution',
                                'title': f"Distribution pattern detected",
                                'description': f"Found {skew_type} skewness ({dist_info['skewness']:.2f}) in the distribution",
                                'details': dist_info
                            })
                            
                    elif analysis_type == 'time_series' and 'seasonality' in analysis:
                        insights.append({
                            'type': 'time_series',
                            'title': "Time series patterns detected",
                            'description': f"Detected {analysis['seasonality']} seasonality in the time series",
                            'details': analysis
                        })
            
            # Cache the insights
            self.cache['insights'] = insights
            
            logger.info(f"Retrieved {len(insights)} insights")
            return insights
        except Exception as e:
            logger.error(f"Error getting insights: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to get insights: {str(e)}")
    
    def create_visualization(self, viz_type: str, **kwargs) -> Any:
        """
        Create a visualization.
        
        Args:
            viz_type: Type of visualization
            **kwargs: Visualization parameters
            
        Returns:
            Visualization figure
            
        Raises:
            ValueError: If visualization type is not supported or data is not available
        """
        try:
            logger.info(f"Creating {viz_type} visualization")
            
            # Handle clustering visualizations
            if viz_type in ['cluster_pca', 'cluster_scatter']:
                clustering_results = self.cache.get('clustering_results')
                if not clustering_results:
                    logger.warning("No clustering results found, falling back to scatter plot")
                    return self.viz_generator.create_visualization('scatter', **kwargs)
                
                # Add clustering information to kwargs
                kwargs.update({
                    'cluster_labels': clustering_results['cluster_labels'],
                    'cluster_centers': clustering_results['cluster_centers'],
                    'visualization_data': clustering_results['visualization_data'],
                    'method': clustering_results['method']
                })
                
                # For PCA visualization, use the PCA coordinates
                if viz_type == 'cluster_pca' and 'visualization_data' in clustering_results:
                    viz_data = clustering_results['visualization_data']
                    if 'pca_coordinates' in viz_data:
                        kwargs['pca_coordinates'] = viz_data['pca_coordinates']
                        if 'explained_variance' in viz_data:
                            kwargs['explained_variance'] = viz_data['explained_variance']
                        if 'cluster_centers_2d' in viz_data:
                            kwargs['cluster_centers_2d'] = viz_data['cluster_centers_2d']
            
            figure = self.viz_generator.create_visualization(viz_type, **kwargs)
            
            # Store in visualization history
            viz_record = {
                'type': viz_type,
                'params': kwargs,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            self.viz_history.append(viz_record)
            
            # Trim history if too long
            if len(self.viz_history) > 100:
                self.viz_history = self.viz_history[-100:]
            
            logger.info(f"{viz_type} visualization created successfully")
            return figure
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to create visualization: {str(e)}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Processed query information including visualizations and insights
            
        Raises:
            ValueError: If query cannot be processed
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Store query in history
            query_record = {
                'query': query,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Parse the query
            query_data = self.nlp_processor.parse_query(query)
            query_record['parsed_data'] = query_data
            
            # Extract visualization intent
            viz_type = query_data.get('visualization_type')
            if not viz_type:
                # Try to recommend visualization type based on query
                viz_type = self.recommend_visualization(query_data)
                query_data['visualization_type'] = viz_type
            
            # Extract columns
            columns = query_data.get('columns', [])
            if not columns:
                # Try to extract columns from query
                columns = self.nlp_processor.extract_columns(query)
                if not columns:
                    # Use numeric columns as default
                    columns = self.data_processor.get_numeric_columns()
                query_data['columns'] = columns
            
            # Create visualization
            viz_params = {
                'columns': columns,
                **query_data.get('parameters', {})
            }
            figure = self.create_visualization(viz_type, **viz_params)
            
            # Get relevant insights
            insights = self.get_insights_for_query(query_data)
            
            # Update query record with results
            query_record.update({
                'visualization_type': viz_type,
                'columns_used': columns,
                'insights_count': len(insights)
            })
            
            # Add to history
            self.query_history.append(query_record)
            
            # Trim history if too long
            if len(self.query_history) > 100:
                self.query_history = self.query_history[-100:]
            
            result = {
                'query_data': query_data,
                'visualization': figure,
                'insights': insights,
                'metadata': {
                    'timestamp': query_record['timestamp'],
                    'processing_time': (pd.Timestamp.now() - pd.Timestamp(query_record['timestamp'])).total_seconds()
                }
            }
            
            logger.info(f"Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to process query: {str(e)}")
    
    def recommend_visualization(self, query_data: Dict[str, Any]) -> str:
        """
        Recommend visualization type based on query intent and data characteristics.
        
        Args:
            query_data: Parsed query data
            
        Returns:
            Recommended visualization type
        """
        try:
            intent = query_data.get('intent', '').lower()
            columns = query_data.get('columns', [])
            
            # Get column types
            numeric_cols = self.data_processor.get_numeric_columns()
            categorical_cols = self.data_processor.get_categorical_columns()
            temporal_cols = self.data_processor.get_temporal_columns()
            
            # Check for clustering intent
            if any(word in intent for word in ['cluster', 'group', 'segment']):
                # If we have clustering results in cache, use them
                if self.cache.get('clustering_results'):
                    if len(numeric_cols) > 2:
                        return 'cluster_pca'  # PCA-based cluster visualization
                    return 'cluster_scatter'  # Direct scatter plot for 2D data
                return 'scatter'  # Default to scatter if no clustering results
            
            # Check for time series intent
            if any(word in intent for word in ['trend', 'over time', 'timeline']):
                if temporal_cols:
                    return 'line'
            
            # Check for comparison intent
            if any(word in intent for word in ['compare', 'comparison', 'versus']):
                if len(categorical_cols) > 0:
                    return 'bar'
                return 'scatter'
            
            # Check for distribution intent
            if any(word in intent for word in ['distribution', 'spread']):
                return 'histogram'
            
            # Check for relationship intent
            if any(word in intent for word in ['correlation', 'relationship']):
                if len(numeric_cols) >= 2:
                    return 'scatter'
                return 'heatmap'
            
            # Check for composition intent
            if any(word in intent for word in ['composition', 'breakdown']):
                return 'pie'
            
            # Default based on number and types of columns
            if len(columns) == 1:
                if columns[0] in numeric_cols:
                    return 'histogram'
                return 'bar'
            elif len(columns) == 2:
                if all(col in numeric_cols for col in columns):
                    return 'scatter'
                return 'bar'
            else:
                return 'bar'
            
        except Exception as e:
            logger.warning(f"Error recommending visualization: {str(e)}")
            return 'bar'  # Safe default
    
    def get_insights_for_query(self, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get relevant insights based on query intent.
        
        Args:
            query_data: Parsed query data
            
        Returns:
            List of relevant insights
        """
        try:
            all_insights = self.get_insights()
            if not all_insights:
                return []
            
            intent = query_data.get('intent', '').lower()
            columns = set(query_data.get('columns', []))
            
            # Filter insights based on relevance
            relevant_insights = []
            
            for insight in all_insights:
                # Check if insight is related to query columns
                if 'details' in insight:
                    insight_columns = set()
                    if 'columns' in insight['details']:
                        insight_columns.update(insight['details']['columns'])
                    if 'column1' in insight['details']:
                        insight_columns.add(insight['details']['column1'])
                    if 'column2' in insight['details']:
                        insight_columns.add(insight['details']['column2'])
                    
                    # Add insight if it involves any queried columns
                    if columns & insight_columns:
                        relevant_insights.append(insight)
                        continue
                
                # Check if insight type matches query intent
                if any(word in intent for word in ['correlation', 'relationship']) and insight['type'] == 'correlation':
                    relevant_insights.append(insight)
                elif any(word in intent for word in ['cluster', 'group']) and insight['type'] == 'clustering':
                    relevant_insights.append(insight)
                elif any(word in intent for word in ['outlier', 'anomaly']) and insight['type'] == 'outlier':
                    relevant_insights.append(insight)
                elif any(word in intent for word in ['distribution', 'spread']) and insight['type'] == 'distribution':
                    relevant_insights.append(insight)
                elif any(word in intent for word in ['trend', 'time']) and insight['type'] == 'time_series':
                    relevant_insights.append(insight)
            
            return relevant_insights
            
        except Exception as e:
            logger.warning(f"Error getting insights for query: {str(e)}")
            return []
    
    def export_dashboard(self, config: Dict[str, Any]) -> str:
        """
        Export dashboard as HTML or PDF.
        
        Args:
            config: Dashboard configuration
            
        Returns:
            Path to exported file
            
        Raises:
            ValueError: If export fails
        """
        try:
            logger.info("Exporting dashboard")
            
            # Get insights if not provided
            if 'insights' not in config:
                config['insights'] = self.get_insights()
            
            # Get current timestamp if not provided
            if 'timestamp' not in config:
                config['timestamp'] = pd.Timestamp.now().isoformat()
            
            # Export dashboard
            result = self.dashboard_generator.export_dashboard(config)
            
            logger.info(f"Dashboard exported successfully to {result}")
            return result
        except Exception as e:
            logger.error(f"Error exporting dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to export dashboard: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.
        
        Returns:
            Dictionary with system status information
        """
        try:
            return {
                'data_loaded': self.data_processor.data is not None,
                'last_data_update': self.cache['last_data_update'],
                'total_queries': len(self.query_history),
                'total_visualizations': len(self.viz_history),
                'available_analyses': list(self.analysis_engine.results.keys()),
                'cache_status': {
                    'insights_cached': self.cache['insights'] is not None,
                    'analysis_summary_cached': self.cache['analysis_summary'] is not None
                },
                'components_status': {
                    'data_processor': 'active',
                    'analysis_engine': 'active',
                    'nlp_processor': 'active',
                    'viz_generator': 'active',
                    'dashboard_generator': 'active'
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'error': str(e),
                'status': 'error'
            } 