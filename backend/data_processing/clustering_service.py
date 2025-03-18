"""
Clustering Service Module

This module provides comprehensive clustering functionality including:
- Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
- Cluster analysis and statistics
- Cluster visualization support
- Cluster validation metrics
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging
from typing import Dict, List, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/system_logs/clustering.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ClusteringService:
    """
    Service for performing clustering analysis on data.
    """
    
    def __init__(self):
        """Initialize the ClusteringService."""
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_centers = None
        self.cluster_stats = None
        self.validation_scores = None
        self.feature_importance = None
        
    def cluster_data(self, df: pd.DataFrame, method: str = 'kmeans', 
                    features: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform clustering analysis on the data.
        
        Args:
            df: DataFrame containing the data
            method: Clustering method ('kmeans', 'dbscan', or 'hierarchical')
            features: List of features to use for clustering
            **kwargs: Additional parameters for clustering algorithms
                - n_clusters: Number of clusters for k-means/hierarchical
                - eps: DBSCAN epsilon parameter
                - min_samples: DBSCAN min_samples parameter
                
        Returns:
            Dictionary containing clustering results and statistics
        """
        try:
            logger.info(f"Starting clustering analysis using {method}")
            
            # Validate and prepare data
            if features is None:
                features = df.select_dtypes(include=['number']).columns.tolist()
            
            if not features:
                raise ValueError("No numeric features available for clustering")
            
            # Prepare data
            X = df[features].copy()
            X = self._prepare_data(X)
            
            # Perform clustering
            if method == 'kmeans':
                n_clusters = kwargs.get('n_clusters', 5)
                self._perform_kmeans(X, n_clusters)
            elif method == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                self._perform_dbscan(X, eps, min_samples)
            elif method == 'hierarchical':
                n_clusters = kwargs.get('n_clusters', 5)
                self._perform_hierarchical(X, n_clusters)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Calculate cluster statistics
            self._calculate_cluster_stats(X, features)
            
            # Calculate validation scores
            self._calculate_validation_scores(X)
            
            # Calculate feature importance
            self._calculate_feature_importance(X, features)
            
            # Prepare visualization data
            viz_data = self._prepare_visualization_data(X, features)
            
            # Compile results
            results = {
                'method': method,
                'n_clusters': len(np.unique(self.cluster_labels)),
                'cluster_labels': self.cluster_labels.tolist(),
                'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
                'cluster_stats': self.cluster_stats,
                'validation_scores': self.validation_scores,
                'feature_importance': self.feature_importance,
                'visualization_data': viz_data
            }
            
            logger.info("Clustering analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            raise
            
    def _prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare data for clustering."""
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale the data
        return self.scaler.fit_transform(X)
        
    def _perform_kmeans(self, X: np.ndarray, n_clusters: int):
        """Perform K-means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        
    def _perform_dbscan(self, X: np.ndarray, eps: float, min_samples: int):
        """Perform DBSCAN clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(X)
        
        # Calculate cluster centers as mean of points in each cluster
        unique_labels = np.unique(self.cluster_labels)
        self.cluster_centers = np.array([
            X[self.cluster_labels == label].mean(axis=0)
            for label in unique_labels if label != -1
        ])
        
    def _perform_hierarchical(self, X: np.ndarray, n_clusters: int):
        """Perform hierarchical clustering."""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.cluster_labels = hierarchical.fit_predict(X)
        
        # Calculate cluster centers as mean of points in each cluster
        unique_labels = np.unique(self.cluster_labels)
        self.cluster_centers = np.array([
            X[self.cluster_labels == label].mean(axis=0)
            for label in unique_labels
        ])
        
    def _calculate_cluster_stats(self, X: np.ndarray, features: List[str]):
        """Calculate detailed statistics for each cluster."""
        unique_labels = np.unique(self.cluster_labels)
        stats = {}
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            cluster_points = X[self.cluster_labels == label]
            cluster_size = len(cluster_points)
            
            # Calculate basic statistics
            stats[f'cluster_{label}'] = {
                'size': cluster_size,
                'percentage': (cluster_size / len(X)) * 100,
                'density': cluster_size / len(X),
                'mean_distance_to_center': np.mean(
                    np.linalg.norm(cluster_points - self.cluster_centers[label], axis=1)
                ) if self.cluster_centers is not None else None,
                'std_distance_to_center': np.std(
                    np.linalg.norm(cluster_points - self.cluster_centers[label], axis=1)
                ) if self.cluster_centers is not None else None,
                'feature_means': dict(zip(features, np.mean(cluster_points, axis=0))),
                'feature_stds': dict(zip(features, np.std(cluster_points, axis=0)))
            }
            
        self.cluster_stats = stats
        
    def _calculate_validation_scores(self, X: np.ndarray):
        """Calculate clustering validation scores."""
        try:
            # Skip validation for DBSCAN noise points
            valid_points = self.cluster_labels != -1
            X_valid = X[valid_points]
            labels_valid = self.cluster_labels[valid_points]
            
            if len(np.unique(labels_valid)) > 1:
                self.validation_scores = {
                    'silhouette': float(silhouette_score(X_valid, labels_valid)),
                    'calinski_harabasz': float(calinski_harabasz_score(X_valid, labels_valid)),
                    'davies_bouldin': float(davies_bouldin_score(X_valid, labels_valid))
                }
            else:
                self.validation_scores = {
                    'message': 'Not enough valid clusters for validation scores'
                }
        except Exception as e:
            logger.warning(f"Error calculating validation scores: {str(e)}")
            self.validation_scores = {
                'message': f'Error calculating validation scores: {str(e)}'
            }
            
    def _calculate_feature_importance(self, X: np.ndarray, features: List[str]):
        """Calculate feature importance for clustering."""
        try:
            # Calculate feature importance based on cluster separation
            importance_scores = {}
            
            for i, feature in enumerate(features):
                # Calculate the ratio of between-cluster to within-cluster variance
                cluster_means = np.array([
                    np.mean(X[self.cluster_labels == label, i])
                    for label in np.unique(self.cluster_labels) if label != -1
                ])
                
                overall_mean = np.mean(X[:, i])
                
                between_cluster_var = np.sum(
                    np.square(cluster_means - overall_mean)
                )
                
                within_cluster_var = np.sum([
                    np.sum(np.square(X[self.cluster_labels == label, i] - cluster_means[j]))
                    for j, label in enumerate(np.unique(self.cluster_labels)) if label != -1
                ])
                
                if within_cluster_var == 0:
                    importance_scores[feature] = 1.0
                else:
                    importance_scores[feature] = between_cluster_var / within_cluster_var
                    
            # Normalize scores
            total_score = sum(importance_scores.values())
            self.feature_importance = {
                feature: score / total_score
                for feature, score in importance_scores.items()
            }
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {str(e)}")
            self.feature_importance = {
                'message': f'Error calculating feature importance: {str(e)}'
            }
            
    def _prepare_visualization_data(self, X: np.ndarray, features: List[str]) -> Dict[str, Any]:
        """Prepare data for visualization."""
        try:
            viz_data = {}
            
            # If more than 2 dimensions, use PCA for visualization
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                explained_variance = pca.explained_variance_ratio_
                
                viz_data['pca_coordinates'] = X_2d.tolist()
                viz_data['explained_variance'] = explained_variance.tolist()
                
                if self.cluster_centers is not None:
                    centers_2d = pca.transform(self.cluster_centers)
                    viz_data['cluster_centers_2d'] = centers_2d.tolist()
            else:
                viz_data['coordinates'] = X.tolist()
                if self.cluster_centers is not None:
                    viz_data['cluster_centers'] = self.cluster_centers.tolist()
                    
            return viz_data
            
        except Exception as e:
            logger.warning(f"Error preparing visualization data: {str(e)}")
            return {
                'message': f'Error preparing visualization data: {str(e)}'
            } 