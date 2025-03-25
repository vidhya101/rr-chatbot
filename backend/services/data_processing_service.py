from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import shap
import lime
import lime.lime_tabular
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import pandas as pd
import dask.dataframe as dd
import vaex
import polars as pl
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import modin.pandas as mpd
from distributed import Client, LocalCluster
import logging
import ray
from tqdm import tqdm
import gc
import os
import asyncio
from pathlib import Path
from collections import defaultdict, deque, Counter
import heapq
from itertools import combinations
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .cache_service import cache_service, cache_decorator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStructures:
    """Class containing various data structures for efficient data processing"""
    
    @staticmethod
    def create_sparse_matrix(df: pd.DataFrame, columns: List[str]) -> csr_matrix:
        """Convert selected columns to sparse matrix for memory efficiency"""
        return csr_matrix(df[columns].values)
    
    @staticmethod
    def create_graph(df: pd.DataFrame, source_col: str, target_col: str) -> nx.Graph:
        """Create a graph from dataframe columns"""
        return nx.from_pandas_edgelist(df, source_col, target_col)
    
    @staticmethod
    def create_adjacency_list(df: pd.DataFrame, source_col: str, target_col: str) -> Dict[Any, List[Any]]:
        """Create adjacency list representation"""
        adj_list = defaultdict(list)
        for _, row in df.iterrows():
            adj_list[row[source_col]].append(row[target_col])
        return dict(adj_list)
    
    @staticmethod
    def create_priority_queue() -> List:
        """Create an empty priority queue"""
        return []

class Algorithms:
    """Class containing various algorithms for data analysis"""
    
    @staticmethod
    def find_connected_components(sparse_matrix: csr_matrix) -> Tuple[int, np.ndarray]:
        """Find connected components in sparse matrix"""
        return connected_components(sparse_matrix)
    
    @staticmethod
    def topological_sort(graph: nx.Graph) -> List:
        """Perform topological sort on graph"""
        return list(nx.topological_sort(graph))
    
    @staticmethod
    def find_shortest_path(graph: nx.Graph, start: Any, end: Any) -> List:
        """Find shortest path between nodes"""
        return nx.shortest_path(graph, start, end)
    
    @staticmethod
    def find_frequent_patterns(df: pd.DataFrame, min_support: float = 0.1) -> Dict[frozenset, int]:
        """Implementation of Apriori algorithm for frequent pattern mining"""
        def create_candidates(prev_candidates: List[frozenset], k: int) -> List[frozenset]:
            candidates = []
            for i in range(len(prev_candidates)):
                for j in range(i + 1, len(prev_candidates)):
                    union = prev_candidates[i].union(prev_candidates[j])
                    if len(union) == k:
                        candidates.append(union)
            return candidates

        transactions = [frozenset(row) for row in df.values]
        items = frozenset().union(*transactions)
        min_support_count = len(transactions) * min_support
        
        # Initialize with single items
        item_counts = Counter()
        for transaction in transactions:
            item_counts.update(transaction)
        
        frequent_patterns = {frozenset([item]): count 
                           for item, count in item_counts.items() 
                           if count >= min_support_count}
        
        k = 2
        while frequent_patterns:
            candidates = create_candidates(list(frequent_patterns.keys()), k)
            candidate_counts = Counter()
            for transaction in transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1
            
            frequent_patterns = {candidate: count 
                               for candidate, count in candidate_counts.items() 
                               if count >= min_support_count}
            k += 1
        
        return frequent_patterns

class DataProcessingService:
    def __init__(self, max_workers: int = 4):
        """Initialize the data processing service with parallel processing capabilities"""
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache_service = cache_service
        self.data_structures = DataStructures()
        self.algorithms = Algorithms()
        
        # Initialize Ray for distributed computing
        try:
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ray: {str(e)}")

        # Initialize Dask client for distributed computing
        try:
            self.dask_client = Client(processes=True, n_workers=max_workers)
            logger.info("Dask client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Dask client: {str(e)}")

    @cache_decorator(prefix="data_load", expiration=3600)
    async def load_data(self, file_path: str, file_type: str = None, **kwargs) -> Union[pd.DataFrame, dd.DataFrame, vaex.DataFrame]:
        """Load data efficiently based on file type and size"""
        try:
            file_size = Path(file_path).stat().st_size
            
            # Use appropriate loading method based on file size and type
            if file_size > 1e9:  # > 1GB
                return await self._load_large_dataset(file_path, file_type, **kwargs)
            else:
                return await self._load_small_dataset(file_path, file_type, **kwargs)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    async def _load_large_dataset(self, file_path: str, file_type: str = None, **kwargs) -> Union[dd.DataFrame, vaex.DataFrame]:
        """Load large datasets using Dask or Vaex"""
        try:
            if file_type == 'parquet':
                return dd.read_parquet(file_path, **kwargs)
            elif file_type == 'csv':
                return vaex.from_csv(file_path, **kwargs)
            else:
                return dd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading large dataset: {str(e)}")
            raise

    async def _load_small_dataset(self, file_path: str, file_type: str = None, **kwargs) -> pd.DataFrame:
        """Load smaller datasets using Pandas"""
        try:
            if file_type == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif file_type == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                return pd.read_excel(file_path, **kwargs)
            else:
                return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading small dataset: {str(e)}")
            raise

    @ray.remote
    def _process_chunk(self, chunk: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a chunk of data using Ray"""
        try:
            for operation in operations:
                op_type = operation['type']
                if op_type == 'filter':
                    chunk = chunk.query(operation['condition'])
                elif op_type == 'transform':
                    chunk = chunk.eval(operation['expression'])
                elif op_type == 'aggregate':
                    chunk = chunk.groupby(operation['by']).agg(operation['agg'])
            return chunk
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise

    async def process_data(self, df: Union[pd.DataFrame, dd.DataFrame, vaex.DataFrame], 
                         operations: List[Dict[str, Any]], chunk_size: Optional[int] = None) -> Any:
        """Process data with optimized performance"""
        try:
            if isinstance(df, dd.DataFrame):
                return await self._process_dask(df, operations)
            elif isinstance(df, vaex.DataFrame):
                return await self._process_vaex(df, operations)
            elif isinstance(df, pd.DataFrame):
                if chunk_size and len(df) > chunk_size:
                    return await self._process_chunked(df, operations, chunk_size)
                return await self._process_pandas(df, operations)
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    async def _process_dask(self, df: dd.DataFrame, operations: List[Dict[str, Any]]) -> dd.DataFrame:
        """Process data using Dask"""
        try:
            for operation in operations:
                op_type = operation['type']
                if op_type == 'filter':
                    df = df.query(operation['condition'])
                elif op_type == 'transform':
                    df = df.map_partitions(lambda x: x.eval(operation['expression']))
                elif op_type == 'aggregate':
                    df = df.groupby(operation['by']).agg(operation['agg'])
            return df.compute()
        except Exception as e:
            logger.error(f"Error processing with Dask: {str(e)}")
            raise

    async def _process_vaex(self, df: vaex.DataFrame, operations: List[Dict[str, Any]]) -> vaex.DataFrame:
        """Process data using Vaex"""
        try:
            for operation in operations:
                op_type = operation['type']
                if op_type == 'filter':
                    df = df.filter(operation['condition'])
                elif op_type == 'transform':
                    df = df.apply(operation['expression'])
                elif op_type == 'aggregate':
                    df = df.groupby(operation['by']).agg(operation['agg'])
            return df
        except Exception as e:
            logger.error(f"Error processing with Vaex: {str(e)}")
            raise

    async def _process_chunked(self, df: pd.DataFrame, operations: List[Dict[str, Any]], 
                             chunk_size: int) -> pd.DataFrame:
        """Process data in chunks using Ray"""
        try:
            chunks = np.array_split(df, len(df) // chunk_size + 1)
            futures = [self._process_chunk.remote(chunk, operations) for chunk in chunks]
            results = await asyncio.gather(*[asyncio.to_thread(ray.get, future) for future in futures])
            return pd.concat(results)
        except Exception as e:
            logger.error(f"Error processing chunked data: {str(e)}")
            raise

    async def _process_pandas(self, df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process data using Pandas"""
        try:
            for operation in operations:
                op_type = operation['type']
                if op_type == 'filter':
                    df = df.query(operation['condition'])
                elif op_type == 'transform':
                    df = df.eval(operation['expression'])
                elif op_type == 'aggregate':
                    df = df.groupby(operation['by']).agg(operation['agg'])
            return df
        except Exception as e:
            logger.error(f"Error processing with Pandas: {str(e)}")
            raise

    async def optimize_storage(self, df: Union[pd.DataFrame, dd.DataFrame, vaex.DataFrame], 
                             output_path: str) -> None:
        """Optimize data storage using Apache Arrow and Parquet"""
        try:
            if isinstance(df, (dd.DataFrame, vaex.DataFrame)):
                df = df.compute()
            
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path, compression='snappy')
            logger.info(f"Data optimized and saved to {output_path}")
        except Exception as e:
            logger.error(f"Error optimizing storage: {str(e)}")
            raise

    async def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset to determine optimal data structures"""
        try:
            analysis = {}
            
            # Check sparsity
            sparsity = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            analysis['sparsity'] = sparsity
            
            # Check for graph-like relationships
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            potential_relationships = []
            for col1, col2 in combinations(categorical_cols, 2):
                unique_pairs = df.groupby([col1, col2]).size().reset_index()
                if len(unique_pairs) > 0:
                    potential_relationships.append((col1, col2))
            analysis['potential_relationships'] = potential_relationships
            
            # Check for time series
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            analysis['time_series_columns'] = list(datetime_cols)
            
            # Recommend data structures
            recommendations = []
            if sparsity > 0.5:
                recommendations.append('sparse_matrix')
            if potential_relationships:
                recommendations.append('graph')
            if datetime_cols.any():
                recommendations.append('sorted_array')
            
            analysis['recommended_structures'] = recommendations
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing data structure: {str(e)}")
            raise

    @ray.remote
    def _process_chunk_with_algorithms(self, chunk: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a chunk of data using various algorithms"""
        try:
            for operation in operations:
                op_type = operation['type']
                
                if op_type == 'filter':
                    chunk = chunk.query(operation['condition'])
                
                elif op_type == 'transform':
                    if operation.get('use_sparse', False):
                        sparse_data = csr_matrix(chunk.values)
                        # Perform operations on sparse matrix
                        chunk = pd.DataFrame(sparse_data.toarray(), columns=chunk.columns)
                    else:
                        chunk = chunk.eval(operation['expression'])
                
                elif op_type == 'aggregate':
                    if operation.get('method') == 'graph':
                        # Use graph-based aggregation
                        graph = self.data_structures.create_graph(
                            chunk, 
                            operation['source_col'], 
                            operation['target_col']
                        )
                        # Perform graph operations
                        result = self.algorithms.find_connected_components(
                            self.data_structures.create_sparse_matrix(chunk, [operation['source_col'], operation['target_col']])
                        )
                        chunk['component'] = result[1]
                    else:
                        chunk = chunk.groupby(operation['by']).agg(operation['agg'])
                
                elif op_type == 'pattern_mining':
                    patterns = self.algorithms.find_frequent_patterns(
                        chunk[operation['columns']], 
                        min_support=operation.get('min_support', 0.1)
                    )
                    # Convert patterns to DataFrame format
                    pattern_df = pd.DataFrame([
                        {'pattern': ','.join(p), 'support': count} 
                        for p, count in patterns.items()
                    ])
                    chunk = pd.concat([chunk, pattern_df], axis=1)
                
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk with algorithms: {str(e)}")
            raise

    async def process_data_with_algorithms(self, df: Union[pd.DataFrame, dd.DataFrame, vaex.DataFrame],
                                        operations: List[Dict[str, Any]], chunk_size: Optional[int] = None) -> Any:
        """Process data using various algorithms and data structures"""
        try:
            # Analyze data structure
            structure_analysis = await self.analyze_data_structure(df)
            
            # Use appropriate processing method based on analysis
            if 'sparse_matrix' in structure_analysis['recommended_structures']:
                return await self._process_sparse_data(df, operations)
            elif 'graph' in structure_analysis['recommended_structures']:
                return await self._process_graph_data(df, operations)
            else:
                return await self._process_regular_data(df, operations, chunk_size)
                
        except Exception as e:
            logger.error(f"Error processing data with algorithms: {str(e)}")
            raise

    async def _process_sparse_data(self, df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process data using sparse matrix operations"""
        try:
            # Convert to sparse matrix
            sparse_data = self.data_structures.create_sparse_matrix(df, df.columns)
            
            # Process operations
            for operation in operations:
                if operation['type'] == 'filter':
                    # Apply filter on sparse matrix
                    mask = eval(operation['condition'], {'X': sparse_data})
                    sparse_data = sparse_data[mask]
                elif operation['type'] == 'transform':
                    # Apply transformation on sparse matrix
                    sparse_data = eval(operation['expression'], {'X': sparse_data})
            
            # Convert back to DataFrame
            return pd.DataFrame(sparse_data.toarray(), columns=df.columns)
            
        except Exception as e:
            logger.error(f"Error processing sparse data: {str(e)}")
            raise

    async def _process_graph_data(self, df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process data using graph algorithms"""
        try:
            # Create graph from data
            graph = self.data_structures.create_graph(
                df, 
                operations[0].get('source_col'), 
                operations[0].get('target_col')
            )
            
            results = []
            for operation in operations:
                if operation['type'] == 'community_detection':
                    communities = nx.community.louvain_communities(graph)
                    df['community'] = [next(i for i, com in enumerate(communities) if node in com) 
                                     for node in df[operations[0].get('source_col')]]
                elif operation['type'] == 'centrality':
                    centrality = nx.centrality.degree_centrality(graph)
                    df['centrality'] = df[operations[0].get('source_col')].map(centrality)
                elif operation['type'] == 'shortest_path':
                    paths = dict(nx.all_pairs_shortest_path_length(graph))
                    df['avg_path_length'] = df[operations[0].get('source_col')].apply(
                        lambda x: np.mean([v for v in paths.get(x, {}).values()])
                    )
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing graph data: {str(e)}")
            raise

    async def _process_regular_data(self, df: pd.DataFrame, operations: List[Dict[str, Any]], 
                                  chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Process regular data with chunking"""
        try:
            if chunk_size and len(df) > chunk_size:
                chunks = np.array_split(df, len(df) // chunk_size + 1)
                futures = [self._process_chunk_with_algorithms.remote(chunk, operations) for chunk in chunks]
                results = await asyncio.gather(*[asyncio.to_thread(ray.get, future) for future in futures])
                return pd.concat(results)
            else:
                return await self._process_chunk_with_algorithms.remote(df, operations)
                
        except Exception as e:
            logger.error(f"Error processing regular data: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
            self.dask_client.close()
            ray.shutdown()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Additional models for advanced analytics
class AdvancedAnalyticsConfig(BaseModel):
    operation: str
    parameters: Dict[str, Any] = {}

class TimeSeriesConfig(BaseModel):
    time_column: str
    value_column: str
    period: Optional[int] = None

class ClusteringConfig(BaseModel):
    method: str = "kmeans"
    n_clusters: Optional[int] = None
    parameters: Dict[str, Any] = {}

class FeatureSelectionConfig(BaseModel):
    method: str = "mutual_info"
    target_column: str
    n_features: int = 10
    parameters: Dict[str, Any] = {}

class ModelInterpretabilityConfig(BaseModel):
    model_type: str
    target_column: str
    sample_size: int = 100
    parameters: Dict[str, Any] = {}

@app.post("/datasets/{dataset_id}/advanced_analytics", response_model=Dict[str, Any])
async def perform_advanced_analytics(dataset_id: str, config: AdvancedAnalyticsConfig):
    """
    Perform advanced analytics operations on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        results = {}
        
        if config.operation == "advanced_statistics":
            # Perform advanced statistical analysis
            stats_results = {}
            
            # Normality tests
            numeric_cols = df.select_dtypes(include=['number']).columns
            normality_tests = {}
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 3:
                    shapiro_test = stats.shapiro(data)
                    normality_tests[col] = {
                        'shapiro_statistic': float(shapiro_test[0]),
                        'shapiro_p_value': float(shapiro_test[1]),
                        'is_normal': shapiro_test[1] > 0.05
                    }
            stats_results['normality_tests'] = normality_tests
            
            # Correlation analysis with p-values
            if len(numeric_cols) > 1:
                correlation_analysis = {}
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        pearson_corr = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                        spearman_corr = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                        correlation_analysis[f"{col1}_vs_{col2}"] = {
                            'pearson': {
                                'correlation': float(pearson_corr[0]),
                                'p_value': float(pearson_corr[1])
                            },
                            'spearman': {
                                'correlation': float(spearman_corr[0]),
                                'p_value': float(spearman_corr[1])
                            }
                        }
                stats_results['correlation_analysis'] = correlation_analysis
            
            results['advanced_statistics'] = stats_results
            
        elif config.operation == "anomaly_detection":
            # Perform anomaly detection
            method = config.parameters.get('method', 'isolation_forest')
            contamination = config.parameters.get('contamination', 0.1)
            
            numeric_data = df.select_dtypes(include=['number'])
            
            if method == 'isolation_forest':
                detector = IsolationForest(contamination=contamination, random_state=42)
            else:
                raise HTTPException(status_code=400, detail="Unsupported anomaly detection method")
            
            # Fit detector and predict
            labels = detector.fit_predict(numeric_data)
            scores = detector.score_samples(numeric_data)
            
            # Calculate anomaly statistics
            anomaly_mask = labels == -1
            results['anomaly_detection'] = {
                'total_anomalies': int(sum(anomaly_mask)),
                'anomaly_percentage': float(sum(anomaly_mask) / len(labels) * 100),
                'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
                'anomaly_scores': scores.tolist()
            }
        
        return {
            "message": "Advanced analytics completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in advanced analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in advanced analytics: {str(e)}")

@app.post("/datasets/{dataset_id}/time_series_analysis", response_model=Dict[str, Any])
async def analyze_time_series(dataset_id: str, config: TimeSeriesConfig):
    """
    Perform time series analysis on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Ensure time column is datetime
        df[config.time_column] = pd.to_datetime(df[config.time_column])
        ts_df = df.sort_values(config.time_column)
        
        # Perform seasonal decomposition
        ts_data = ts_df.set_index(config.time_column)[config.value_column]
        period = config.period or min(len(ts_data), 12)
        decomposition = seasonal_decompose(ts_data, period=period)
        
        # Perform stationarity test
        adf_test = adfuller(ts_data.dropna())
        
        results = {
            'metrics': {
                'mean': float(ts_data.mean()),
                'std': float(ts_data.std()),
                'trend_direction': 'increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'decreasing',
                'seasonality_strength': float(decomposition.seasonal.std() / ts_data.std()),
                'stationarity': {
                    'adf_statistic': float(adf_test[0]),
                    'p_value': float(adf_test[1]),
                    'is_stationary': adf_test[1] < 0.05
                }
            },
            'components': {
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'residual': decomposition.resid.tolist()
            }
        }
        
        return {
            "message": "Time series analysis completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in time series analysis: {str(e)}")

@app.post("/datasets/{dataset_id}/clustering", response_model=Dict[str, Any])
async def perform_clustering(dataset_id: str, config: ClusteringConfig):
    """
    Perform clustering analysis on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Prepare numeric data
        numeric_data = df.select_dtypes(include=['number'])
        
        # Scale the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Determine optimal number of clusters if not specified
        if config.method == 'kmeans' and config.n_clusters is None:
            max_clusters = min(10, len(numeric_data) // 2)
            silhouette_scores = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_data)
                silhouette_scores.append(silhouette_score(scaled_data, labels))
            n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            n_clusters = config.n_clusters or 2
        
        # Perform clustering
        if config.method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif config.method == 'dbscan':
            clusterer = DBSCAN(eps=config.parameters.get('eps', 0.5),
                             min_samples=config.parameters.get('min_samples', 5))
        else:
            raise HTTPException(status_code=400, detail="Unsupported clustering method")
        
        # Fit clustering model
        labels = clusterer.fit_predict(scaled_data)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        results = {
            'method': config.method,
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'pca_coordinates': pca_result.tolist(),
            'feature_importance': {
                'pc1': dict(zip(numeric_data.columns, pca.components_[0])),
                'pc2': dict(zip(numeric_data.columns, pca.components_[1]))
            }
        }
        
        return {
            "message": "Clustering analysis completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in clustering analysis: {str(e)}")

@app.post("/datasets/{dataset_id}/feature_selection", response_model=Dict[str, Any])
async def select_features(dataset_id: str, config: FeatureSelectionConfig):
    """
    Perform feature selection on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Prepare data
        X = df.drop(columns=[config.target_column])
        y = df[config.target_column]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols)
        
        if config.method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=config.n_features)
            selector.fit(X_encoded, y)
            selected_features = pd.DataFrame({
                'feature': X_encoded.columns,
                'importance': selector.scores_
            }).sort_values('importance', ascending=False)
            
        elif config.method == 'rfe':
            base_model = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator=base_model, n_features_to_select=config.n_features)
            selector.fit(X_encoded, y)
            selected_features = pd.DataFrame({
                'feature': X_encoded.columns,
                'selected': selector.support_,
                'rank': selector.ranking_
            }).sort_values('rank')
        
        results = {
            'method': config.method,
            'selected_features': selected_features.to_dict('records'),
            'n_features_selected': config.n_features
        }
        
        return {
            "message": "Feature selection completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in feature selection: {str(e)}")

@app.post("/datasets/{dataset_id}/model_interpretability", response_model=Dict[str, Any])
async def interpret_model(dataset_id: str, config: ModelInterpretabilityConfig):
    """
    Generate model interpretability insights using SHAP and LIME
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Prepare data
        X = df.drop(columns=[config.target_column])
        y = df[config.target_column]
        
        # Sample data for interpretation
        if len(X) > config.sample_size:
            X_sample = X.sample(n=config.sample_size, random_state=42)
        else:
            X_sample = X
        
        # Train a simple model for interpretation
        if config.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        model.fit(X, y)
        
        # SHAP Analysis
        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        # LIME Analysis
        categorical_features = [i for i, col in enumerate(X.columns) 
                             if col in df.select_dtypes(include=['object', 'category']).columns]
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample.values,
            feature_names=X.columns,
            class_names=[config.target_column],
            categorical_features=categorical_features,
            mode='regression'
        )
        
        results = {
            'global_importance': dict(zip(X.columns, np.abs(shap_values).mean(0))),
            'sample_explanations': []
        }
        
        # Generate explanations for a few samples
        for i in range(min(5, len(X_sample))):
            lime_exp = lime_explainer.explain_instance(
                X_sample.iloc[i].values,
                model.predict,
                num_features=10
            )
            
            results['sample_explanations'].append({
                'instance_id': i,
                'feature_values': X_sample.iloc[i].to_dict(),
                'prediction': float(model.predict(X_sample.iloc[i:i+1])[0]),
                'lime_explanation': dict(lime_exp.as_list()),
                'shap_values': dict(zip(X.columns, shap_values[i]))
            })
        
        return {
            "message": "Model interpretation completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in model interpretation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in model interpretation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("data_processing_service:app", host="0.0.0.0", port=8001, reload=True) 