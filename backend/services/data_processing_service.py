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
from pydantic import BaseModel
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
        self.datasets = {}
        
        # Initialize Ray for distributed computing
        try:
            # Only initialize Ray if not already initialized
            if not ray.is_initialized():
                # Set runtime_env allowing module import across processes
                ray.init(ignore_reinit_error=True, 
                         runtime_env={"py_modules": []},
                         include_dashboard=False)
                logger.info("Ray initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ray: {str(e)}")

        # Initialize Dask client for distributed computing
        try:
            # Only create client if needed
            self.dask_client = Client(processes=False, n_workers=max_workers, threads_per_worker=1)
            logger.info("Dask client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Dask client: {str(e)}")
            self.dask_client = None

    @cache_decorator(prefix="data_load", expiration=3600)
    async def load_data(self, file_path: str, file_type: str = None, **kwargs) -> Union[pd.DataFrame, dd.DataFrame]:
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

    async def _load_large_dataset(self, file_path: str, file_type: str = None, **kwargs) -> dd.DataFrame:
        """Load large datasets using Dask"""
        try:
            if file_type == 'parquet':
                return dd.read_parquet(file_path, **kwargs)
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

    async def process_data(self, df: Union[pd.DataFrame, dd.DataFrame], 
                     operations: List[Dict[str, Any]], chunk_size: Optional[int] = None) -> Any:
        """Process data with various operations, choosing the right backend based on data type"""
        try:
            if isinstance(df, dd.DataFrame):
                return await self._process_dask(df, operations)
            elif isinstance(df, pd.DataFrame):
                if chunk_size:
                    return await self._process_chunked(df, operations, chunk_size)
                else:
                    return await self._process_pandas(df, operations)
            else:
                raise ValueError(f"Unsupported dataframe type: {type(df)}")
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
            
    async def _process_dask(self, df: dd.DataFrame, operations: List[Dict[str, Any]]) -> dd.DataFrame:
        """Process data using Dask dataframe"""
        try:
            for operation in operations:
                op_type = operation['type']
                if op_type == 'filter':
                    df = df.query(operation['condition'])
                elif op_type == 'transform':
                    df = df.map_partitions(lambda x: x.apply(eval(operation['function']), axis=1))
                elif op_type == 'groupby':
                    df = df.groupby(operation['columns']).agg(operation['aggregation'])
                elif op_type == 'sort':
                    df = df.sort_values(operation['columns'], ascending=operation.get('ascending', True))
            return df
        except Exception as e:
            logger.error(f"Error processing with Dask: {str(e)}")
            raise
            
    # Replace vaex processing with dask processing
    async def _process_vaex(self, df: dd.DataFrame, operations: List[Dict[str, Any]]) -> dd.DataFrame:
        """Fallback to process data using Dask dataframe instead of Vaex"""
        logger.warning("Vaex processing requested but not available, using Dask instead")
        return await self._process_dask(df, operations)

    async def _process_chunked(self, df: pd.DataFrame, operations: List[Dict[str, Any]], 
                             chunk_size: int) -> pd.DataFrame:
        """Process data in chunks using ThreadPoolExecutor instead of Ray"""
        try:
            chunks = np.array_split(df, len(df) // chunk_size + 1)
            
            # Use ThreadPoolExecutor instead of Ray for better Windows compatibility
            with ThreadPoolExecutor(max_workers=self.executor._max_workers) as executor:
                # Process chunks in parallel using local processing instead of Ray
                future_results = [executor.submit(self._process_chunk_local, chunk, operations) 
                                 for chunk in chunks]
                results = [future.result() for future in future_results]
            
            return pd.concat(results)
        except Exception as e:
            logger.error(f"Error processing chunked data: {str(e)}")
            raise
            
    def _process_chunk_local(self, chunk: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a chunk of data locally without Ray"""
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
            logger.error(f"Error processing chunk locally: {str(e)}")
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

    async def optimize_storage(self, df: Union[pd.DataFrame, dd.DataFrame], 
                         output_path: str) -> None:
        """Optimize data storage format for better performance"""
        try:
            if isinstance(df, dd.DataFrame):
                # Use Dask's optimized parquet writer
                df.to_parquet(output_path, compression='snappy', engine='pyarrow')
            elif isinstance(df, pd.DataFrame):
                # Use PyArrow's optimized parquet writer
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path, compression='snappy')
            else:
                raise ValueError(f"Unsupported dataframe type for storage optimization: {type(df)}")
                
            logger.info(f"Successfully optimized data storage at {output_path}")
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

    async def process_data_with_algorithms(self, df: Union[pd.DataFrame, dd.DataFrame],
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
        """Safely cleanup resources"""
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
                
            if hasattr(self, 'dask_client') and self.dask_client:
                try:
                    self.dask_client.close()
                except:
                    pass
                    
            # Only shutdown Ray if it was initialized
            if ray.is_initialized():
                try:
                    ray.shutdown()
                except:
                    pass
                    
            logger.info("Resources cleaned up successfully")
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

# Initialize the data processing service
# We only initialize this if the module is imported directly (not through multiprocessing)
# This prevents issues with multiprocessing on Windows
if __name__ != "__main__":
    try:
        data_processing_service = DataProcessingService()
        logger.info("Data processing service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing data processing service: {str(e)}")
        # Create a minimal instance that will be replaced when actually used
        data_processing_service = DataProcessingService() 
else:
    # This will only be used in direct script execution, which should not happen
    data_processing_service = None 