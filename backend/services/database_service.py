import os
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from utils.retry import retry, RetryContext
from utils.circuit_breaker import CircuitBreaker
from utils.exceptions import (
    DatabaseError, ConnectionError, QueryError, TransactionError,
    ValidationError, ResourceCleanupError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for handling database operations with connection pooling and error handling"""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database service with connection pooling"""
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise DatabaseError("Database URL not provided")
        
        # Configure connection pool
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=20,  # Maximum number of connections
            max_overflow=10,  # Maximum number of connections that can be created beyond pool_size
            pool_timeout=30,  # Timeout for getting a connection from the pool
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True  # Enable connection health checks
        )
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(
            bind=self.engine,
            expire_on_commit=False
        ))
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0
        )
        
        # Initialize metrics
        self.metrics = {
            'connections_active': 0,
            'connections_max': 0,
            'queries_total': 0,
            'queries_failed': 0,
            'transactions_total': 0,
            'transactions_failed': 0,
            'last_error_time': None,
            'last_error_message': None
        }
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    @circuit_breaker
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query with retries and circuit breaker"""
        try:
            self.metrics['queries_total'] += 1
            
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row) for row in result]
                return []
                
        except OperationalError as e:
            self._update_error_metrics("Query execution failed (operational)", str(e))
            raise ConnectionError(f"Database connection error: {str(e)}")
        except SQLAlchemyError as e:
            self._update_error_metrics("Query execution failed (SQLAlchemy)", str(e))
            raise QueryError(f"Query execution error: {str(e)}")
        except Exception as e:
            self._update_error_metrics("Query execution failed (unknown)", str(e))
            raise DatabaseError(f"Unexpected database error: {str(e)}")
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    @circuit_breaker
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute multiple operations in a transaction with retries"""
        try:
            self.metrics['transactions_total'] += 1
            
            with self.get_session() as session:
                for operation in operations:
                    query = operation.get('query')
                    params = operation.get('params', {})
                    if not query:
                        raise ValidationError("Missing query in transaction operation")
                    session.execute(text(query), params)
                return True
                
        except IntegrityError as e:
            self._update_error_metrics("Transaction failed (integrity)", str(e))
            raise TransactionError(f"Transaction integrity error: {str(e)}")
        except OperationalError as e:
            self._update_error_metrics("Transaction failed (operational)", str(e))
            raise ConnectionError(f"Database connection error: {str(e)}")
        except SQLAlchemyError as e:
            self._update_error_metrics("Transaction failed (SQLAlchemy)", str(e))
            raise TransactionError(f"Transaction execution error: {str(e)}")
        except Exception as e:
            self._update_error_metrics("Transaction failed (unknown)", str(e))
            raise DatabaseError(f"Unexpected database error: {str(e)}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current database connection statistics"""
        try:
            stats = {
                'pool_size': self.engine.pool.size(),
                'connections_active': self.engine.pool.checkedin(),
                'connections_available': self.engine.pool.checkedout(),
                'connections_max': self.engine.pool.overflow(),
                'queries_total': self.metrics['queries_total'],
                'queries_failed': self.metrics['queries_failed'],
                'transactions_total': self.metrics['transactions_total'],
                'transactions_failed': self.metrics['transactions_failed'],
                'last_error_time': self.metrics['last_error_time'],
                'last_error_message': self.metrics['last_error_message']
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {str(e)}")
            return {}
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        try:
            # Execute simple query to check connectivity
            start_time = datetime.now()
            self.execute_query("SELECT 1")
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Get connection stats
            stats = self.get_connection_stats()
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'connections': stats,
                'last_error': {
                    'time': self.metrics['last_error_time'],
                    'message': self.metrics['last_error_message']
                }
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_error': {
                    'time': self.metrics['last_error_time'],
                    'message': self.metrics['last_error_message']
                }
            }
    
    def optimize_database(self) -> bool:
        """Perform database optimization tasks"""
        try:
            # List of optimization queries
            optimization_queries = [
                "ANALYZE VERBOSE",
                "VACUUM ANALYZE"  # Note: This might lock tables
            ]
            
            for query in optimization_queries:
                try:
                    self.execute_query(query)
                except Exception as e:
                    logger.warning(f"Optimization query failed ({query}): {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database optimization failed: {str(e)}")
            return False
    
    def _update_error_metrics(self, error_type: str, error_message: str) -> None:
        """Update error metrics"""
        try:
            self.metrics['queries_failed'] += 1
            self.metrics['last_error_time'] = datetime.now()
            self.metrics['last_error_message'] = f"{error_type}: {error_message}"
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup_resources()
    
    def _cleanup_resources(self) -> None:
        """Clean up database resources"""
        try:
            # Close all connections in the pool
            self.engine.dispose()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise ResourceCleanupError(f"Failed to clean up database resources: {str(e)}") 