import redis
from typing import Any, Optional, Union, Dict
import json
import pickle
import hashlib
import logging
from datetime import timedelta
import os
from functools import wraps

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis cache service"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_timeout=5
            )
            self.binary_redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,
                socket_timeout=5
            )
            logger.info("Cache service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing cache service: {str(e)}")
            raise

    def generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key based on parameters"""
        param_str = json.dumps(params, sort_keys=True)
        key_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def get_binary(self, key: str) -> Optional[Any]:
        """Get binary value from cache"""
        try:
            value = self.binary_redis.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error retrieving binary from cache: {str(e)}")
            return None

    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """Set value in cache with optional expiration"""
        try:
            serialized_value = json.dumps(value)
            if expiration:
                return self.redis_client.setex(key, expiration, serialized_value)
            return self.redis_client.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False

    def set_binary(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """Set binary value in cache with optional expiration"""
        try:
            serialized_value = pickle.dumps(value)
            if expiration:
                return self.binary_redis.setex(key, expiration, serialized_value)
            return self.binary_redis.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Error setting binary cache: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False

    def clear_prefix(self, prefix: str) -> bool:
        """Clear all keys with given prefix"""
        try:
            keys = self.redis_client.keys(f"{prefix}:*")
            if keys:
                return bool(self.redis_client.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Error clearing prefix from cache: {str(e)}")
            return False

    def cache_dataframe(self, key: str, df: Any, expiration: Optional[int] = None) -> bool:
        """Cache a dataframe efficiently"""
        try:
            return self.set_binary(f"df:{key}", df, expiration)
        except Exception as e:
            logger.error(f"Error caching dataframe: {str(e)}")
            return False

    def get_dataframe(self, key: str) -> Optional[Any]:
        """Retrieve a cached dataframe"""
        try:
            return self.get_binary(f"df:{key}")
        except Exception as e:
            logger.error(f"Error retrieving dataframe: {str(e)}")
            return None

    def cache_visualization(self, key: str, plot_data: Dict[str, Any], expiration: Optional[int] = None) -> bool:
        """Cache visualization data"""
        try:
            return self.set(f"viz:{key}", plot_data, expiration)
        except Exception as e:
            logger.error(f"Error caching visualization: {str(e)}")
            return False

    def get_visualization(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached visualization data"""
        try:
            return self.get(f"viz:{key}")
        except Exception as e:
            logger.error(f"Error retrieving visualization: {str(e)}")
            return None

def cache_decorator(prefix: str, expiration: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_params = {
                'args': args,
                'kwargs': kwargs,
                'func_name': func.__name__
            }
            cache_key = self.cache_service.generate_cache_key(prefix, cache_params)

            # Try to get from cache
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            self.cache_service.set(cache_key, result, expiration)
            logger.info(f"Cached result for {func.__name__}")
            return result

        return wrapper
    return decorator

# Initialize global cache service
cache_service = CacheService(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0))
) 