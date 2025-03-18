import os
import json
import redis.asyncio as redis
from dotenv import load_dotenv
import logging
from typing import Optional, Any, Dict, List, Union

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get Redis URL from environment variables
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

async def init_redis_pool() -> redis.Redis:
    """Initialize Redis connection pool"""
    try:
        redis_pool = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        # Test connection
        await redis_pool.ping()
        logger.info("Redis connection established")
        return redis_pool
    except Exception as e:
        logger.error(f"Error connecting to Redis: {str(e)}")
        logger.warning("Running without Redis cache")
        return None

async def get_redis(request) -> redis.Redis:
    """Get Redis connection from request state"""
    return request.app.state.redis_pool

# Cache functions
async def set_cache(
    redis_client: redis.Redis,
    key: str,
    value: Any,
    expire: int = 3600
) -> bool:
    """Set a value in Redis cache with expiration"""
    if not redis_client:
        return False
    
    try:
        # Convert value to JSON string if it's not a string
        if not isinstance(value, str):
            value = json.dumps(value)
        
        await redis_client.set(key, value, ex=expire)
        return True
    except Exception as e:
        logger.error(f"Error setting cache: {str(e)}")
        return False

async def get_cache(
    redis_client: redis.Redis,
    key: str
) -> Optional[Any]:
    """Get a value from Redis cache"""
    if not redis_client:
        return None
    
    try:
        value = await redis_client.get(key)
        if not value:
            return None
        
        # Try to parse as JSON, return as string if not valid JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    except Exception as e:
        logger.error(f"Error getting cache: {str(e)}")
        return None

async def delete_cache(
    redis_client: redis.Redis,
    key: str
) -> bool:
    """Delete a value from Redis cache"""
    if not redis_client:
        return False
    
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Error deleting cache: {str(e)}")
        return False

# Pub/Sub functions
async def publish_message(
    redis_client: redis.Redis,
    channel: str,
    message: Union[str, Dict, List]
) -> bool:
    """Publish a message to a Redis channel"""
    if not redis_client:
        return False
    
    try:
        # Convert message to JSON string if it's not a string
        if not isinstance(message, str):
            message = json.dumps(message)
        
        await redis_client.publish(channel, message)
        return True
    except Exception as e:
        logger.error(f"Error publishing message: {str(e)}")
        return False

async def subscribe_to_channel(
    redis_client: redis.Redis,
    channel: str
):
    """Subscribe to a Redis channel and yield messages"""
    if not redis_client:
        return
    
    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                try:
                    # Try to parse as JSON, return as string if not valid JSON
                    data = json.loads(message["data"])
                except json.JSONDecodeError:
                    data = message["data"]
                
                yield data
    except Exception as e:
        logger.error(f"Error subscribing to channel: {str(e)}")
    finally:
        await pubsub.unsubscribe(channel) 