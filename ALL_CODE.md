# Complete Codebase for RR-Chatbot

This file contains all the code from the RR-Chatbot project, organized by file path.

## Backend Utilities

### backend/utils/ai_utils.py
```python
import numpy as np
import time
import threading
import queue
import logging
import json
import os
import hashlib
from collections import OrderedDict, deque
import sqlite3
from datetime import datetime, timedelta
import re
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize thread pool
ai_executor = ThreadPoolExecutor(max_workers=4)

# Response wrapper class to store metadata
class ResponseWrapper:
    """Wrapper class for AI responses to store metadata"""
    def __init__(self, content, original_query=None):
        self.content = content
        self.original_query = original_query
    
    def __str__(self):
        return self.content

# LRU Cache for responses
class LRUCache:
    """LRU Cache implementation for caching AI responses"""
    def __init__(self, capacity=1000):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        
        # If at capacity, remove least recently used
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        # Add new item
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)

# Initialize response cache
response_cache = LRUCache(capacity=1000)

# Semantic similarity calculation for better response matching
def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts using simple TF-IDF"""
    # Tokenize and lowercase
    def tokenize(text):
        # Simple tokenization by splitting on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # Create vocabulary
    vocab = set(tokens1 + tokens2)
    
    # Calculate term frequency
    def term_frequency(tokens):
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        return tf
    
    tf1 = term_frequency(tokens1)
    tf2 = term_frequency(tokens2)
    
    # Calculate dot product
    dot_product = sum(tf1.get(token, 0) * tf2.get(token, 0) for token in vocab)
    
    # Calculate magnitude
    magnitude1 = sum(tf1.get(token, 0) ** 2 for token in vocab) ** 0.5
    magnitude2 = sum(tf2.get(token, 0) ** 2 for token in vocab) ** 0.5
    
    # Avoid division by zero
    if magnitude1 * magnitude2 == 0:
        return 0
    
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)

# Generate cache key for messages
def generate_cache_key(messages, model):
    """Generate a cache key for a set of messages and model"""
    # Extract only the last few messages for the key to avoid cache misses due to long history
    last_messages = messages[-3:] if len(messages) > 3 else messages
    
    # Convert to string and hash
    message_str = json.dumps(last_messages, sort_keys=True)
    hash_obj = hashlib.md5(message_str.encode())
    
    return f"{model}:{hash_obj.hexdigest()}"

# Check cache for similar queries
def check_cache_for_similar(messages, model, threshold=0.85):
    """Check cache for similar queries above a similarity threshold"""
    # Generate key for exact match
    key = generate_cache_key(messages, model)
    
    # Check for exact match
    exact_match = response_cache.get(key)
    if exact_match:
        logger.info(f"Cache hit for {key}")
        return exact_match
    
    # If no exact match, check for similar queries
    if len(messages) == 0:
        return None
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            last_user_message = msg.get('content', '')
            break
    
    if not last_user_message:
        return None
    
    # Check all cache entries for similar queries
    for cache_key, cache_value in response_cache.cache.items():
        # Only check same model
        if not cache_key.startswith(f"{model}:"):
            continue
        
        # Extract the messages from the cache key
        try:
            # Reconstruct the messages from the cache key
            cache_model, cache_hash = cache_key.split(':')
            
            # If we have cached the original messages
            if hasattr(cache_value, 'original_query') and cache_value.original_query:
                cached_last_user_message = None
                for msg in reversed(cache_value.original_query):
                    if msg.get('role') == 'user':
                        cached_last_user_message = msg.get('content', '')
                        break
                
                if cached_last_user_message:
                    # Calculate similarity
                    similarity = calculate_similarity(last_user_message, cached_last_user_message)
                    
                    if similarity >= threshold:
                        logger.info(f"Similar cache hit for {key} (similarity: {similarity:.2f})")
                        return cache_value
        except Exception as e:
            logger.error(f"Error checking cache similarity: {str(e)}")
    
    return None

# Optimized response generation with fallbacks
def generate_optimized_response(messages, model, timeout=30, fallback_model=None):
    """Generate an optimized response with caching and fallbacks"""
    # Check cache first
    cache_result = check_cache_for_similar(messages, model)
    if cache_result:
        return cache_result.content if hasattr(cache_result, 'content') else cache_result
    
    # If not in cache, generate response with timeout
    try:
        # Import here to avoid circular imports
        from services.ai_service import generate_response
        
        # Submit task to thread pool with timeout
        future = ai_executor.submit(generate_response, messages, model)
        response_text = future.result(timeout=timeout)
        
        # Wrap the response in our wrapper class
        response_obj = ResponseWrapper(response_text, original_query=messages)
        
        # Cache the response
        cache_key = generate_cache_key(messages, model)
        response_cache.put(cache_key, response_obj)
        
        return response_text
    
    except TimeoutError:
        logger.error(f"Timeout generating response for model {model}")
        
        # Try fallback model if provided
        if fallback_model and fallback_model != model:
            logger.info(f"Trying fallback model {fallback_model}")
            try:
                # Import here to avoid circular imports
                from services.ai_service import generate_response
                
                # Submit task to thread pool with shorter timeout
                future = ai_executor.submit(generate_response, messages, fallback_model)
                response_text = future.result(timeout=timeout * 0.8)  # Shorter timeout for fallback
                
                return response_text
            except Exception as e:
                logger.error(f"Error with fallback model: {str(e)}")
        
        # If all else fails, return a simple response
        return generate_simple_response(messages)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return generate_simple_response(messages)

# Simple response generation for fallbacks
def generate_simple_response(messages):
    """Generate a simple response when all other methods fail"""
    try:
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                last_user_message = msg.get('content', '')
                break
        
        if not last_user_message:
            return "I'm sorry, I couldn't understand your request. Could you please try again?"
        
        # Simple keyword-based responses
        last_user_message = last_user_message.lower()
        
        if "hello" in last_user_message or "hi" in last_user_message:
            return "Hello! How can I help you today?"
        
        if "how are you" in last_user_message:
            return "I'm functioning well, thank you for asking! How can I assist you?"
        
        return "I apologize, but I'm having trouble processing your request. Could you please try again?"
    
    except Exception as e:
        logger.error(f"Error generating simple response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Could you please try again?"

# Rate limiter for API calls
class RateLimiter:
    """Rate limiter implementation for API calls"""
    def __init__(self, max_requests=60, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_rate_limited(self, ip_address):
        """Check if an IP address is rate limited"""
        current_time = time.time()
        
        # Clean up old entries
        self.requests = {
            ip: timestamps for ip, timestamps in self.requests.items()
            if timestamps[-1] > current_time - self.time_window
        }
        
        # Get timestamps for this IP
        timestamps = self.requests.get(ip_address, [])
        
        # Remove timestamps older than the time window
        timestamps = [ts for ts in timestamps if ts > current_time - self.time_window]
        
        # Check if rate limited
        if len(timestamps) >= self.max_requests:
            return True
        
        # Add current timestamp
        timestamps.append(current_time)
        self.requests[ip_address] = timestamps
        
        return False

# Initialize rate limiter
rate_limiter = RateLimiter()

# Log API usage
def log_api_usage(user_id, model, tokens, success, error=None, ip_address=None):
    """Log API usage for monitoring and billing"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO api_usage (
            user_id, model, tokens, success, error, ip_address, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, model, tokens, success, error, ip_address, datetime.now()))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error logging API usage: {str(e)}")

# Initialize cache
def init_cache():
    """Initialize the response cache from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT key, value, original_query FROM response_cache')
        rows = cursor.fetchall()
        
        for row in rows:
            key = row['key']
            value = row['value']
            original_query = json.loads(row['original_query']) if row['original_query'] else None
            
            response_obj = ResponseWrapper(value, original_query)
            response_cache.put(key, response_obj)
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error initializing cache: {str(e)}")

# Save cache to database
def save_cache_to_db():
    """Save the response cache to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clear existing cache entries
        cursor.execute('DELETE FROM response_cache')
        
        # Insert current cache entries
        for key, value in response_cache.cache.items():
            original_query = json.dumps(value.original_query) if hasattr(value, 'original_query') else None
            cursor.execute('''
            INSERT INTO response_cache (key, value, original_query)
            VALUES (?, ?, ?)
            ''', (key, str(value), original_query))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error saving cache to database: {str(e)}")

# Start cache saving thread
def start_cache_saving_thread():
    """Start a thread to periodically save the cache to database"""
    def cache_saver():
        while True:
            try:
                save_cache_to_db()
            except Exception as e:
                logger.error(f"Error in cache saver thread: {str(e)}")
            time.sleep(300)  # Save every 5 minutes
    
    thread = threading.Thread(target=cache_saver, daemon=True)
    thread.start()
```

### backend/utils/decorators.py
```python
import functools
import time
from typing import Any, Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker implementation to handle external service failures"""
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        
        return True  # half-open state

    def record_success(self) -> None:
        """Record a successful execution"""
        self.failures = 0
        if self.state == "half-open":
            self.state = "closed"

    def record_failure(self) -> None:
        """Record a failed execution"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"

# Global circuit breaker instance
_circuit_breaker = CircuitBreaker()

def circuit_breaker(func: Callable) -> Callable:
    """Decorator to apply circuit breaker pattern to a function"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker is open for {func.__name__}")
            raise Exception(f"Service is temporarily unavailable")
        
        try:
            result = func(*args, **kwargs)
            _circuit_breaker.record_success()
            return result
        except Exception as e:
            _circuit_breaker.record_failure()
            logger.error(f"Circuit breaker recorded failure for {func.__name__}: {str(e)}")
            raise
    
    return wrapper
```

### backend/utils/validation.py
```python
import os
from typing import Dict, Any, List, Optional
from .exceptions import ValidationError

def validate_file_path(file_path: str) -> None:
    """Validate file path exists and is accessible"""
    if not file_path:
        raise ValidationError("File path is required")
    
    if not os.path.exists(file_path):
        raise ValidationError(f"File not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"File is not readable: {file_path}")

def validate_file_extension(file_path: str, allowed_extensions: List[str]) -> None:
    """Validate file extension is in allowed list"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError(f"Unsupported file extension: {ext}. Allowed extensions: {', '.join(allowed_extensions)}")

def validate_visualization_params(params: Dict[str, Any]) -> None:
    """Validate visualization parameters"""
    required_fields = ['type']
    for field in required_fields:
        if field not in params:
            raise ValidationError(f"Missing required parameter: {field}")
    
    if params['type'] not in ['bar', 'line', 'scatter', 'histogram']:
        raise ValidationError(f"Unsupported visualization type: {params['type']}")

def validate_dashboard_config(config: Dict[str, Any]) -> None:
    """Validate dashboard configuration"""
    required_fields = ['title', 'visualizations']
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field in dashboard config: {field}")
    
    if not isinstance(config['visualizations'], list):
        raise ValidationError("Visualizations must be a list")

def validate_data_format(data: Dict[str, Any]) -> None:
    """Validate data format for analysis"""
    if not data:
        raise ValidationError("Data is required")
    
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary")
    
    required_fields = ['columns', 'rows']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field in data: {field}")

def validate_text_input(
    text: str,
    min_length: int = 1,
    max_length: int = 1000,
    allow_empty: bool = False
) -> str:
    """
    Validate text input.
    
    Args:
        text: The text to validate
        min_length: Minimum length of the text
        max_length: Maximum length of the text
        allow_empty: Whether to allow empty strings
        
    Returns:
        The validated text
        
    Raises:
        ValidationError: If validation fails
    """
    if not text and not allow_empty:
        raise ValidationError("Text cannot be empty")
    
    if text and len(text) < min_length:
        raise ValidationError(f"Text must be at least {min_length} characters long")
    
    if text and len(text) > max_length:
        raise ValidationError(f"Text cannot exceed {max_length} characters")
    
    return text.strip()

def validate_email(email: str) -> str:
    """
    Validate email address.
    
    Args:
        email: The email address to validate
        
    Returns:
        The validated email address
        
    Raises:
        ValidationError: If validation fails
    """
    if not email:
        raise ValidationError("Email cannot be empty")
    
    if '@' not in email or '.' not in email:
        raise ValidationError("Invalid email format")
    
    return email.strip().lower()

def validate_password(password: str, min_length: int = 8) -> str:
    """
    Validate password.
    
    Args:
        password: The password to validate
        min_length: Minimum length of the password
        
    Returns:
        The validated password
        
    Raises:
        ValidationError: If validation fails
    """
    if not password:
        raise ValidationError("Password cannot be empty")
    
    if len(password) < min_length:
        raise ValidationError(f"Password must be at least {min_length} characters long")
    
    return password

def validate_username(username: str, min_length: int = 3, max_length: int = 50) -> str:
    """
    Validate username.
    
    Args:
        username: The username to validate
        min_length: Minimum length of the username
        max_length: Maximum length of the username
        
    Returns:
        The validated username
        
    Raises:
        ValidationError: If validation fails
    """
    if not username:
        raise ValidationError("Username cannot be empty")
    
    if len(username) < min_length:
        raise ValidationError(f"Username must be at least {min_length} characters long")
    
    if len(username) > max_length:
        raise ValidationError(f"Username cannot exceed {max_length} characters")
    
    if not username.isalnum():
        raise ValidationError("Username can only contain letters and numbers")
    
    return username.strip().lower()
```

### backend/utils/exceptions.py
```python
"""Custom exceptions for the application."""

class BaseError(Exception):
    """Base exception class for all custom exceptions."""
    def __init__(self, message: str = None, status_code: int = 500):
        self.message = message or "An unexpected error occurred"
        self.status_code = status_code
        super().__init__(self.message)

class AuthenticationError(BaseError):
    """Raised when authentication fails."""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message=message, status_code=401)

class AuthorizationError(BaseError):
    """Raised when user doesn't have required permissions."""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message=message, status_code=403)

class ValidationError(BaseError):
    """Raised when input validation fails."""
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message=message, status_code=400)

class DatabaseError(BaseError):
    """Raised when database operations fail."""
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message=message, status_code=500)

class AIServiceError(BaseError):
    """Raised when AI service operations fail."""
    def __init__(self, message: str = "AI service operation failed"):
        super().__init__(message=message, status_code=500)

class ExternalServiceError(BaseError):
    """Raised when external service calls fail."""
    def __init__(self, message: str = "External service call failed"):
        super().__init__(message=message, status_code=503)

class ResourceNotFoundError(BaseError):
    """Raised when a requested resource is not found."""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message=message, status_code=404)

class ResourceConflictError(BaseError):
    """Raised when there's a conflict with existing resources."""
    def __init__(self, message: str = "Resource conflict"):
        super().__init__(message=message, status_code=409)

class RateLimitError(BaseError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message=message, status_code=429)

class CircuitBreakerError(BaseError):
    """Raised when circuit breaker is open."""
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message=message, status_code=503)

class ConfigurationError(BaseError):
    """Raised when there's a configuration error."""
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message=message, status_code=500)

class ResourceCleanupError(BaseError):
    """Raised when resource cleanup fails."""
    def __init__(self, message: str = "Resource cleanup failed"):
        super().__init__(message=message, status_code=500)

class TokenError(BaseError):
    """Raised when there's a token-related error."""
    def __init__(self, message: str = "Token error"):
        super().__init__(message=message, status_code=401)

class FileOperationError(BaseError):
    """Raised when file operations fail."""
    def __init__(self, message: str = "File operation failed"):
        super().__init__(message=message, status_code=500)

class ModelError(BaseError):
    """Raised when there's a model-related error."""
    def __init__(self, message: str = "Model error"):
        super().__init__(message=message, status_code=500)

class CacheError(BaseError):
    """Raised when cache operations fail."""
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message=message, status_code=500)

class DataAnalysisError(Exception):
    """Base class for data analysis errors."""
    pass

class FileProcessingError(DataAnalysisError):
    """Raised when file processing fails."""
    pass

class VisualizationError(DataAnalysisError):
    """Raised when visualization generation fails."""
    pass

class PermissionError(AuthenticationError):
    """Raised when user doesn't have required permissions."""
    pass

class RetryError(BaseError):
    """Raised when maximum retry attempts are exceeded."""
    def __init__(self, message: str = "Maximum retry attempts exceeded"):
        super().__init__(message=message, status_code=500)
```

### backend/utils/circuit_breaker.py
```python
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from .exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0
        self.last_success_time = 0
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker moved to HALF-OPEN state")
                else:
                    raise ExternalServiceError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failures = 0
                    self.last_success_time = time.time()
                    logger.info("Circuit breaker moved to CLOSED state")
                
                return result
            
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(f"Circuit breaker moved to OPEN state after {self.failures} failures")
                
                raise ExternalServiceError(f"Service call failed: {str(e)}")
        
        return wrapper

class CircuitBreakerContext:
    """Context manager for circuit breaker pattern"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            half_open_timeout=half_open_timeout
        )
    
    def __enter__(self):
        return self.circuit_breaker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
```

### backend/utils/retry.py
```python
import time
import logging
from functools import wraps
from typing import Type, Union, Tuple, Callable, Any
from .exceptions import RetryError

logger = logging.getLogger(__name__)

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
) -> Callable:
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Multiplier for delay after each attempt
        exceptions: Exception(s) to catch and retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(f"Max retry attempts ({max_attempts}) reached for {func.__name__}")
                        raise RetryError(f"Failed after {max_attempts} attempts: {str(last_exception)}")
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise RetryError(f"Failed after {max_attempts} attempts: {str(last_exception)}")
        
        return wrapper
    return decorator

class RetryContext:
    """Context manager for retrying operations"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
        self.current_delay = delay
        self.attempt = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        if not isinstance(exc_val, self.exceptions):
            return False
        
        self.attempt += 1
        if self.attempt >= self.max_attempts:
            logger.error(f"Max retry attempts ({self.max_attempts}) reached")
            return False
        
        logger.warning(
            f"Attempt {self.attempt}/{self.max_attempts} failed: {str(exc_val)}. "
            f"Retrying in {self.current_delay} seconds..."
        )
        time.sleep(self.current_delay)
        self.current_delay *= self.backoff
        return True
```

### backend/utils/db_utils.py
```python
import sqlite3
import os
import json
import time
import threading
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.environ.get('DB_PATH', 'app.db')

def get_db_connection():
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        profile_picture TEXT,
        bio TEXT,
        role TEXT DEFAULT 'user',
        is_active INTEGER DEFAULT 1,
        last_login TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        theme TEXT DEFAULT 'light',
        language TEXT DEFAULT 'en',
        notifications_enabled INTEGER DEFAULT 1,
        default_model TEXT DEFAULT 'mistral'
    )
    ''')
    
    # Create chats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT,
        user_id TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT 'mistral',
        is_pinned INTEGER DEFAULT 0,
        is_archived INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tokens INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
    )
    ''')
    
    # Create files table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        mime_type TEXT NOT NULL,
        is_processed INTEGER DEFAULT 0,
        processing_status TEXT DEFAULT 'pending',
        processing_error TEXT,
        file_metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id TEXT PRIMARY KEY,
        level TEXT NOT NULL,
        source TEXT NOT NULL,
        message TEXT NOT NULL,
        user_id TEXT,
        details TEXT,
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
    )
    ''')
    
    # Create api_usage table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_usage (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        model TEXT NOT NULL,
        tokens INTEGER NOT NULL,
        success INTEGER NOT NULL,
        error TEXT,
        ip_address TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create response_cache table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS response_cache (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        original_query TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def archive_old_logs(days=30):
    """Archive logs older than specified days"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create archive table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS archived_logs (
            id TEXT PRIMARY KEY,
            level TEXT NOT NULL,
            source TEXT NOT NULL,
            message TEXT NOT NULL,
            user_id TEXT,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at TIMESTAMP,
            archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Move old logs to archive
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute('''
        INSERT INTO archived_logs
        SELECT *, CURRENT_TIMESTAMP
        FROM logs
        WHERE created_at < ?
        ''', (cutoff_date,))
        
        # Delete archived logs
        cursor.execute('DELETE FROM logs WHERE created_at < ?', (cutoff_date,))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error archiving logs: {str(e)}")

def purge_expired_sessions():
    """Purge expired user sessions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete expired sessions
        cursor.execute('''
        DELETE FROM sessions
        WHERE expires_at < CURRENT_TIMESTAMP
        ''')
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error purging sessions: {str(e)}")

def generate_id():
    """Generate a unique ID"""
    return str(uuid.uuid4())

def start_maintenance_task():
    """Start the database maintenance task"""
    def maintenance_worker():
        while True:
            try:
                # Archive old logs
                archive_old_logs()
                
                # Purge expired sessions
                purge_expired_sessions()
                
                # Save cache to database
                save_cache_to_db()
                
            except Exception as e:
                logger.error(f"Error in maintenance task: {str(e)}")
            
            time.sleep(3600)  # Run every hour
    
    thread = threading.Thread(target=maintenance_worker, daemon=True)
    thread.start()

def log_info(source, message, user_id=None, details=None, ip_address=None, user_agent=None):
    """Log an info message"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO logs (
            id, level, source, message, user_id, details, ip_address, user_agent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (generate_id(), 'INFO', source, message, user_id, details, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error logging info message: {str(e)}")

def log_error(source, message, user_id=None, details=None, ip_address=None, user_agent=None):
    """Log an error message"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO logs (
            id, level, source, message, user_id, details, ip_address, user_agent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (generate_id(), 'ERROR', source, message, user_id, details, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error logging error message: {str(e)}")

def log_warning(source, message, user_id=None, details=None, ip_address=None, user_agent=None):
    """Log a warning message"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO logs (
            id, level, source, message, user_id, details, ip_address, user_agent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (generate_id(), 'WARNING', source, message, user_id, details, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error logging warning message: {str(e)}")
``` 