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