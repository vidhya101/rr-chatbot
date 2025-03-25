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