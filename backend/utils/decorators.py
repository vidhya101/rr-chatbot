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