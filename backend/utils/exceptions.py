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
    """Raised when there's a configuration issue."""
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message=message, status_code=500)

class ResourceCleanupError(BaseError):
    """Raised when resource cleanup fails."""
    def __init__(self, message: str = "Resource cleanup failed"):
        super().__init__(message=message, status_code=500)

class TokenError(BaseError):
    """Raised when there's an issue with tokens."""
    def __init__(self, message: str = "Token error"):
        super().__init__(message=message, status_code=401)

class FileOperationError(BaseError):
    """Raised when file operations fail."""
    def __init__(self, message: str = "File operation failed"):
        super().__init__(message=message, status_code=500)

class ModelError(BaseError):
    """Raised when there's an issue with AI models."""
    def __init__(self, message: str = "Model error"):
        super().__init__(message=message, status_code=500)

class CacheError(BaseError):
    """Raised when cache operations fail."""
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message=message, status_code=500)

class DataAnalysisError(Exception):
    """Base exception for data analysis errors"""
    pass

class FileProcessingError(DataAnalysisError):
    """Exception for file processing errors"""
    pass

class VisualizationError(DataAnalysisError):
    """Exception for visualization errors"""
    pass

class PermissionError(AuthenticationError):
    """Exception for permission-related errors"""
    pass

class RetryError(BaseError):
    """Raised when retry attempts are exhausted."""
    def __init__(self, message: str = "Maximum retry attempts exceeded"):
        super().__init__(message=message, status_code=500) 