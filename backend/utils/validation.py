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