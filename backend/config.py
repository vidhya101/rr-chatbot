"""
Configuration management for the RR-Chatbot application.
This module handles all configuration settings, environment variables,
and platform-specific configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from typing import Dict, Any, Optional
import platform
import json

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # Platform-specific settings
    PLATFORM = platform.system().lower()
    IS_WINDOWS = PLATFORM == 'windows'
    IS_LINUX = PLATFORM == 'linux'
    IS_MAC = PLATFORM == 'darwin'
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / 'uploads'
    STATIC_DIR = BASE_DIR / 'static'
    LOG_DIR = BASE_DIR / 'logs'
    MODEL_DIR = BASE_DIR / 'models'
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/app.db')
    
    # Redis settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # API Keys (with validation)
    API_KEYS = {
        'CLAUDE': os.getenv('CLAUDE_API_KEY'),
        'HUGGINGFACE': os.getenv('HUGGINGFACE_API_KEY'),
        'MISTRAL': os.getenv('MISTRAL_API_KEY')
    }
    
    # Model settings with platform-specific paths
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'ollama')
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODELS = Path(os.getenv('OLLAMA_MODELS', str(MODEL_DIR)))
    
    # Feature flags
    FEATURES = {
        'PRIVATE_DATA_ANALYSIS': os.getenv('ENABLE_PRIVATE_DATA_ANALYSIS', 'true').lower() == 'true',
        'FALLBACK_TO_LOCAL': os.getenv('FALLBACK_TO_LOCAL', 'true').lower() == 'true',
        'ENABLE_LOGGING': os.getenv('ENABLE_LOGGING', 'true').lower() == 'true',
        'ENABLE_MONITORING': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
    }
    
    # Model configurations with priority and fallback options
    MODELS: Dict[str, Dict[str, Any]] = {
        'ollama': {
            'data_analysis': 'mixtral:latest',
            'chat': 'mistral:latest',
            'code': 'codellama:7b',
            'requires_api': False,
            'priority': 0,
            'fallback': None
        },
        'claude': {
            'model': 'claude-3-opus-20240229',
            'requires_api': True,
            'priority': 1,
            'fallback': 'ollama'
        },
        'mistral': {
            'model': 'mistral-large-latest',
            'requires_api': True,
            'priority': 2,
            'fallback': 'claude'
        },
        'huggingface': {
            'model': 'meta-llama/Llama-2-70b-chat-hf',
            'requires_api': True,
            'priority': 3,
            'fallback': 'mistral'
        }
    }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories for the application."""
        directories = [
            cls.UPLOAD_DIR,
            cls.STATIC_DIR,
            cls.STATIC_DIR / 'visualizations',
            cls.STATIC_DIR / 'dashboards',
            cls.LOG_DIR,
            cls.MODEL_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate all API keys and return their status."""
        return {
            key: bool(value) for key, value in cls.API_KEYS.items()
        }
    
    @classmethod
    def get_model_for_task(cls, task_type: str = 'chat') -> Optional[Dict[str, Any]]:
        """Get the appropriate model configuration for a specific task."""
        available_models = [
            model for model in cls.MODELS.items()
            if not model[1]['requires_api'] or cls.API_KEYS.get(model[0])
        ]
        
        if not available_models:
            return None
            
        # Sort by priority and find the best available model
        available_models.sort(key=lambda x: x[1]['priority'])
        return available_models[0][1]
    
    @classmethod
    def check_ollama_available(cls) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{cls.OLLAMA_HOST}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    @classmethod
    def get_platform_specific_path(cls, path: str) -> Path:
        """Convert a path to be platform-specific."""
        return Path(path).resolve()
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'platform': cls.PLATFORM,
            'database_url': cls.DATABASE_URL,
            'redis_config': {
                'host': cls.REDIS_HOST,
                'port': cls.REDIS_PORT,
                'db': cls.REDIS_DB
            },
            'features': cls.FEATURES,
            'api_keys_status': cls.validate_api_keys(),
            'ollama_available': cls.check_ollama_available()
        }
    
    @classmethod
    def save_config(cls, path: Optional[Path] = None) -> None:
        """Save current configuration to a JSON file."""
        if path is None:
            path = cls.BASE_DIR / 'config.json'
        
        with open(path, 'w') as f:
            json.dump(cls.to_dict(), f, indent=4) 