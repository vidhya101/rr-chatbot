from functools import lru_cache
from typing import Dict
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv
from .services.model_manager import ModelManager, ModelConfig, ModelProvider

# Load environment variables
load_dotenv()

@lru_cache()
def get_model_config() -> Dict:
    """
    Load model configuration from YAML file
    """
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

@lru_cache()
def get_model_manager() -> ModelManager:
    """
    Create and configure ModelManager instance
    """
    manager = ModelManager()
    config = get_model_config()

    # Update model configurations with environment variables
    if "CLAUDE_API_KEY" in os.environ:
        manager.models["claude-3-sonnet"].api_key = os.environ["CLAUDE_API_KEY"]
    
    if "MISTRAL_API_KEY" in os.environ:
        manager.models["mistral-medium"].api_key = os.environ["MISTRAL_API_KEY"]
    
    if "HUGGINGFACE_API_KEY" in os.environ:
        manager.models["huggingface-mixtral"].api_key = os.environ["HUGGINGFACE_API_KEY"]

    # Set default model if specified
    default_model = os.environ.get("DEFAULT_MODEL")
    if default_model and default_model in manager.models:
        manager.current_model = default_model

    return manager 