import os
import json
import requests
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

# Thread pool for model operations
model_executor = ThreadPoolExecutor(max_workers=2)

# Cache for available models
available_models_cache = {
    "timestamp": 0,
    "models": []
}

# Cache expiry time (60 seconds)
CACHE_EXPIRY = 60

def get_ollama_models() -> List[Dict[str, Any]]:
    """Get list of available models from OLLAMA"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = []
            
            for model in data.get("models", []):
                models.append({
                    "id": model.get("name"),
                    "name": model.get("name"),
                    "provider": "ollama",
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", ""),
                    "parameters": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                })
            
            return models
        else:
            logger.error(f"Failed to get OLLAMA models: {response.status_code} - {response.text}")
            return []
    
    except requests.exceptions.ConnectionError:
        logger.error("OLLAMA server not running or not reachable")
        return []
    except Exception as e:
        logger.error(f"Error getting OLLAMA models: {str(e)}")
        return []

def get_mistral_models() -> List[Dict[str, Any]]:
    """Get list of available models from Mistral AI"""
    if not MISTRAL_API_KEY:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.mistral.ai/v1/models",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            
            for model in data.get("data", []):
                models.append({
                    "id": model.get("id"),
                    "name": model.get("id"),
                    "provider": "mistral",
                    "description": model.get("description", ""),
                    "parameters": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                })
            
            return models
        else:
            logger.error(f"Failed to get Mistral models: {response.status_code} - {response.text}")
            return []
    
    except Exception as e:
        logger.error(f"Error getting Mistral models: {str(e)}")
        return []

def get_huggingface_models() -> List[Dict[str, Any]]:
    """Get list of available models from Hugging Face"""
    if not HUGGINGFACE_API_KEY:
        return []
    
    # Predefined list of popular models
    popular_models = [
        {
            "id": "gpt2",
            "name": "GPT-2",
            "provider": "huggingface",
            "description": "OpenAI's GPT-2 model",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024
            }
        },
        {
            "id": "facebook/bart-large-cnn",
            "name": "BART Large CNN",
            "provider": "huggingface",
            "description": "BART model fine-tuned on CNN Daily Mail",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024
            }
        },
        {
            "id": "google/flan-t5-xxl",
            "name": "Flan-T5 XXL",
            "provider": "huggingface",
            "description": "Google's Flan-T5 XXL model",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024
            }
        }
    ]
    
    return popular_models

def get_available_models(force_refresh=False) -> List[Dict[str, Any]]:
    """Get list of all available models from all providers"""
    global available_models_cache
    
    current_time = time.time()
    
    # Return cached models if available and not expired
    if not force_refresh and available_models_cache["timestamp"] > 0 and \
       current_time - available_models_cache["timestamp"] < CACHE_EXPIRY:
        return available_models_cache["models"]
    
    # Get models from all providers
    ollama_models = get_ollama_models()
    mistral_models = get_mistral_models()
    huggingface_models = get_huggingface_models()
    
    # Combine all models
    all_models = ollama_models + mistral_models + huggingface_models
    
    # Add default models if no models are available
    if not all_models:
        all_models = [
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "default",
                "description": "Default model for chat",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
        ]
    
    # Update cache
    available_models_cache["timestamp"] = current_time
    available_models_cache["models"] = all_models
    
    return all_models

def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model details by ID"""
    models = get_available_models()
    
    for model in models:
        if model["id"] == model_id:
            return model
    
    return None

def check_ollama_status() -> Tuple[bool, str]:
    """Check if OLLAMA server is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        
        if response.status_code == 200:
            return True, "OLLAMA server is running"
        else:
            return False, f"OLLAMA server returned status code {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return False, "OLLAMA server not running or not reachable"
    except Exception as e:
        return False, f"Error checking OLLAMA status: {str(e)}"

def pull_ollama_model(model_name: str) -> Tuple[bool, str]:
    """Pull a model from OLLAMA"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            timeout=30
        )
        
        if response.status_code == 200:
            return True, f"Successfully pulled model {model_name}"
        else:
            return False, f"Failed to pull model {model_name}: {response.status_code} - {response.text}"
    
    except Exception as e:
        return False, f"Error pulling model {model_name}: {str(e)}"

def start_model_refresh_thread():
    """Start a thread to periodically refresh the model list"""
    def model_refresher():
        while True:
            try:
                # Sleep for 5 minutes
                time.sleep(300)
                
                # Refresh models
                get_available_models(force_refresh=True)
                logger.info("Refreshed model list")
            except Exception as e:
                logger.error(f"Error in model refresher thread: {str(e)}")
    
    # Start thread
    model_thread = threading.Thread(target=model_refresher, daemon=True)
    model_thread.start()
    
    logger.info("Model refresh thread started")

def update_model_parameters(model_id: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """Update model parameters"""
    model = get_model_by_id(model_id)
    
    if not model:
        return False, f"Model {model_id} not found"
    
    # Update parameters
    model["parameters"].update(parameters)
    
    # Force refresh of model list
    get_available_models(force_refresh=True)
    
    return True, f"Successfully updated parameters for model {model_id}"

# Initialize model refresh thread
start_model_refresh_thread() 