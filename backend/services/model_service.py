import logging
import threading
import requests
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODELS = os.environ.get('OLLAMA_MODELS', '/mnt/f/MyLLMModels')

# Connection pool for Ollama
ollama_session = requests.Session()

def get_available_models(force_refresh=False) -> List[Dict[str, Any]]:
    """Get list of available models from Ollama"""
    try:
        response = ollama_session.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if response.status_code == 200:
            ollama_models = response.json().get('models', [])
            models_list = []
            for model in ollama_models:
                model_name = model.get('name')
                if model_name:
                    models_list.append({
                        "id": model_name,
                        "name": f"{model_name} (Local)",
                        "provider": "ollama",
                        "parameters": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 2048
                        }
                    })
            return models_list
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
    
    # Fallback to default model if Ollama is not available
    return [{
        "id": "mistral",
        "name": "Mistral (Local)",
        "provider": "ollama",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }
    }]

def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model by ID"""
    models = get_available_models()
    for model in models:
        if model["id"] == model_id:
            return model
    return None

def update_model_parameters(model_id: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """Update model parameters"""
    model = get_model_by_id(model_id)
    if not model:
        return False, f"Model {model_id} not found"
    
    model["parameters"].update(parameters)
    return True, "Parameters updated successfully"

def start_model_refresh_thread():
    """Start model refresh thread"""
    def model_refresher():
        while True:
            try:
                logger.info("Model refresh thread started")
                # Refresh available models every 60 seconds
                get_available_models(force_refresh=True)
                threading.Event().wait(60)  # Sleep for 60 seconds
            except Exception as e:
                logger.error(f"Error in model refresh thread: {str(e)}")
                threading.Event().wait(60)  # Sleep and retry

    thread = threading.Thread(target=model_refresher, daemon=True)
    thread.start() 