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

def train_model(file_id: str, model_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a model on the provided file with the given parameters.
    
    Args:
        file_id: ID of the file to train on
        model_type: Type of model to train (regression, classification, clustering)
        parameters: Model parameters
        
    Returns:
        Dictionary with model training results
    """
    try:
        logger.info(f"Training {model_type} model on file {file_id} with parameters {parameters}")
        
        # In a real application, we would:
        # 1. Load the file using file_id
        # 2. Preprocess the data
        # 3. Train the requested model
        # 4. Return model metrics and visualizations
        
        # For demo purposes, return mock results
        mock_results = {
            "id": f"{model_type}_{file_id[:8]}",
            "type": model_type,
            "file_id": file_id,
            "parameters": parameters,
            "metrics": {
                "accuracy": 0.92 if model_type == "classification" else None,
                "precision": 0.91 if model_type == "classification" else None,
                "recall": 0.89 if model_type == "classification" else None,
                "f1_score": 0.90 if model_type == "classification" else None,
                "r2_score": 0.85 if model_type == "regression" else None,
                "mse": 0.15 if model_type == "regression" else None,
                "silhouette_score": 0.78 if model_type == "clustering" else None
            },
            "plots": [
                {
                    "data": [
                        {
                            "x": [1, 2, 3, 4, 5],
                            "y": [2, 4, 5, 3, 6],
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Actual"
                        },
                        {
                            "x": [1, 2, 3, 4, 5],
                            "y": [2.2, 3.8, 5.2, 3.3, 5.8],
                            "type": "scatter",
                            "mode": "lines",
                            "name": "Predicted"
                        }
                    ],
                    "layout": {
                        "title": "Model Performance",
                        "xaxis": {"title": "X"},
                        "yaxis": {"title": "Y"}
                    }
                }
            ]
        }
        
        # Return mock results
        return mock_results
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise 