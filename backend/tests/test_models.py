import os
import sys
import json
import pytest
from dotenv import load_dotenv

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.model_service import ModelService
from config import Config

# Load environment variables
load_dotenv()

@pytest.fixture
def model_service():
    return ModelService()

def test_ollama_connection():
    """Test if Ollama server is running and accessible"""
    import requests
    try:
        response = requests.get(f"{Config.OLLAMA_HOST}/api/tags")
        assert response.status_code == 200
        models = response.json().get('models', [])
        print("\n✅ Ollama server is running and accessible")
        print("\nAvailable models:")
        for model in models:
            print(f"- {model['name']} ({model['details'].get('parameter_size', 'N/A')})")
    except Exception as e:
        print(f"\n❌ Ollama server test failed: {str(e)}")
        raise

def test_ollama_models(model_service):
    """Test Ollama models for different tasks"""
    tasks = {
        'chat': Config.MODELS['ollama']['chat'],
        'data_analysis': Config.MODELS['ollama']['data_analysis'],
        'code': Config.MODELS['ollama']['code']
    }
    
    print("\nTesting Ollama models:")
    for task, model in tasks.items():
        try:
            prompt = f"This is a test prompt for {task}. Please provide a short response."
            response = model_service._get_ollama_response(prompt, model)
            print(f"\n✅ {model} ({task}) test passed")
            print(f"Sample response: {response[:200]}...")
        except Exception as e:
            print(f"\n❌ {model} ({task}) test failed: {str(e)}")

def test_model_selection():
    """Test model selection logic"""
    tasks = ['chat', 'data_analysis', 'code']
    
    print("\nTesting model selection:")
    for task in tasks:
        try:
            provider, model = Config.get_model_for_task(task)
            print(f"✅ {task}: {provider} -> {model}")
        except Exception as e:
            print(f"❌ {task} failed: {str(e)}")

if __name__ == '__main__':
    print("\n=== Testing Ollama Integration ===")
    
    print("\n1. Testing Ollama Server Connection:")
    test_ollama_connection()
    
    print("\n2. Testing Model Service:")
    service = ModelService()
    test_ollama_models(service)
    
    print("\n3. Testing Model Selection:")
    test_model_selection() 