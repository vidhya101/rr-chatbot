import os
from dotenv import load_dotenv
import requests

load_dotenv()

class Config:
    # API Keys
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    
    # Model Settings
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'ollama')  # Default to Ollama
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODELS = os.getenv('OLLAMA_MODELS', '/mnt/f/MyLLMModels')
    ENABLE_PRIVATE_DATA_ANALYSIS = os.getenv('ENABLE_PRIVATE_DATA_ANALYSIS', 'true').lower() == 'true'
    FALLBACK_TO_LOCAL = os.getenv('FALLBACK_TO_LOCAL', 'true').lower() == 'true'
    
    # Model Configurations
    MODELS = {
        'ollama': {
            'data_analysis': 'mixtral:latest',  # Using Mixtral for data analysis
            'chat': 'mistral:latest',  # Using Mistral for chat
            'code': 'codellama:7b',  # Using CodeLlama for code
            'requires_api': False
        },
        'claude': {
            'model': 'claude-3-opus-20240229',
            'requires_api': True,
            'priority': 1
        },
        'mistral': {
            'model': 'mistral-large-latest',
            'requires_api': True,
            'priority': 2
        },
        'huggingface': {
            'model': 'meta-llama/Llama-2-70b-chat-hf',
            'requires_api': True,
            'priority': 3
        }
    }
    
    @staticmethod
    def check_ollama_available():
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{Config.OLLAMA_HOST}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def get_model_for_task(task_type='chat'):
        """
        Determine which model to use based on task type and availability
        
        Args:
            task_type: str, one of 'chat', 'data_analysis', 'code'
            
        Returns:
            tuple: (provider, model_name)
        """
        # For data analysis, prefer local models if enabled
        if task_type == 'data_analysis' and Config.ENABLE_PRIVATE_DATA_ANALYSIS:
            if Config.check_ollama_available():
                return 'ollama', Config.MODELS['ollama']['data_analysis']
        
        # If default model is auto, try APIs first
        if Config.DEFAULT_MODEL == 'auto':
            # Try APIs in priority order
            for provider in sorted(Config.MODELS.items(), key=lambda x: x[1].get('priority', 999)):
                provider_name = provider[0]
                if provider_name == 'ollama':
                    continue
                    
                if Config.MODELS[provider_name]['requires_api']:
                    api_key = getattr(Config, f"{provider_name.upper()}_API_KEY", None)
                    if api_key:
                        return provider_name, Config.MODELS[provider_name]['model']
        
        # If specific model is requested
        elif Config.DEFAULT_MODEL in Config.MODELS:
            provider = Config.DEFAULT_MODEL
            if not Config.MODELS[provider]['requires_api']:
                return provider, Config.MODELS[provider].get(task_type, 'mistral:latest')
            elif getattr(Config, f"{provider.upper()}_API_KEY", None):
                return provider, Config.MODELS[provider]['model']
        
        # Fallback to Ollama if available
        if Config.FALLBACK_TO_LOCAL and Config.check_ollama_available():
            return 'ollama', Config.MODELS['ollama'].get(task_type, 'mistral:latest')
            
        raise ValueError("No available AI service configured") 