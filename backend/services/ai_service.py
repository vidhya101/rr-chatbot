import os
import json
import requests
import logging
from dotenv import load_dotenv
import tiktoken
from flask import current_app
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
from mistralai.client import MistralClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
from config import Config

# For Mistral AI
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logging.warning("Mistral AI client not available. Install with: pip install mistralai")

# For Hugging Face
try:
    import huggingface_hub
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Hugging Face Hub not available. Install with: pip install huggingface_hub")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys and configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODELS = os.environ.get('OLLAMA_MODELS', '/mnt/f/MyLLMModels')
HF_TOKEN = os.environ.get('HF_TOKEN')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

# Connection pool for Ollama
ollama_session = requests.Session()
ollama_executor = ThreadPoolExecutor(max_workers=4)
ollama_health_status = {"is_healthy": False, "last_checked": 0}

# Available models
AVAILABLE_MODELS = {
    # Ollama models
    'mistral': {
        'name': 'Mistral (Local)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': True,  # Set as default
        'provider': 'ollama'
    },
    'llama2': {
        'name': 'Llama 2 (Local)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'llama3': {
        'name': 'Llama 3 (Local)',
        'max_tokens': 8192,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'mixtral': {
        'name': 'Mixtral (Local)',
        'max_tokens': 8192,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'phi3:3.8b': {
        'name': 'Phi-3 3.8B (Local)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'phi4': {
        'name': 'Phi-4 (Local)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'deepseek-r1:14b': {
        'name': 'DeepSeek R1 14B (Local)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'deepseek-r1:32b': {
        'name': 'DeepSeek R1 32B (Local)',
        'max_tokens': 8192,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'deepseek-r1:70b': {
        'name': 'DeepSeek R1 70B (Local)',
        'max_tokens': 8192,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'command-r-plus': {
        'name': 'Command R Plus (Local)',
        'max_tokens': 8192,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    'falcon': {
        'name': 'Falcon (Local)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'ollama'
    },
    
    # Mistral API models
    'mistral-small': {
        'name': 'Mistral Small (API)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'mistral'
    },
    'mistral-medium': {
        'name': 'Mistral Medium (API)',
        'max_tokens': 8192,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'mistral'
    },
    
    # Hugging Face models
    'meta-llama/Llama-2-7b-chat-hf': {
        'name': 'Llama 2 7B (HF)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'huggingface'
    },
    'mistralai/Mistral-7B-Instruct-v0.2': {
        'name': 'Mistral 7B (HF)',
        'max_tokens': 4096,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'huggingface'
    },
    'gpt2': {
        'name': 'GPT-2 (Hugging Face)',
        'max_tokens': 1024,
        'temperature': 0.7,
        'is_default': False,
        'provider': 'huggingface'
    }
}

# Check Ollama health periodically
def check_ollama_health():
    """Check if Ollama is healthy and update status"""
    current_time = time.time()
    # Only check every 30 seconds
    if current_time - ollama_health_status["last_checked"] < 30:
        return ollama_health_status["is_healthy"]
    
    try:
        response = ollama_session.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        ollama_health_status["is_healthy"] = response.status_code == 200
    except:
        ollama_health_status["is_healthy"] = False
    
    ollama_health_status["last_checked"] = current_time
    return ollama_health_status["is_healthy"]

def get_available_models():
    """Get list of available models with improved caching and error handling"""
    models_list = []
    
    # First add models we know about
    for model_id, model_info in AVAILABLE_MODELS.items():
        models_list.append({
            'id': model_id,
            'name': model_info['name'],
            'isDefault': model_info['is_default'],
            'provider': model_info['provider']
        })
    
    # Then try to get Ollama models if Ollama is healthy
    if check_ollama_health():
        try:
            response = ollama_session.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            if response.status_code == 200:
                ollama_models = response.json().get('models', [])
                
                # Add any new Ollama models we didn't know about
                for model in ollama_models:
                    model_name = model.get('name')
                    if model_name and model_name not in AVAILABLE_MODELS:
                        models_list.append({
                            'id': model_name,
                            'name': f"{model_name} (Local)",
                            'isDefault': False,
                            'provider': 'ollama'
                        })
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
    
    return models_list

def count_tokens(text):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        # Approximate token count (1 token ≈ 4 chars)
        return len(text) // 4

def generate_response(messages, model="mistral", user=None):
    """Generate a response from the selected AI model with improved reliability"""
    try:
        # Get model info
        model_info = AVAILABLE_MODELS.get(model)
        
        # If model not found, use default
        if not model_info:
            logger.warning(f"Model {model} not found, using default")
            model = next((m for m, info in AVAILABLE_MODELS.items() if info['is_default']), 'mistral')
            model_info = AVAILABLE_MODELS[model]
        
        provider = model_info['provider']
        
        # Generate response based on provider with fallback mechanisms
        if provider == 'ollama':
            # Check if Ollama is healthy
            if check_ollama_health():
                try:
                    # Try Ollama with a shorter timeout
                    return generate_ollama_response(messages, model, model_info)
                except Exception as e:
                    logger.warning(f"Ollama error, falling back: {str(e)}")
            
            # If Ollama is not available or failed, try Mistral API if available
            if MISTRAL_AVAILABLE and MISTRAL_API_KEY:
                logger.info("Falling back to Mistral API")
                try:
                    return generate_mistral_response(messages, "mistral-small", AVAILABLE_MODELS.get("mistral-small", {}))
                except Exception as e:
                    logger.warning(f"Mistral API error: {str(e)}")
            
            # If all else fails, use a simple response
            return generate_simple_response(messages)
            
        elif provider == 'mistral':
            if MISTRAL_AVAILABLE and MISTRAL_API_KEY:
                try:
                    return generate_mistral_response(messages, model, model_info)
                except Exception as e:
                    logger.warning(f"Mistral API error: {str(e)}")
                    return generate_simple_response(messages)
            else:
                logger.warning("Mistral API not available")
                return generate_simple_response(messages)
                
        elif provider == 'huggingface':
            if HF_AVAILABLE and HF_TOKEN:
                try:
                    return generate_huggingface_response(messages, model, model_info)
                except Exception as e:
                    logger.warning(f"Hugging Face error: {str(e)}")
                    return generate_simple_response(messages)
            else:
                logger.warning("Hugging Face not available")
                return generate_simple_response(messages)
        else:
            logger.error(f"Unknown provider: {provider}")
            return generate_simple_response(messages)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."

def generate_ollama_response(messages, model, model_info):
    """Generate response using Ollama API with improved reliability"""
    try:
        # Format messages for Ollama
        formatted_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        
        # Call Ollama API with reduced timeout
        response = ollama_session.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": model_info.get('temperature', 0.7),
                    "num_predict": model_info.get('max_tokens', 4096) // 2,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=15  # Reduced timeout to 15 seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["message"]["content"]
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise Exception(f"Ollama API error: {response.status_code}")
    
    except requests.exceptions.Timeout:
        logger.error("Ollama API timeout")
        raise Exception("Ollama API timeout")
    except requests.exceptions.ConnectionError:
        logger.error("Ollama connection error")
        raise Exception("Ollama connection error")
    except Exception as e:
        logger.error(f"Ollama error: {str(e)}")
        raise

def generate_mistral_response(messages, model, model_info):
    """Generate response using Mistral API"""
    if not MISTRAL_AVAILABLE or not MISTRAL_API_KEY:
        raise Exception("Mistral API not available")
    
    try:
        client = MistralClient(api_key=MISTRAL_API_KEY)
        
        # Format messages for Mistral
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            # Mistral doesn't support 'assistant' role, use 'assistant' instead
            if role == "assistant":
                role = "assistant"
            formatted_messages.append(ChatMessage(role=role, content=msg["content"]))
        
        # Call Mistral API
        response = client.chat(
            model=model,
            messages=formatted_messages,
            temperature=model_info.get('temperature', 0.7),
            max_tokens=model_info.get('max_tokens', 4096) // 2,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Mistral API error: {str(e)}")
        raise

def generate_huggingface_response(messages, model, model_info):
    """Generate response using Hugging Face API"""
    if not HF_AVAILABLE or not HF_TOKEN:
        raise Exception("Hugging Face API not available")
    
    try:
        # Format messages for Hugging Face
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        
        # Call Hugging Face API
        api = huggingface_hub.InferenceClient(token=HF_TOKEN)
        response = api.text_generation(
            prompt,
            model=f"huggingface/{model}",
            max_new_tokens=model_info.get('max_tokens', 1024) // 2,
            temperature=model_info.get('temperature', 0.7),
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise

def generate_simple_response(messages):
    """Generate a simple response when all other methods fail"""
    try:
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return "I'm sorry, I couldn't understand your request. Could you please try again?"
        
        # Simple keyword-based responses
        last_user_message = last_user_message.lower()
        
        if "hello" in last_user_message or "hi" in last_user_message:
            return "Hello! How can I help you today?"
        
        if "how are you" in last_user_message:
            return "I'm functioning well, thank you for asking! How can I assist you?"
        
        if "help" in last_user_message:
            return "I'd be happy to help! I can assist with data analysis, answer questions, or have a conversation. What would you like to know?"
        
        if "thank" in last_user_message:
            return "You're welcome! Is there anything else I can help you with?"
        
        if "bye" in last_user_message or "goodbye" in last_user_message:
            return "Goodbye! Feel free to come back if you have more questions."
        
        # Default response
        return "I'm currently experiencing some technical difficulties, but I'm still here to help. Could you try rephrasing your question or asking something else?"
    
    except Exception as e:
        logger.error(f"Error generating simple response: {str(e)}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."

def analyze_sentiment(text):
    """Analyze sentiment of text using the default model"""
    try:
        # First try Ollama if it's healthy
        if check_ollama_health():
            try:
                response = ollama_session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": "Analyze the sentiment of the following text and respond with only one word: 'positive', 'negative', or 'neutral'.\n\nText: " + text,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 10
                        }
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result["response"].strip().lower()
                    if sentiment in ['positive', 'negative', 'neutral']:
                        return sentiment
            except Exception as e:
                logger.warning(f"Error using Ollama for sentiment analysis: {str(e)}")
        
        # Simple rule-based sentiment analysis as fallback
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'like', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'dislike', 'worst']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return 'neutral'

def summarize_text(text, max_length=100):
    """Summarize text to specified length using the default model"""
    try:
        # Use Ollama for data privacy
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "mistral",
                "prompt": f"Summarize the following text in {max_length} words or less.\n\nText: " + text,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": max_length * 2
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["response"].strip()
        else:
            logger.error(f"Summarization error: {response.status_code}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return text[:max_length] + "..." if len(text) > max_length else text

# Initialize Ollama health check on module load
check_ollama_health()

class AIService:
    def __init__(self):
        self.config = Config()
        self.current_model = self.config.get_active_model()
        
    def _check_ollama_available(self):
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.config.OLLAMA_HOST}/api/tags")
            return response.status_code == 200
        except:
            return False
            
    def _get_appropriate_model(self, task_type='chat'):
        """Get appropriate model based on task type and availability"""
        if task_type == 'data_analysis' and self.config.ENABLE_PRIVATE_DATA_ANALYSIS:
            return 'ollama', self.config.MODELS['ollama']['data_analysis']
            
        if self.current_model == 'ollama':
            if not self._check_ollama_available():
                # Try to fall back to API services if Ollama is unavailable
                for service in ['claude', 'mistral', 'huggingface']:
                    if self.config.MODELS[service]['api_key_required'] and getattr(self.config, f"{service.upper()}_API_KEY"):
                        return service, self.config.MODELS[service]['model']
                raise RuntimeError("Ollama is unavailable and no API keys are configured")
            return 'ollama', self.config.MODELS['ollama']['chat']
            
        return self.current_model, self.config.MODELS[self.current_model]['model']
        
    async def generate_response(self, prompt, task_type='chat'):
        """Generate response using the appropriate AI model"""
        service, model = self._get_appropriate_model(task_type)
        
        try:
            if service == 'ollama':
                response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
                return response['message']['content']
                
            elif service == 'claude':
                client = Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return response.content[0].text
                
            elif service == 'mistral':
                client = MistralClient(api_key=self.config.MISTRAL_API_KEY)
                response = client.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return response.messages[0].content
                
            elif service == 'huggingface':
                tokenizer = AutoTokenizer.from_pretrained(model, token=self.config.HUGGINGFACE_API_KEY)
                model = AutoModelForCausalLM.from_pretrained(model, token=self.config.HUGGINGFACE_API_KEY)
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=1000)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        except Exception as e:
            logger.error(f"Error generating response with {service}: {str(e)}")
            if service != 'ollama' and self.config.FALLBACK_TO_LOCAL:
                logger.info("Falling back to Ollama")
                self.current_model = 'ollama'
                return await self.generate_response(prompt, task_type)
            raise
            
    async def analyze_data(self, data, analysis_type):
        """Analyze data using local Ollama models for privacy"""
        if not self.config.ENABLE_PRIVATE_DATA_ANALYSIS:
            raise ValueError("Private data analysis is disabled")
            
        model = self.config.MODELS['ollama']['data_analysis']
        prompt = f"Analyze the following data for {analysis_type}:\n{data}"
        
        try:
            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            raise 