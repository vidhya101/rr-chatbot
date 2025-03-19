import os
import json
import requests
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from huggingface_hub import InferenceClient
from config import Config

class ModelService:
    def __init__(self):
        """Initialize model clients"""
        self.clients = {}
        
        # Initialize Claude client if API key exists
        if Config.CLAUDE_API_KEY:
            self.clients['claude'] = Anthropic(api_key=Config.CLAUDE_API_KEY)
            
        # Initialize Mistral client if API key exists
        if Config.MISTRAL_API_KEY:
            self.clients['mistral'] = MistralClient(api_key=Config.MISTRAL_API_KEY)
            
        # Initialize Hugging Face client if API key exists
        if Config.HUGGINGFACE_API_KEY:
            self.clients['huggingface'] = InferenceClient(token=Config.HUGGINGFACE_API_KEY)
    
    def get_response(self, prompt, task_type='chat', **kwargs):
        """
        Get response from appropriate model based on task type
        
        Args:
            prompt: str, the input prompt
            task_type: str, one of 'chat', 'data_analysis', 'code'
            **kwargs: Additional arguments for specific models
            
        Returns:
            str: The model's response
        """
        try:
            provider, model = Config.get_model_for_task(task_type)
            
            if provider == 'ollama':
                return self._get_ollama_response(prompt, model)
            elif provider == 'claude':
                return self._get_claude_response(prompt, model)
            elif provider == 'mistral':
                return self._get_mistral_response(prompt, model)
            elif provider == 'huggingface':
                return self._get_huggingface_response(prompt, model)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            # Log error and try fallback if enabled
            print(f"Error getting response from {provider}: {str(e)}")
            if Config.FALLBACK_TO_LOCAL and provider != 'ollama':
                try:
                    return self._get_ollama_response(prompt, Config.MODELS['ollama'].get(task_type, 'mistral'))
                except Exception as e2:
                    raise Exception(f"Both primary and fallback models failed. Primary: {str(e)}, Fallback: {str(e2)}")
            raise
    
    def _get_ollama_response(self, prompt, model):
        """Get response from Ollama model"""
        response = requests.post(
            f"{Config.OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        )
        response.raise_for_status()
        return response.json()['response']
    
    def _get_claude_response(self, prompt, model):
        """Get response from Claude model"""
        client = self.clients.get('claude')
        if not client:
            raise ValueError("Claude client not initialized")
            
        response = client.messages.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000
        )
        return response.content[0].text
    
    def _get_mistral_response(self, prompt, model):
        """Get response from Mistral model"""
        client = self.clients.get('mistral')
        if not client:
            raise ValueError("Mistral client not initialized")
            
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        response = client.chat(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    
    def _get_huggingface_response(self, prompt, model):
        """Get response from Hugging Face model"""
        client = self.clients.get('huggingface')
        if not client:
            raise ValueError("Hugging Face client not initialized")
            
        response = client.text_generation(
            prompt=prompt,
            model=model,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )
        return response[0]['generated_text'] 