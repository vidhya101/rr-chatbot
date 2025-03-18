import os
import requests
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.current_model = None
        self.available_models = self.get_available_models()
        self.chat_history = []

    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                return {model['name']: model for model in response.json().get('models', [])}
            logger.error(f"Failed to get models: {response.status_code}")
            return {}
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return {}

    def select_model(self, model_name: str) -> bool:
        """Select a model to use"""
        if model_name in self.available_models:
            self.current_model = model_name
            logger.info(f"Selected model: {model_name}")
            return True
        return False

    def chat(self, message: str, model: Optional[str] = None, chat_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat message to Ollama"""
        try:
            model_name = model or self.current_model or "mistral:latest"
            
            # Get chat history for context
            messages = self._get_chat_history(chat_id)
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "")
                
                # Save messages to history
                self._save_to_history(chat_id, "user", message)
                self._save_to_history(chat_id, "assistant", assistant_message)
                
                return {
                    "success": True,
                    "response": assistant_message,
                    "model": model_name,
                    "chat_id": chat_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

        except requests.Timeout:
            error_msg = "Request timed out"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def _get_chat_history(self, chat_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Get chat history for a specific chat"""
        if not chat_id:
            return []
        return [msg for msg in self.chat_history if msg.get('chat_id') == chat_id]

    def _save_to_history(self, chat_id: Optional[str], role: str, content: str):
        """Save message to chat history"""
        if not chat_id:
            return
        
        self.chat_history.append({
            'chat_id': chat_id,
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep history size manageable (last 100 messages per chat)
        chat_messages = [msg for msg in self.chat_history if msg.get('chat_id') == chat_id]
        if len(chat_messages) > 100:
            # Remove oldest messages
            to_remove = len(chat_messages) - 100
            self.chat_history = [msg for msg in self.chat_history 
                               if msg.get('chat_id') != chat_id or 
                               msg['timestamp'] > chat_messages[to_remove-1]['timestamp']]

    def get_chat_history(self, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get chat history with optional filtering by chat_id"""
        if chat_id:
            return [msg for msg in self.chat_history if msg.get('chat_id') == chat_id]
        return self.chat_history

    def clear_chat_history(self, chat_id: Optional[str] = None):
        """Clear chat history for a specific chat or all chats"""
        if chat_id:
            self.chat_history = [msg for msg in self.chat_history if msg.get('chat_id') != chat_id]
        else:
            self.chat_history = []

    def health_check(self) -> Dict[str, Any]:
        """Check health of chat service"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "models_available": len(self.available_models),
                    "current_model": self.current_model,
                    "active_chats": len(set(msg.get('chat_id') for msg in self.chat_history if msg.get('chat_id')))
                }
            return {
                "status": "unhealthy",
                "error": f"Ollama API returned {response.status_code}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Create singleton instance
chat_service = ChatService() 