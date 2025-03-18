from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import uuid

class MessageRole(str, Enum):
    """Enum for message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    BOT = "bot"  # Alias for assistant

class Message(BaseModel):
    """Model for a chat message"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        use_enum_values = True

class ChatHistory(BaseModel):
    """Model for chat history"""
    messages: List[Message] = []

class ChatRequest(BaseModel):
    """Model for chat request"""
    message: str
    chat_history: List[Dict[str, Any]] = []
    model: str = "gpt-3.5-turbo"
    settings: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    stream: bool = False
    
    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    """Model for chat response"""
    message_id: str
    response: str
    model: str
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "message_id": "550e8400-e29b-41d4-a716-446655440000",
                "response": "Hello! How can I assist you today?",
                "model": "gpt-3.5-turbo",
                "timestamp": "2023-11-15T12:34:56.789Z"
            }
        }

class ModelInfo(BaseModel):
    """Model for AI model information"""
    id: str
    name: str
    provider: str
    description: str
    max_tokens: int
    pricing: Dict[str, float]  # input/output pricing
    capabilities: List[str]
    available: bool = True

class ModelList(BaseModel):
    """Model for list of available AI models"""
    models: List[ModelInfo] 