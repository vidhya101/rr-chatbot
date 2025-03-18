import os
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from datetime import datetime

# Import AI model clients
from openai import AsyncOpenAI
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

# Import models
from api.models.chat import Message, MessageRole, ChatHistory

# Import data visualization service
from api.services.data_viz_service import DataVizService

# Configure logging
logger = logging.getLogger(__name__)

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
mistral_client = MistralAsyncClient(api_key=os.environ.get("MISTRAL_API_KEY"))

# Initialize data visualization service
data_viz_service = DataVizService()

async def generate_response(
    message: str,
    chat_history: List[Dict[str, Any]] = None,
    model: str = "gpt-3.5-turbo",
    settings: Dict[str, Any] = None,
    data_context: Dict[str, Any] = None
) -> Union[str, Dict[str, Any]]:
    """
    Generate a response from the AI model
    
    Args:
        message: The user's message
        chat_history: List of previous messages
        model: The model to use (e.g., gpt-3.5-turbo, mistral-medium)
        settings: Additional settings for the model
        data_context: Optional data context for visualization requests
        
    Returns:
        The generated response or visualization data
    """
    if chat_history is None:
        chat_history = []
    
    if settings is None:
        settings = {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    try:
        # Check if this is a data visualization request
        if data_context and 'type' in data_context:
            if data_context['type'] == 'visualization':
                # Create visualization
                return await data_viz_service.create_visualization(
                    data_context['viz_type'],
                    data_context['params']
                )
            elif data_context['type'] == 'insights':
                # Get insights
                return await data_viz_service.get_insights()
            elif data_context['type'] == 'query':
                # Process data query
                return await data_viz_service.process_query(message)
        
        # Format the chat history
        formatted_history = format_chat_history(chat_history)
        
        # Add system message for data context if available
        if data_context:
            formatted_history.insert(0, {
                "role": "system",
                "content": f"Current data context: {data_context}"
            })
        
        # Add the user's message
        formatted_history.append({"role": "user", "content": message})
        
        # Generate response based on the model provider
        if "gpt" in model.lower():
            return await generate_openai_response(formatted_history, model, settings)
        elif "mistral" in model.lower():
            return await generate_mistral_response(formatted_history, model, settings)
        else:
            # Default to OpenAI
            logger.warning(f"Unknown model {model}, defaulting to gpt-3.5-turbo")
            return await generate_openai_response(formatted_history, "gpt-3.5-turbo", settings)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I'm sorry, but I encountered an error: {str(e)}"

async def generate_streaming_response(
    message: str,
    chat_history: List[Dict[str, Any]] = None,
    model: str = "gpt-3.5-turbo",
    settings: Dict[str, Any] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from the AI model
    
    Args:
        message: The user's message
        chat_history: List of previous messages
        model: The model to use
        settings: Additional settings for the model
        
    Yields:
        Chunks of the generated response
    """
    if chat_history is None:
        chat_history = []
    
    if settings is None:
        settings = {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    try:
        # Format the chat history
        formatted_history = format_chat_history(chat_history)
        
        # Add the user's message
        formatted_history.append({"role": "user", "content": message})
        
        # Generate streaming response based on the model provider
        if "gpt" in model.lower():
            async for chunk in generate_openai_streaming_response(formatted_history, model, settings):
                yield chunk
        elif "mistral" in model.lower():
            async for chunk in generate_mistral_streaming_response(formatted_history, model, settings):
                yield chunk
        else:
            # Default to OpenAI
            logger.warning(f"Unknown model {model}, defaulting to gpt-3.5-turbo")
            async for chunk in generate_openai_streaming_response(formatted_history, "gpt-3.5-turbo", settings):
                yield chunk
    
    except Exception as e:
        logger.error(f"Error generating streaming response: {str(e)}")
        yield f"I'm sorry, but I encountered an error: {str(e)}"

def format_chat_history(chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format chat history for the AI model"""
    formatted_history = []
    
    for message in chat_history:
        role = message.get("role", "user").lower()
        # Map roles to OpenAI format
        if role == "bot" or role == "assistant":
            role = "assistant"
        elif role != "system" and role != "user":
            role = "user"
        
        formatted_history.append({
            "role": role,
            "content": message.get("content", "")
        })
    
    return formatted_history

async def generate_openai_response(
    messages: List[Dict[str, str]],
    model: str,
    settings: Dict[str, Any]
) -> str:
    """Generate a response using OpenAI"""
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings.get("temperature", 0.7),
            max_tokens=settings.get("max_tokens", 1000),
            top_p=settings.get("top_p", 1.0),
            frequency_penalty=settings.get("frequency_penalty", 0.0),
            presence_penalty=settings.get("presence_penalty", 0.0)
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

async def generate_openai_streaming_response(
    messages: List[Dict[str, str]],
    model: str,
    settings: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using OpenAI"""
    try:
        stream = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings.get("temperature", 0.7),
            max_tokens=settings.get("max_tokens", 1000),
            top_p=settings.get("top_p", 1.0),
            frequency_penalty=settings.get("frequency_penalty", 0.0),
            presence_penalty=settings.get("presence_penalty", 0.0),
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        logger.error(f"OpenAI streaming API error: {str(e)}")
        raise

async def generate_mistral_response(
    messages: List[Dict[str, str]],
    model: str,
    settings: Dict[str, Any]
) -> str:
    """Generate a response using Mistral AI"""
    try:
        # Convert to Mistral format
        mistral_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        response = await mistral_client.chat(
            model=model,
            messages=mistral_messages,
            temperature=settings.get("temperature", 0.7),
            max_tokens=settings.get("max_tokens", 1000),
            top_p=settings.get("top_p", 1.0)
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Mistral API error: {str(e)}")
        raise

async def generate_mistral_streaming_response(
    messages: List[Dict[str, str]],
    model: str,
    settings: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using Mistral AI"""
    try:
        # Convert to Mistral format
        mistral_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        stream = await mistral_client.chat_stream(
            model=model,
            messages=mistral_messages,
            temperature=settings.get("temperature", 0.7),
            max_tokens=settings.get("max_tokens", 1000),
            top_p=settings.get("top_p", 1.0)
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        logger.error(f"Mistral streaming API error: {str(e)}")
        raise 