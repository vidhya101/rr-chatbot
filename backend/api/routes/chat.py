from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, Union
import os
import logging
import json
import uuid
import asyncio
from datetime import datetime

from api.db.database import get_db
from api.db.redis_client import get_redis, publish_message, subscribe_to_channel
from api.services.chat_service import generate_response, generate_streaming_response
from api.models.chat import (
    ChatRequest,
    ChatResponse,
    MessageRole,
    Message,
    ChatHistory,
    DataVisualizationRequest
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Send a message to the chatbot and get a response"""
    try:
        # Generate a unique message ID
        message_id = str(uuid.uuid4())
        
        # Log the request
        logger.info(f"Chat request received: {request.message[:50]}...")
        
        # Generate response
        response = await generate_response(
            message=request.message,
            chat_history=request.chat_history,
            model=request.model,
            settings=request.settings,
            data_context=request.data_context if hasattr(request, 'data_context') else None
        )
        
        # Create response object
        chat_response = ChatResponse(
            message_id=message_id,
            response=response,
            model=request.model,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Publish the message to Redis for real-time updates
        if redis_client:
            background_tasks.add_task(
                publish_message,
                redis_client,
                f"chat:{request.user_id}",
                chat_response.dict()
            )
        
        return chat_response
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )

@router.websocket("/ws/chat/{user_id}")
async def websocket_chat(
    websocket: WebSocket,
    user_id: str,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    # Store the connection
    active_connections[user_id] = websocket
    
    try:
        # Send a welcome message
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to chat server",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Listen for messages from the client
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message_data = json.loads(data)
                
                # Create a chat request
                chat_request = ChatRequest(
                    message=message_data.get("message", ""),
                    chat_history=message_data.get("chat_history", []),
                    model=message_data.get("model", "gpt-3.5-turbo"),
                    settings=message_data.get("settings", {}),
                    user_id=user_id,
                    stream=True
                )
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "message_received",
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Generate streaming response
                async for chunk in generate_streaming_response(
                    message=chat_request.message,
                    chat_history=chat_request.chat_history,
                    model=chat_request.model,
                    settings=chat_request.settings
                ):
                    # Send the chunk to the client
                    await websocket.send_json({
                        "type": "response_chunk",
                        "content": chunk,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Send end of response
                await websocket.send_json({
                    "type": "response_complete",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except json.JSONDecodeError:
                # Send error message
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON data",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                # Send error message
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        # Remove the connection
        if user_id in active_connections:
            del active_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        # Remove the connection
        if user_id in active_connections:
            del active_connections[user_id]

@router.post("/simple-chat")
async def simple_chat(
    message: str,
    model: str = "gpt-3.5-turbo",
    db: AsyncSession = Depends(get_db)
):
    """Simple chat endpoint that doesn't require authentication"""
    try:
        # Generate response
        response = await generate_response(
            message=message,
            chat_history=[],
            model=model,
            settings={}
        )
        
        return {
            "success": True,
            "message": response,
            "model": model,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in simple chat endpoint: {str(e)}")
        return {
            "success": False,
            "error": "Failed to generate response",
            "message": f"I'm sorry, but I encountered an error processing your message: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/chat/visualize")
async def visualize_data(
    request: DataVisualizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a visualization based on the request"""
    try:
        # Generate response with visualization context
        response = await generate_response(
            message=request.query,
            chat_history=request.chat_history,
            model=request.model,
            settings=request.settings,
            data_context={
                'type': 'visualization',
                'viz_type': request.viz_type,
                'params': request.params
            }
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in visualization endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create visualization: {str(e)}"
        ) 