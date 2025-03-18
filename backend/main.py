"""
RR-Chatbot Backend Main Application - Simplified Version

This is a simplified version of the main entry point for the RR-Chatbot backend application.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create request model
class ChatMessage(BaseModel):
    message: str
    chatId: Optional[str] = None
    model: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="RR-Chatbot API",
    description="API for RR-Chatbot - An AI-powered data analysis and visualization platform",
    version="1.0.0"
)

# Add CORS middleware with more permissive configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Create upload directories if they don't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(os.path.join(UPLOAD_FOLDER, "data"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "temp"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "visualizations"), exist_ok=True)

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "ok", "message": "API is running"}

# Root endpoint
@app.get("/api")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to RR-Chatbot API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Chat endpoint
@app.post("/api/chat")
async def chat(chat_message: ChatMessage):
    """Chat endpoint that handles messages"""
    try:
        logger.info(f"Received message: {chat_message.message}")
        logger.info(f"Chat ID: {chat_message.chatId}")
        logger.info(f"Model: {chat_message.model}")

        # Here you would typically process the message with your AI model
        # For now, we'll just echo back a response
        response = {
            "success": True,
            "message": f"I received your message: {chat_message.message}",
            "chatId": chat_message.chatId or "new_chat_123",
            "model": chat_message.model or "default"
        }
        
        logger.info(f"Sending response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 5000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    ) 