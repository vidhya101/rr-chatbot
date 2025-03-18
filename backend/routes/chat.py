from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from ..services.model_manager import ModelManager
from ..dependencies import get_model_manager

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = datetime.now()

class ChatRequest(BaseModel):
    message: str
    model_preference: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    model_used: str
    latency: float
    conversation_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> ChatResponse:
    """
    Chat endpoint that handles model switching and fallback
    """
    try:
        start_time = datetime.now()
        
        # Use preferred model if specified and available
        if request.model_preference:
            if request.model_preference in model_manager.models:
                model_manager.current_model = request.model_preference
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested model {request.model_preference} not available"
                )
        
        # Get response from model manager
        response = await model_manager.get_response(request.message)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            message=response,
            model_used=model_manager.current_model,
            latency=latency,
            conversation_id=request.conversation_id or "new_conversation"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@router.get("/models/status")
async def get_models_status(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get status of all configured models
    """
    return {
        name: {
            "is_active": config.is_active,
            "error_count": config.error_count,
            "avg_latency": config.avg_latency,
            "last_used": config.last_used
        }
        for name, config in model_manager.models.items()
    }

@router.post("/models/{model_name}/toggle")
async def toggle_model(
    model_name: str,
    is_active: bool,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Toggle model availability
    """
    if model_name not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found"
        )
    
    model_manager.update_model_status(model_name, is_active)
    return {"status": "success", "model": model_name, "is_active": is_active} 