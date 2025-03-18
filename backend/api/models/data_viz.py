from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class DataVisualizationRequest(BaseModel):
    """Request model for data visualization"""
    query: str = Field(..., description="Natural language query about the data")
    viz_type: str = Field(..., description="Type of visualization to create")
    params: Dict[str, Any] = Field(default_factory=dict, description="Visualization parameters")
    chat_history: List[Dict[str, Any]] = Field(default_factory=list, description="Chat history")
    model: str = Field(default="gpt-3.5-turbo", description="Model to use for processing")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Model settings")

class DataVisualizationResponse(BaseModel):
    """Response model for data visualization"""
    status: str = Field(..., description="Response status")
    type: str = Field(..., description="Response type (e.g., image)")
    format: str = Field(..., description="Response format (e.g., png)")
    data: str = Field(..., description="Base64 encoded visualization data")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")

class DataInsightsResponse(BaseModel):
    """Response model for data insights"""
    status: str = Field(..., description="Response status")
    insights: Dict[str, Any] = Field(..., description="Data insights")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")

class DataQueryResponse(BaseModel):
    """Response model for data queries"""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data_info: Optional[Dict[str, Any]] = Field(None, description="Data information")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp") 