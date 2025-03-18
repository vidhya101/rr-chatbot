from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, Union
import os
import logging
import json
from datetime import datetime

from api.db.database import get_db
from api.db.redis_client import get_redis, set_cache, get_cache
from api.models.visualization import VisualizationRequest, VisualizationResponse, DashboardRequest, DashboardResponse
from api.services.visualization_service import generate_visualization, generate_dashboard

# Configure logging
logger = logging.getLogger(__name__)

# Set up visualization directory
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "./uploads")
VISUALIZATION_FOLDER = os.path.join(UPLOAD_FOLDER, "visualizations")
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Create router
router = APIRouter()

@router.post("/visualize", response_model=VisualizationResponse)
async def visualize(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Generate a visualization for a dataset"""
    try:
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.file_path}"
            )
        
        # Check cache if not forcing refresh
        if redis_client and not request.force_refresh:
            cache_key = f"viz:{request.file_path}:{request.viz_type}:{json.dumps(request.params)}"
            cached_data = await get_cache(redis_client, cache_key)
            if cached_data:
                return VisualizationResponse(**cached_data)
        
        # Generate visualization
        result = await generate_visualization(
            file_path=request.file_path,
            viz_type=request.viz_type,
            params=request.params
        )
        
        # Create response
        response = VisualizationResponse(
            visualization_id=result["visualization_id"],
            visualization_url=result["visualization_url"],
            title=result["title"],
            description=result["description"],
            viz_type=result["viz_type"],
            file_path=result["file_path"],
            created_at=result["created_at"]
        )
        
        # Cache the result
        if redis_client:
            background_tasks.add_task(
                set_cache,
                redis_client,
                f"viz:{request.file_path}:{request.viz_type}:{json.dumps(request.params)}",
                response.dict(),
                expire=3600
            )
        
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization: {str(e)}"
        )

@router.post("/dashboard", response_model=DashboardResponse)
async def dashboard(
    request: DashboardRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Generate a comprehensive dashboard for a dataset"""
    try:
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.file_path}"
            )
        
        # Check cache if not forcing refresh
        if redis_client and not request.force_refresh:
            cache_key = f"dashboard:{request.file_path}"
            cached_data = await get_cache(redis_client, cache_key)
            if cached_data:
                return DashboardResponse(**cached_data)
        
        # Generate dashboard
        result = await generate_dashboard(
            file_path=request.file_path,
            title=request.title
        )
        
        # Create response
        response = DashboardResponse(
            dashboard_id=result["dashboard_id"],
            title=result.get("title", "Dashboard"),
            visualizations=result["visualizations"],
            stats=result["stats"],
            file_path=result["file_path"],
            created_at=result["created_at"]
        )
        
        # Cache the result
        if redis_client:
            background_tasks.add_task(
                set_cache,
                redis_client,
                f"dashboard:{request.file_path}",
                response.dict(),
                expire=3600
            )
        
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating dashboard: {str(e)}"
        )

@router.get("/visualizations/{filename}")
async def get_visualization(
    filename: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a visualization image"""
    try:
        # Construct the file path
        file_path = os.path.join(VISUALIZATION_FOLDER, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Visualization not found: {filename}"
            )
        
        # Return the file
        return FileResponse(
            path=file_path,
            media_type="image/png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting visualization: {str(e)}"
        ) 