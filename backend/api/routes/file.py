from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime
import json

from api.db.database import get_db
from api.db.redis_client import get_redis, set_cache, get_cache, delete_cache
from api.services.file_service import (
    save_upload_file, 
    delete_file, 
    get_file_info, 
    list_files, 
    is_allowed_file,
    UPLOAD_FOLDER,
    DATA_FOLDER
)
from api.services.visualization_service import generate_visualization, generate_dashboard

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Upload a file"""
    try:
        # Check if file is allowed
        if not is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(sorted(is_allowed_file.__defaults__[0]))}"
            )
        
        # Save the file
        file_path = await save_upload_file(file)
        
        # Get file info
        file_info = await get_file_info(file_path)
        
        # Cache the file info
        if redis_client and background_tasks:
            cache_key = f"file:{file_path}"
            background_tasks.add_task(set_cache, redis_client, cache_key, file_info, expire=3600)
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "file_info": file_info
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@router.get("/files")
async def get_files(
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Get a list of all files"""
    try:
        # Check cache
        if redis_client:
            cache_key = "files:list"
            cached_data = await get_cache(redis_client, cache_key)
            if cached_data:
                return {
                    "success": True,
                    "files": cached_data,
                    "cached": True
                }
        
        # Get files
        files = await list_files()
        
        # Cache the result
        if redis_client:
            await set_cache(redis_client, "files:list", files, expire=300)  # Cache for 5 minutes
        
        return {
            "success": True,
            "files": files,
            "cached": False
        }
    
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting files: {str(e)}"
        )

@router.get("/files/{file_path:path}")
async def get_file(
    file_path: str,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Get information about a file"""
    try:
        # Check cache
        if redis_client:
            cache_key = f"file:{file_path}"
            cached_data = await get_cache(redis_client, cache_key)
            if cached_data:
                return {
                    "success": True,
                    "file_info": cached_data,
                    "cached": True
                }
        
        # Get file info
        file_info = await get_file_info(file_path)
        
        # Cache the result
        if redis_client:
            await set_cache(redis_client, cache_key, file_info, expire=3600)
        
        return {
            "success": True,
            "file_info": file_info,
            "cached": False
        }
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}"
        )
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting file info: {str(e)}"
        )

@router.delete("/files/{file_path:path}")
async def delete_file_endpoint(
    file_path: str,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Delete a file"""
    try:
        # Delete the file
        result = await delete_file(file_path)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"File not found or could not be deleted: {file_path}"
            )
        
        # Delete cache
        if redis_client and background_tasks:
            cache_key = f"file:{file_path}"
            background_tasks.add_task(delete_cache, redis_client, cache_key)
            background_tasks.add_task(delete_cache, redis_client, "files:list")
        
        return {
            "success": True,
            "message": f"File deleted: {file_path}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting file: {str(e)}"
        )

@router.get("/download/{file_path:path}")
async def download_file(
    file_path: str,
    db: AsyncSession = Depends(get_db)
):
    """Download a file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Return the file
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading file: {str(e)}"
        )

@router.post("/visualize")
async def visualize_file(
    file_path: str,
    viz_type: str,
    params: Dict[str, Any] = None,
    force_refresh: bool = False,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Generate a visualization for a file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Check cache if not forcing refresh
        if redis_client and not force_refresh:
            cache_key = f"viz:{file_path}:{viz_type}:{json.dumps(params)}"
            cached_data = await get_cache(redis_client, cache_key)
            if cached_data:
                return {
                    "success": True,
                    "visualization": cached_data,
                    "cached": True
                }
        
        # Generate visualization
        visualization = await generate_visualization(file_path, viz_type, params)
        
        # Cache the result
        if redis_client and background_tasks:
            cache_key = f"viz:{file_path}:{viz_type}:{json.dumps(params)}"
            background_tasks.add_task(set_cache, redis_client, cache_key, visualization, expire=3600)
        
        return {
            "success": True,
            "visualization": visualization,
            "cached": False
        }
    
    except ValueError as e:
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

@router.post("/dashboard")
async def generate_file_dashboard(
    file_path: str,
    title: str = None,
    force_refresh: bool = False,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Generate a dashboard for a file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Check cache if not forcing refresh
        if redis_client and not force_refresh:
            cache_key = f"dashboard:{file_path}"
            cached_data = await get_cache(redis_client, cache_key)
            if cached_data:
                return {
                    "success": True,
                    "dashboard": cached_data,
                    "cached": True
                }
        
        # Generate dashboard
        dashboard = await generate_dashboard(file_path, title)
        
        # Cache the result
        if redis_client and background_tasks:
            cache_key = f"dashboard:{file_path}"
            background_tasks.add_task(set_cache, redis_client, cache_key, dashboard, expire=3600)
        
        return {
            "success": True,
            "dashboard": dashboard,
            "cached": False
        }
    
    except ValueError as e:
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
        file_path = os.path.join(UPLOAD_FOLDER, "visualizations", filename)
        
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