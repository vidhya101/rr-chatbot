from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
import pandas as pd
import logging

from api.db.database import get_db
from api.services.data_viz_service import DataVizService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create service instance
data_viz_service = DataVizService()

@router.post("/load-data")
async def load_data(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Load data from a file"""
    try:
        # Read file content
        content = await file.read()
        
        # Save to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Load data
            result = await data_viz_service.load_data(temp_path)
            return result
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data: {str(e)}"
        )

@router.post("/create-visualization")
async def create_visualization(
    viz_type: str,
    params: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Create a visualization"""
    try:
        result = await data_viz_service.create_visualization(viz_type, params)
        return result
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create visualization: {str(e)}"
        )

@router.get("/get-insights")
async def get_insights(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get insights about the loaded data"""
    try:
        result = await data_viz_service.get_insights()
        return result
    
    except Exception as e:
        logger.error(f"Error getting insights: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get insights: {str(e)}"
        )

@router.post("/process-query")
async def process_query(
    query: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Process a natural language query about the data"""
    try:
        result = await data_viz_service.process_query(query)
        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

@router.get("/data-preview")
async def data_preview(
    rows: int = 10,
    columns: str = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get a preview of the data"""
    try:
        # Get data from service
        if not data_viz_service.data is None:
            # Convert columns string to list if provided
            column_list = columns.split(',') if columns else None
            
            # Get preview data
            if column_list:
                try:
                    preview = data_viz_service.data[column_list].head(rows)
                except KeyError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid column names: {str(e)}"
                    )
            else:
                preview = data_viz_service.data.head(rows)
            
            # Convert to JSON
            preview_json = preview.to_dict(orient='records')
            
            # Get total counts
            total_rows = len(data_viz_service.data)
            total_columns = len(data_viz_service.data.columns)
            
            return {
                'status': 'success',
                'preview': preview_json,
                'total_rows': total_rows,
                'total_columns': total_columns
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="No data loaded"
            )
    
    except Exception as e:
        logger.error(f"Error getting data preview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get data preview: {str(e)}"
        )

@router.post("/export-dashboard")
async def export_dashboard(
    config: Dict[str, Any],
    format: str = 'html',
    db: AsyncSession = Depends(get_db)
):
    """Export a dashboard"""
    try:
        # Export dashboard
        output_path = await data_viz_service.export_dashboard(config, format)
        
        try:
            # Return file
            return FileResponse(
                output_path,
                media_type='text/html' if format == 'html' else 'application/pdf',
                filename=f'dashboard.{format}'
            )
        finally:
            # Clean up temp file
            import os
            if os.path.exists(output_path):
                os.remove(output_path)
    
    except Exception as e:
        logger.error(f"Error exporting dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export dashboard: {str(e)}"
        ) 