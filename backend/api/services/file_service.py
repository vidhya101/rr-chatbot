import os
import logging
import uuid
import shutil
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
from fastapi import UploadFile, HTTPException

# Configure logging
logger = logging.getLogger(__name__)

# Set up upload directory
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "./uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create subdirectories
DATA_FOLDER = os.path.join(UPLOAD_FOLDER, "data")
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, "temp")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 
    'txt', 'pdf', 'doc', 'docx', 'ppt', 'pptx'
}

async def save_upload_file(upload_file: UploadFile, folder: str = DATA_FOLDER) -> str:
    """
    Save an uploaded file to the specified folder
    
    Args:
        upload_file: The uploaded file
        folder: The folder to save the file to
        
    Returns:
        The path to the saved file
    """
    try:
        # Generate a unique filename
        filename = f"{uuid.uuid4()}_{upload_file.filename}"
        file_path = os.path.join(folder, filename)
        
        # Save the file
        async with aiofiles.open(file_path, 'wb') as out_file:
            # Read the file in chunks
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)
        
        logger.info(f"File saved: {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

async def delete_file(file_path: str) -> bool:
    """
    Delete a file
    
    Args:
        file_path: The path to the file to delete
        
    Returns:
        True if the file was deleted, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File deleted: {file_path}")
            return True
        else:
            logger.warning(f"File not found: {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False

async def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file
    
    Args:
        file_path: The path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file stats
        stats = os.stat(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower().lstrip('.')
        
        # Basic file info
        file_info = {
            "file_path": file_path,
            "file_name": file_name,
            "file_size": stats.st_size,
            "file_type": file_ext,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat()
        }
        
        # For data files, add preview and stats
        if file_ext in ['csv', 'xlsx', 'xls', 'json', 'parquet', 'feather']:
            # Run in a thread pool to avoid blocking
            preview_data, stats = await asyncio.to_thread(
                get_data_preview_and_stats,
                file_path
            )
            
            file_info["preview"] = preview_data
            file_info["stats"] = stats
        
        return file_info
    
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        raise

def get_data_preview_and_stats(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get a preview and statistics for a data file
    
    Args:
        file_path: The path to the data file
        
    Returns:
        Tuple of (preview_data, stats)
    """
    try:
        # Load the data
        df = load_data(file_path)
        
        # Get preview (first 10 rows)
        preview = df.head(10).to_dict(orient='records')
        
        # Get basic stats
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "missing_values": int(df.isna().sum().sum())
        }
        
        return preview, stats
    
    except Exception as e:
        logger.error(f"Error getting data preview: {str(e)}")
        return [], {"error": str(e)}

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a file into a pandas DataFrame"""
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        elif file_ext == '.feather':
            return pd.read_feather(file_path)
        else:
            # Try CSV as a default
            return pd.read_csv(file_path)
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

async def list_files(folder: str = DATA_FOLDER) -> List[Dict[str, Any]]:
    """
    List all files in the specified folder
    
    Args:
        folder: The folder to list files from
        
    Returns:
        List of dictionaries with file information
    """
    try:
        files = []
        
        # List all files in the folder
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Get file stats
            stats = os.stat(file_path)
            file_ext = os.path.splitext(file_name)[1].lower().lstrip('.')
            
            # Add file info
            files.append({
                "file_path": file_path,
                "file_name": file_name,
                "file_size": stats.st_size,
                "file_type": file_ext,
                "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })
        
        # Sort by modified time (newest first)
        files.sort(key=lambda x: x["modified_at"], reverse=True)
        
        return files
    
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise

def is_allowed_file(filename: str) -> bool:
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 