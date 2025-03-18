from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime

class VisualizationType(str, Enum):
    AUTO = "auto"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    BAR = "bar"
    LINE = "line"
    HEATMAP = "heatmap"
    BOXPLOT = "boxplot"
    PIE = "pie"
    AREA = "area"
    BUBBLE = "bubble"
    RADAR = "radar"
    POLAR = "polar"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    NETWORK = "network"

class VisualizationRequest(BaseModel):
    file_path: str = Field(..., description="Path to the data file")
    viz_type: VisualizationType = Field(default=VisualizationType.AUTO, description="Type of visualization to generate")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the visualization")
    force_refresh: bool = Field(default=False, description="Force refresh the visualization instead of using cache")

class VisualizationResponse(BaseModel):
    visualization_id: str = Field(..., description="Unique ID of the visualization")
    visualization_url: str = Field(..., description="URL to access the visualization")
    title: str = Field(..., description="Title of the visualization")
    description: str = Field(..., description="Description of the visualization")
    viz_type: str = Field(..., description="Type of visualization")
    file_path: str = Field(..., description="Path to the data file")
    created_at: str = Field(..., description="Timestamp when the visualization was created")
    
    class Config:
        schema_extra = {
            "example": {
                "visualization_id": "viz_123456",
                "visualization_url": "/api/visualization/visualizations/viz_123456.png",
                "title": "Distribution of Age",
                "description": "Histogram showing the distribution of age in the dataset",
                "viz_type": "histogram",
                "file_path": "uploads/data.csv",
                "created_at": "2023-11-15T12:34:56.789Z"
            }
        }

class DashboardRequest(BaseModel):
    file_path: str = Field(..., description="Path to the data file")
    title: Optional[str] = Field(default=None, description="Title of the dashboard")
    force_refresh: bool = Field(default=False, description="Force refresh the dashboard instead of using cache")

class Visualization(BaseModel):
    id: str = Field(..., description="Unique ID of the visualization")
    url: str = Field(..., description="URL to access the visualization")
    title: str = Field(..., description="Title of the visualization")
    description: str = Field(..., description="Description of the visualization")
    type: str = Field(..., description="Type of visualization")

class DataStats(BaseModel):
    rows: int = Field(..., description="Number of rows in the dataset")
    columns: int = Field(..., description="Number of columns in the dataset")
    numeric_columns: List[str] = Field(..., description="List of numeric columns")
    categorical_columns: List[str] = Field(..., description="List of categorical columns")
    missing_values: int = Field(..., description="Number of missing values")
    column_types: Dict[str, str] = Field(..., description="Types of each column")

class DashboardResponse(BaseModel):
    dashboard_id: str = Field(..., description="Unique ID of the dashboard")
    visualizations: List[Visualization] = Field(..., description="List of visualizations in the dashboard")
    stats: DataStats = Field(..., description="Statistics about the dataset")
    file_path: str = Field(..., description="Path to the data file")
    created_at: str = Field(..., description="Timestamp when the dashboard was created")
    
    class Config:
        schema_extra = {
            "example": {
                "dashboard_id": "dash_123456",
                "visualizations": [
                    {
                        "id": "viz_123456",
                        "url": "/api/visualization/visualizations/viz_123456.png",
                        "title": "Distribution of Age",
                        "description": "Histogram showing the distribution of age in the dataset",
                        "type": "histogram"
                    }
                ],
                "stats": {
                    "rows": 1000,
                    "columns": 10,
                    "numeric_columns": ["age", "income", "score"],
                    "categorical_columns": ["gender", "country", "education"],
                    "missing_values": 15,
                    "column_types": {
                        "age": "int64",
                        "income": "float64",
                        "score": "float64",
                        "gender": "object",
                        "country": "object",
                        "education": "object"
                    }
                },
                "file_path": "uploads/data.csv",
                "created_at": "2023-11-15T12:34:56.789Z"
            }
        } 