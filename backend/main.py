"""
RR-Chatbot Backend Main Application

This is the main entry point for the RR-Chatbot backend application.
It initializes the FastAPI application and includes all the necessary routes.
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import time
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Import custom modules
from data_processing.data_loader import DataLoader
from data_processing.data_explorer import DataExplorer
from data_processing.data_cleaner import DataCleaner
from data_processing.data_wrangler import DataWrangler
from data_processing.data_analyzer import DataAnalyzer
from visualization.visualizer import Visualizer
from model_management.model_manager import ModelManager
from utils.logger import setup_logger
from utils.system_health import SystemHealthMonitor
from utils.user_logger import UserLogger
from api.ollama_api import OllamaAPI
from api.mistral_api import MistralAPI
from api.huggingface_api import HuggingFaceAPI

# Initialize FastAPI app
app = FastAPI(
    title="RR-Chatbot API",
    description="API for RR-Chatbot, an AI-powered data analysis and visualization chatbot",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logger = setup_logger("main_app", "logs/system_logs/app.log")

# Initialize components
data_loader = DataLoader()
data_explorer = DataExplorer()
data_cleaner = DataCleaner()
data_wrangler = DataWrangler()
data_analyzer = DataAnalyzer()
visualizer = Visualizer()
model_manager = ModelManager()
system_monitor = SystemHealthMonitor()
user_logger = UserLogger()
ollama_api = OllamaAPI()
mistral_api = MistralAPI()
huggingface_api = HuggingFaceAPI()

# Models for request/response
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    suggestions: list = []

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    message: str

class VisualizationRequest(BaseModel):
    file_id: str
    visualization_type: str
    parameters: dict = {}

class VisualizationResponse(BaseModel):
    visualization_id: str
    visualization_url: str
    data_summary: dict = {}

class DashboardRequest(BaseModel):
    visualization_ids: list
    dashboard_title: str
    layout: dict = {}

class DashboardResponse(BaseModel):
    dashboard_id: str
    dashboard_url: str

# Routes
@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "RR-Chatbot API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_data = system_monitor.check_health()
    return health_data

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint to interact with the chatbot.
    """
    try:
        # Log user interaction
        user_logger.log_interaction(request.user_id, "chat", request.message)
        
        # Process the message with Mistral API for general conversation
        response = mistral_api.generate_response(request.message)
        
        # Generate suggestions for next questions
        suggestions = mistral_api.generate_suggestions(request.message)
        
        return ChatResponse(response=response, suggestions=suggestions)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    """
    Upload a file for processing.
    """
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create user directory if it doesn't exist
        user_dir = f"uploads/{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        
        # Save the file
        file_path = f"{user_dir}/{timestamp}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Log the upload
        user_logger.log_interaction(user_id, "upload", file.filename)
        
        # Process the file in the background
        # This will be implemented in the data processing modules
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            status="success",
            message="File uploaded successfully and is being processed."
        )
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=dict)
async def analyze_data(file_id: str = Form(...), user_id: str = Form(...)):
    """
    Analyze the uploaded data.
    """
    try:
        # Find the file
        user_dir = f"uploads/{user_id}"
        files = os.listdir(user_dir)
        file_path = None
        
        for file in files:
            if file_id in file:
                file_path = f"{user_dir}/{file}"
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load the data
        df = data_loader.load_data(file_path)
        
        # Explore the data
        exploration_results = data_explorer.explore_data(df)
        
        # Clean the data
        cleaned_df = data_cleaner.clean_data(df)
        
        # Analyze the data
        analysis_results = data_analyzer.analyze_data(cleaned_df)
        
        # Log the analysis
        user_logger.log_interaction(user_id, "analyze", file_id)
        
        return {
            "exploration": exploration_results,
            "analysis": analysis_results,
        }
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize", response_model=VisualizationResponse)
async def create_visualization(request: VisualizationRequest, user_id: str = Form(...)):
    """
    Create a visualization based on the uploaded data.
    """
    try:
        # Find the file
        user_dir = f"uploads/{user_id}"
        files = os.listdir(user_dir)
        file_path = None
        
        for file in files:
            if request.file_id in file:
                file_path = f"{user_dir}/{file}"
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load the data
        df = data_loader.load_data(file_path)
        
        # Clean the data
        cleaned_df = data_cleaner.clean_data(df)
        
        # Create the visualization
        visualization_id = str(uuid.uuid4())
        visualization_path = visualizer.create_visualization(
            cleaned_df,
            request.visualization_type,
            request.parameters,
            visualization_id,
            user_id
        )
        
        # Get data summary
        data_summary = data_analyzer.get_summary(cleaned_df)
        
        # Log the visualization
        user_logger.log_interaction(user_id, "visualize", request.visualization_type)
        
        return VisualizationResponse(
            visualization_id=visualization_id,
            visualization_url=f"/visualizations/{user_id}/{visualization_id}",
            data_summary=data_summary
        )
    except Exception as e:
        logger.error(f"Error in visualize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard", response_model=DashboardResponse)
async def create_dashboard(request: DashboardRequest, user_id: str = Form(...)):
    """
    Create a dashboard with multiple visualizations.
    """
    try:
        # Generate a unique dashboard ID
        dashboard_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create the dashboard
        dashboard_path = visualizer.create_dashboard(
            request.visualization_ids,
            request.dashboard_title,
            request.layout,
            dashboard_id,
            user_id,
            timestamp
        )
        
        # Log the dashboard creation
        user_logger.log_interaction(user_id, "dashboard", request.dashboard_title)
        
        return DashboardResponse(
            dashboard_id=dashboard_id,
            dashboard_url=f"/dashboards/{user_id}/{dashboard_id}"
        )
    except Exception as e:
        logger.error(f"Error in dashboard endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{resource_type}/{resource_id}")
async def download_resource(resource_type: str, resource_id: str, format: str = "pdf", user_id: str = None):
    """
    Download a visualization or dashboard.
    """
    try:
        if resource_type not in ["visualization", "dashboard"]:
            raise HTTPException(status_code=400, detail="Invalid resource type")
        
        if format not in ["pdf", "png", "html"]:
            raise HTTPException(status_code=400, detail="Invalid format")
        
        # Find the resource
        resource_path = None
        if resource_type == "visualization":
            resource_path = f"visualizations/{user_id}/{resource_id}.{format}"
        else:
            resource_path = f"dashboards/{user_id}/{resource_id}.{format}"
        
        if not os.path.exists(resource_path):
            raise HTTPException(status_code=404, detail="Resource not found")
        
        # Log the download
        if user_id:
            user_logger.log_interaction(user_id, "download", f"{resource_type}_{resource_id}.{format}")
        
        return FileResponse(resource_path)
    except Exception as e:
        logger.error(f"Error in download endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """
    List available Ollama models.
    """
    try:
        models = ollama_api.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error in models endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/switch")
async def switch_model(model_name: str = Form(...)):
    """
    Switch the active Ollama model.
    """
    try:
        success = ollama_api.switch_model(model_name)
        if success:
            return {"status": "success", "message": f"Switched to model {model_name}"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to switch to model {model_name}")
    except Exception as e:
        logger.error(f"Error in switch model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 