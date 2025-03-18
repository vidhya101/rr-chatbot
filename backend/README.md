# RR-Chatbot Backend

The backend for RR-Chatbot, built with FastAPI, SQLAlchemy, and Redis.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Async Support**: Fully asynchronous API endpoints
- **SQLAlchemy ORM**: SQL toolkit and ORM with async support
- **Redis Caching**: Efficient caching for improved performance
- **AI Model Integration**: Support for OpenAI and Mistral AI models
- **Data Visualization**: Generate visualizations and dashboards from data
- **WebSocket Support**: Real-time communication for chat

## Getting Started

### Prerequisites

- Python 3.9+
- Redis server
- PostgreSQL (optional, SQLite is used by default)

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp ../.env.example .env
# Edit .env with your configuration
```

4. Start the server:
```bash
uvicorn main:app --reload
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
backend/
├── api/
│   ├── db/
│   │   ├── database.py - Database connection and session management
│   │   └── redis_client.py - Redis connection and utilities
│   ├── models/
│   │   ├── chat.py - Pydantic models for chat functionality
│   │   └── visualization.py - Models for visualization features
│   ├── routes/
│   │   ├── chat.py - Chat API endpoints
│   │   ├── file.py - File management endpoints
│   │   └── visualization.py - Visualization endpoints
│   └── services/
│       ├── chat_service.py - Chat generation logic
│       ├── file_service.py - File handling utilities
│       └── visualization_service.py - Visualization generation
├── uploads/ - Directory for uploaded files and visualizations
├── main.py - Application entry point
└── requirements.txt - Project dependencies
```

## Environment Variables

The backend uses the following environment variables:

```
# Database
DATABASE_URI=sqlite:///app.db

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key

# App Settings
UPLOAD_FOLDER=./uploads
DEBUG=True
PORT=8000
HOST=0.0.0.0

# Security
SECRET_KEY=your_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## API Endpoints

### Chat Endpoints

- `POST /api/chat/chat`: Send a message to the chatbot
- `WebSocket /api/chat/ws/chat/{user_id}`: Real-time chat with streaming responses
- `POST /api/chat/simple-chat`: Simple chat endpoint without authentication

### File Endpoints

- `POST /api/file/upload`: Upload a file
- `GET /api/file/files`: List all uploaded files
- `GET /api/file/files/{file_path}`: Get information about a file
- `DELETE /api/file/files/{file_path}`: Delete a file
- `GET /api/file/download/{file_path}`: Download a file

### Visualization Endpoints

- `POST /api/visualization/visualize`: Generate a visualization
- `POST /api/visualization/dashboard`: Generate a dashboard
- `GET /api/visualization/visualizations/{filename}`: Get a visualization image 