# RR-Chatbot Backend

This is the backend for the RR-Chatbot application, built with Flask and SQLAlchemy.

## Features

- RESTful API with Flask
- JWT authentication
- Database integration with SQLAlchemy
- File upload and processing
- OpenAI integration
- Socket.IO for real-time communication
- User management
- Dashboard and analytics

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. The API will be available at `http://localhost:5000`

## Project Structure

```
backend/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── models/                # Database models
├── routes/                # API routes
├── services/              # Business logic
├── utils/                 # Utility functions
└── requirements.txt       # Python dependencies
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login a user
- `POST /api/auth/refresh` - Refresh access token
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout a user

### Chat
- `GET /api/chat/chats` - Get all chats
- `GET /api/chat/chats/<chat_id>` - Get a specific chat
- `POST /api/chat/chats` - Create a new chat
- `PUT /api/chat/chats/<chat_id>` - Update a chat
- `DELETE /api/chat/chats/<chat_id>` - Delete a chat
- `POST /api/chat/chats/<chat_id>/messages` - Send a message
- `GET /api/chat/chats/<chat_id>/messages` - Get all messages for a chat

### Files
- `POST /api/files/upload` - Upload a file
- `GET /api/files/files` - Get all files
- `GET /api/files/files/<file_id>` - Get a specific file
- `GET /api/files/files/<file_id>/download` - Download a file
- `DELETE /api/files/files/<file_id>` - Delete a file

### User
- `GET /api/user/profile` - Get user profile
- `PUT /api/user/profile` - Update user profile
- `POST /api/user/profile/picture` - Upload profile picture
- `GET /api/user/settings` - Get user settings
- `PUT /api/user/settings` - Update user settings
- `PUT /api/user/password` - Change user password
- `GET /api/user/stats` - Get user statistics

### Dashboard
- `GET /api/dashboard/dashboards` - Get all dashboards
- `GET /api/dashboard/dashboards/<dashboard_id>` - Get a specific dashboard
- `POST /api/dashboard/dashboards` - Create a new dashboard
- `PUT /api/dashboard/dashboards/<dashboard_id>` - Update a dashboard
- `DELETE /api/dashboard/dashboards/<dashboard_id>` - Delete a dashboard
- `POST /api/dashboard/dashboards/<dashboard_id>/charts` - Create a new chart
- `PUT /api/dashboard/charts/<chart_id>` - Update a chart
- `DELETE /api/dashboard/charts/<chart_id>` - Delete a chart
- `GET /api/dashboard/stats` - Get dashboard statistics 