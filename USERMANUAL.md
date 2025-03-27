# RR-Chatbot User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [Features and Functionality](#features-and-functionality)
8. [API Endpoints](#api-endpoints)
9. [Troubleshooting](#troubleshooting)

## Introduction
RR-Chatbot is a sophisticated chatbot application that combines modern web technologies with advanced AI capabilities. The application features a React-based frontend and a Flask-based backend, utilizing various AI models including Mistral AI for natural language processing.

## Project Structure
```
rr-chatbot/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── chat.py          # Chat-related endpoints
│   │   └── models.py        # Model management endpoints
│   ├── blueprints/
│   │   ├── __init__.py
│   │   ├── admin.py         # Admin panel routes
│   │   ├── auth.py          # Authentication routes
│   │   └── chat.py          # Chat interface routes
│   ├── config/
│   │   ├── __init__.py
│   │   ├── development.py   # Development settings
│   │   ├── production.py    # Production settings
│   │   └── testing.py       # Testing settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── logging.py       # Logging configuration
│   │   ├── security.py      # Security utilities
│   │   └── validation.py    # Input validation
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models/          # Database models
│   │   │   ├── user.py
│   │   │   ├── chat.py
│   │   │   └── model.py
│   │   └── migrations/      # Database migrations
│   ├── logs/
│   │   ├── app.log         # Application logs
│   │   ├── error.log       # Error logs
│   │   └── access.log      # Access logs
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ai/             # AI model implementations
│   │   │   ├── mistral.py
│   │   │   └── gpt.py
│   │   └── training/       # Model training scripts
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat_routes.py  # Chat-related routes
│   │   ├── auth_routes.py  # Authentication routes
│   │   └── admin_routes.py # Admin panel routes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_service.py   # AI service implementation
│   │   ├── chat_service.py # Chat service implementation
│   │   └── user_service.py # User service implementation
│   ├── static/
│   │   ├── css/           # CSS stylesheets
│   │   ├── js/            # JavaScript files
│   │   └── images/        # Image assets
│   ├── templates/
│   │   ├── base.html      # Base template
│   │   ├── chat.html      # Chat interface
│   │   └── admin.html     # Admin panel
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_api.py    # API tests
│   │   ├── test_models.py # Model tests
│   │   └── test_services.py # Service tests
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── ai_utils.py    # AI utilities
│   │   ├── db_utils.py    # Database utilities
│   │   └── file_utils.py  # File handling utilities
│   ├── app.py             # Main application file
│   ├── config.py          # Configuration settings
│   ├── database.py        # Database connection
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── public/
│   │   ├── index.html     # Main HTML file
│   │   ├── favicon.ico    # Favicon
│   │   └── assets/        # Public assets
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── Chat/
│   │   │   │   ├── ChatWindow.js
│   │   │   │   ├── MessageList.js
│   │   │   │   └── MessageInput.js
│   │   │   ├── Auth/
│   │   │   │   ├── Login.js
│   │   │   │   └── Register.js
│   │   │   └── Admin/
│   │   │       ├── Dashboard.js
│   │   │       └── UserManagement.js
│   │   ├── services/      # API services
│   │   │   ├── api.js
│   │   │   ├── auth.js
│   │   │   └── chat.js
│   │   ├── utils/         # Utility functions
│   │   │   ├── helpers.js
│   │   │   └── validation.js
│   │   ├── styles/        # CSS styles
│   │   │   ├── main.css
│   │   │   └── components/
│   │   ├── context/       # React context
│   │   │   ├── AuthContext.js
│   │   │   └── ChatContext.js
│   │   ├── hooks/         # Custom hooks
│   │   │   ├── useAuth.js
│   │   │   └── useChat.js
│   │   ├── App.js         # Main App component
│   │   ├── index.js       # Entry point
│   │   └── routes.js      # Route definitions
│   ├── package.json       # Node.js dependencies
│   └── .env              # Frontend environment variables
├── venv_new/             # Python virtual environment
├── docs/                 # Documentation
│   ├── api/             # API documentation
│   ├── deployment/      # Deployment guides
│   └── development/     # Development guides
├── scripts/             # Utility scripts
│   ├── setup.sh        # Setup script
│   ├── deploy.sh       # Deployment script
│   └── backup.sh       # Backup script
├── uploads/            # File uploads directory
├── instance/          # Instance-specific files
├── .gitignore        # Git ignore rules
├── README.md         # Project documentation
├── ADMIN_GUIDE.md    # Admin guide
├── GETTING_STARTED.md # Getting started guide
└── ENHANCEMENTS.md   # Planned enhancements
```

### Directory Structure Details

#### Backend Structure
1. **api/** - API endpoints and routes
   - Contains all REST API endpoints
   - Organized by functionality (auth, chat, models)
   - Handles request/response logic

2. **blueprints/** - Flask blueprints
   - Modular route organization
   - Separates different application features
   - Includes admin, auth, and chat blueprints

3. **config/** - Configuration files
   - Environment-specific settings
   - Development, production, and testing configs
   - Centralized configuration management

4. **core/** - Core functionality
   - Essential application components
   - Logging, security, and validation
   - Shared utilities and helpers

5. **database/** - Database related files
   - SQLAlchemy models
   - Database migrations
   - Model relationships and schemas

6. **logs/** - Application logs
   - Different log types (app, error, access)
   - Log rotation and management
   - Debugging and monitoring

7. **models/** - AI models
   - AI model implementations
   - Model training scripts
   - Model management utilities

8. **routes/** - Route handlers
   - URL routing logic
   - Request handling
   - Response formatting

9. **services/** - Business logic
   - Service layer implementation
   - Business rules and logic
   - External service integration

10. **static/** - Static files
    - CSS, JavaScript, and images
    - Client-side assets
    - Static resource management

11. **templates/** - HTML templates
    - Jinja2 templates
    - Page layouts
    - Dynamic content rendering

12. **tests/** - Test files
    - Unit tests
    - Integration tests
    - Test utilities and helpers

13. **utils/** - Utility functions
    - Helper functions
    - Common utilities
    - Shared tools

#### Frontend Structure
1. **public/** - Public assets
   - Static files served directly
   - Index.html and favicon
   - Public resources

2. **src/** - Source code
   - React components
   - Application logic
   - State management
   - Routing

3. **components/** - React components
   - Reusable UI components
   - Feature-specific components
   - Layout components

4. **services/** - API services
   - API integration
   - Data fetching
   - State updates

5. **utils/** - Utility functions
   - Helper functions
   - Common utilities
   - Shared tools

6. **styles/** - CSS styles
   - Global styles
   - Component styles
   - Theme definitions

7. **context/** - React context
   - Global state management
   - Shared state
   - Context providers

8. **hooks/** - Custom hooks
   - Reusable logic
   - State management
   - Side effects

#### Supporting Directories
1. **docs/** - Documentation
   - API documentation
   - Deployment guides
   - Development guides

2. **scripts/** - Utility scripts
   - Setup scripts
   - Deployment scripts
   - Maintenance scripts

3. **uploads/** - File uploads
   - User uploaded files
   - Temporary storage
   - File processing

4. **instance/** - Instance files
   - Instance-specific config
   - Local settings
   - Development data

## System Requirements
- Python 3.8 or higher
- Node.js 14.0 or higher
- Redis server
- PostgreSQL database
- Git

## Installation

### Backend Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv_new
source venv_new/bin/activate  # On Windows: venv_new\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Frontend Setup
1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

### Backend Configuration
Key configuration files:
- `backend/.env`: Environment variables
- `backend/config.py`: Application configuration
- `backend/database.py`: Database settings

### Frontend Configuration
Key configuration files:
- `frontend/.env`: Environment variables
- `frontend/package.json`: Dependencies and scripts

## Running the Application

### Backend Server
```bash
cd backend
python app.py
```
The backend server will start on http://localhost:5000

### Frontend Server
```bash
cd frontend
npm start
```
The frontend server will start on http://localhost:3000

## Features and Functionality

### Core Features
1. Chat Interface
   - Real-time messaging
   - Support for multiple AI models
   - Message history
   - File upload capabilities

2. AI Integration
   - Mistral AI integration
   - Model selection
   - Response streaming
   - Context management

3. User Management
   - Authentication
   - User profiles
   - Session management

4. Data Management
   - Chat history storage
   - File storage
   - Database integration

### Technical Features
1. Real-time Communication
   - WebSocket support
   - Socket.IO integration
   - Event handling

2. Security
   - JWT authentication
   - CORS protection
   - Input validation

3. Performance
   - Redis caching
   - Database optimization
   - Response compression

## API Endpoints

### Chat Endpoints
- `GET /models`: Get available AI models
- `POST /simple-chat`: Send a message to the chatbot
- `POST /public/chat`: Public chat endpoint

### Authentication Endpoints
- `POST /auth/login`: User login
- `POST /auth/register`: User registration
- `POST /auth/logout`: User logout

## Dependencies

### Backend Dependencies
Core dependencies include:
- Flask and Flask extensions
- SQLAlchemy for database
- Redis for caching
- Socket.IO for real-time communication
- AI/ML libraries (TensorFlow, PyTorch, etc.)
- Data processing libraries
- Security libraries

### Frontend Dependencies
Core dependencies include:
- React and React Router
- Material-UI components
- Chart.js for visualization
- Axios for API calls
- Socket.IO client
- Testing libraries

## Troubleshooting

### Common Issues
1. Backend Connection Issues
   - Check Redis server status
   - Verify database connection
   - Check environment variables

2. Frontend Issues
   - Clear browser cache
   - Check API endpoint configuration
   - Verify environment variables

3. AI Model Issues
   - Check API key configuration
   - Verify model availability
   - Check rate limits

### Logging
- Backend logs: `backend/logs/app.log`
- Frontend logs: Browser console

## Support
For additional support:
1. Check the documentation in the `docs/` directory
2. Review the `ADMIN_GUIDE.md` for administrative tasks
3. Check `GETTING_STARTED.md` for setup instructions
4. Review `ENHANCEMENTS.md` for planned improvements

## Security Considerations
1. Keep API keys secure
2. Regularly update dependencies
3. Monitor logs for suspicious activity
4. Follow security best practices for deployment

## Performance Optimization
1. Use Redis caching effectively
2. Optimize database queries
3. Implement proper indexing
4. Monitor resource usage

## Maintenance
1. Regular dependency updates
2. Database backups
3. Log rotation
4. System health monitoring 