# RR-Chatbot: Advanced AI-Powered Data Analysis and Visualization Platform

A modern, scalable platform for AI-powered chat, data analysis, and visualization built with FastAPI, React, and modern AI technologies.

## Features

- **AI-Powered Chat**: Engage with multiple AI models including OpenAI GPT and Mistral AI
- **Real-time Communication**: WebSocket support for streaming responses
- **Data Visualization**: Generate interactive visualizations and dashboards from uploaded data
- **File Management**: Upload, analyze, and manage various file formats
- **Caching**: Redis-based caching for improved performance
- **Asynchronous Processing**: Non-blocking operations for better scalability
- **Modern Architecture**: FastAPI backend with React frontend

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **Redis**: In-memory data store for caching and pub/sub
- **Pandas/NumPy**: Data processing and analysis
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **OpenAI/Mistral AI**: AI model integration

### Frontend (Planned)
- **React**: UI library for building interactive interfaces
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React Query**: Data fetching and state management
- **Socket.IO**: Real-time communication

## Getting Started

### Prerequisites
- Python 3.9+
- Redis server
- Node.js 16+ (for frontend)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rr-chatbot.git
cd rr-chatbot
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

5. (Optional) Start Redis server:
```bash
redis-server
```

### Environment Variables

Create a `.env` file in the project root with the following variables:

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
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
rr-chatbot/
├── backend/
│   ├── api/
│   │   ├── db/
│   │   │   ├── database.py
│   │   │   └── redis_client.py
│   │   ├── models/
│   │   │   ├── chat.py
│   │   │   └── visualization.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── file.py
│   │   │   └── visualization.py
│   │   └── services/
│   │       ├── chat_service.py
│   │       ├── file_service.py
│   │       └── visualization_service.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
├── .env
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Mistral AI for their models
- FastAPI team for the excellent framework
- All open-source libraries used in this project 