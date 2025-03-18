#!/bin/bash

echo "Starting RR-Chatbot with Ollama, Hugging Face, and Mistral AI integration..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js 14 or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install npm."
    exit 1
fi

# Check if Ollama is running
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "Warning: Ollama is not running. Please start Ollama before using local models."
    echo "You can still use Hugging Face and Mistral AI models."
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "Creating Python virtual environment..."
    cd backend
    python -m venv venv
    cd ..
fi

# Activate virtual environment and install backend dependencies
echo "Installing backend dependencies..."
cd backend
source venv/bin/activate || source venv/Scripts/activate
pip install -r requirements.txt

# Start backend server in the background
echo "Starting backend server..."
python app.py &
BACKEND_PID=$!
cd ..

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend server
echo "Starting frontend server..."
cd frontend
npm start

# When frontend is closed, kill the backend server
kill $BACKEND_PID

echo ""
echo "RR-Chatbot is starting up!"
echo ""
echo "Backend server will be available at: http://localhost:5000"
echo "Frontend application will open at: http://localhost:3000"
echo ""
echo "Admin credentials:"
echo "Email: admin@example.com"
echo "Password: admin123"
echo ""
echo "Press Ctrl+C to exit this script..."
wait 