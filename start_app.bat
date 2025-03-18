@echo off
echo Starting RR-Chatbot with Ollama, Hugging Face, and Mistral AI integration...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is not installed. Please install Node.js 14 or higher.
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo npm is not installed. Please install npm.
    exit /b 1
)

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Ollama is not running. Please start Ollama before using local models.
    echo You can still use Hugging Face and Mistral AI models.
)

REM Create Python virtual environment if it doesn't exist
if not exist backend\venv (
    echo Creating Python virtual environment...
    cd backend
    python -m venv venv
    cd ..
)

REM Activate virtual environment and install backend dependencies
echo Installing backend dependencies...
cd backend
call venv\Scripts\activate
pip install -r requirements.txt

REM Start backend server in a new window
echo Starting backend server...
start cmd /k "venv\Scripts\activate && python app.py"
cd ..

REM Install frontend dependencies if node_modules doesn't exist
if not exist frontend\node_modules (
    echo Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

REM Start frontend server
echo Starting frontend server...
cd frontend
npm start

echo.
echo RR-Chatbot is now running!
echo Backend server: http://localhost:5000
echo Frontend server: http://localhost:3000

echo.
echo RR-Chatbot is starting up!
echo.
echo Backend server will be available at: http://localhost:5000
echo Frontend application will open at: http://localhost:3000
echo.
echo Admin credentials:
echo Email: admin@example.com
echo Password: admin123
echo.
echo Press any key to exit this window...
pause > nul 