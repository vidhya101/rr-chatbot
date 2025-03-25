#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command existence
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is not installed"
        return 1
    fi
}

# Function to check Python version
check_python_version() {
    if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" &> /dev/null; then
        echo -e "${GREEN}✓${NC} Python version is 3.8 or higher"
        return 0
    else
        echo -e "${RED}✗${NC} Python version must be 3.8 or higher"
        return 1
    fi
}

# Function to check Node.js version
check_node_version() {
    if node -v | grep -q "v1[4-9]\|v[2-9][0-9]"; then
        echo -e "${GREEN}✓${NC} Node.js version is 14 or higher"
        return 0
    else
        echo -e "${RED}✗${NC} Node.js version must be 14 or higher"
        return 1
    fi
}

# Function to check Redis
check_redis() {
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓${NC} Redis is running"
        return 0
    else
        echo -e "${RED}✗${NC} Redis is not running"
        return 1
    fi
}

# Function to check ports
check_port() {
    if ! lsof -i :$1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} Port $1 is available"
        return 0
    else
        echo -e "${RED}✗${NC} Port $1 is in use"
        return 1
    fi
}

# Function to check directory existence
check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Directory $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} Directory $1 is missing"
        return 1
    fi
}

# Function to check file existence
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} File $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} File $1 is missing"
        return 1
    fi
}

echo "Checking environment and dependencies..."
echo "======================================="

# Check basic requirements
echo -e "\nChecking basic requirements:"
check_command "python"
check_command "node"
check_command "npm"
check_command "redis-server"

# Check versions
echo -e "\nChecking versions:"
check_python_version
check_node_version

# Check services
echo -e "\nChecking services:"
check_redis

# Check ports
echo -e "\nChecking ports:"
check_port 3000
check_port 5000

# Check backend setup
echo -e "\nChecking backend setup:"
check_directory "backend/venv"
check_file "backend/requirements.txt"
check_file "backend/.env"
check_directory "backend/uploads"
check_directory "backend/static/visualizations"
check_directory "backend/static/dashboards"

# Check frontend setup
echo -e "\nChecking frontend setup:"
check_directory "frontend/node_modules"
check_file "frontend/package.json"
check_file "frontend/.env"

# Final summary
echo -e "\nEnvironment check complete!"
echo "If any checks failed, please run setup.sh to fix the issues."
echo "If all checks passed, you can run start.sh to start the servers." 