import os
import sys
import subprocess
import importlib.util
from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
from flask_socketio import SocketIO
from datetime import timedelta
import logging
from logging.handlers import RotatingFileHandler
import uuid
import time
from datetime import datetime
import requests
from requests.exceptions import RequestException
from services.data_analysis_service import data_analysis_bp
from services.visualization_service import visualization_bp
from routes.file_routes import file_bp
from routes.model_routes import model_bp
from services.model_service import model_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Check for required packages and install if missing
required_packages = [
    'flask_jwt_extended',
    'mistralai',
    'huggingface_hub',
    'tiktoken',
    'requests',
    'numpy',
    'flask_socketio',
    'flask_sqlalchemy',
    'flask_migrate',
    'flask_cors',
    'python-dotenv',
    'pandas',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'flask_caching',
    'marshmallow',
    'shap',
    'lime',
    'plotly',
    'fastapi',
    'uvicorn'
]

def check_and_install_packages():
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("All required packages installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            print("Please install the required packages manually:")
            print(f"pip install {' '.join(missing_packages)}")

# Check for required packages before importing them
check_and_install_packages()

# Import routes
from routes.auth_routes import auth_bp
from routes.chat_routes import chat_bp
from routes.model_routes import model_bp
from routes.file_routes import file_bp
from routes.user_routes import user_bp
from routes.dashboard_routes import dashboard_bp
from routes.feedback_routes import feedback_routes

# Import database configuration
from models.db import db, init_db

# Import services
from utils.db_utils import init_db as init_db_utils, start_maintenance_task

# Import enhanced data visualization API
from data_visualization_api import init_app as init_data_viz_api

# Create Flask app
app = Flask(__name__)

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['API_VERSION'] = '1.1.0'  # Add API version

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', app.config['SECRET_KEY'])
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Enable CORS
CORS(app)

# Initialize JWT
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(chat_bp, url_prefix='/api')
app.register_blueprint(model_bp, url_prefix='/api/model')
app.register_blueprint(file_bp, url_prefix='/api/files')
app.register_blueprint(user_bp, url_prefix='/api/users')
app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
app.register_blueprint(feedback_routes, url_prefix='/api/feedback')
app.register_blueprint(data_analysis_bp, url_prefix='/api/data')
app.register_blueprint(visualization_bp, url_prefix='/api/visualization')

# Initialize database
init_db(app)

# Initialize enhanced data visualization API
init_data_viz_api(app)

# Start database maintenance task
start_maintenance_task()

def check_ollama_health(max_retries=3, timeout=5):
    """Check Ollama health with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=timeout
            )
            if response.status_code == 200:
                logger.info("Ollama health check response: 200")
                return True
            else:
                logger.warning(f"Ollama health check failed with status code: {response.status_code}")
        except RequestException as e:
            logger.warning(f"Ollama health check attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    logger.error("All Ollama health check attempts failed")
    return False

# Check Ollama health
if not check_ollama_health():
    logger.warning("Ollama is not running or not accessible. Some features may be limited.")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with system metrics"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"

    return jsonify({
        "status": "ok" if db_status == "connected" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - app.start_time,
        "version": app.config.get('API_VERSION', '1.0.0'),
        "components": {
            "database": db_status,
            "api": "ok"
        },
        "environment": os.environ.get('FLASK_ENV', 'production')
    })

# Store app start time
app.start_time = time.time()

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    logger.error(f"Bad request: {error}")
    return jsonify({
        "error": "Bad request",
        "message": str(error),
        "status_code": 400
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    logger.error(f"Unauthorized access: {error}")
    return jsonify({
        "error": "Unauthorized",
        "message": "Authentication required",
        "status_code": 401
    }), 401

@app.errorhandler(403)
def forbidden(error):
    logger.error(f"Forbidden access: {error}")
    return jsonify({
        "error": "Forbidden",
        "message": "You don't have permission to access this resource",
        "status_code": 403
    }), 403

@app.errorhandler(404)
def not_found(error):
    logger.error(f"Resource not found: {error}")
    return jsonify({
        "error": "Not found",
        "message": "The requested resource was not found",
        "status_code": 404
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.error(f"Method not allowed: {error}")
    return jsonify({
        "error": "Method not allowed",
        "message": "The method is not allowed for the requested URL",
        "status_code": 405
    }), 405

@app.errorhandler(429)
def too_many_requests(error):
    logger.error(f"Too many requests: {error}")
    return jsonify({
        "error": "Too many requests",
        "message": "Rate limit exceeded",
        "status_code": 429
    }), 429

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }), 500

@app.route('/')
def index():
    return "API is running"

# Run the app
if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 