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
    'scikit-learn'
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
from routes.model_routes import model_routes
from routes.file_routes import file_bp
from routes.user_routes import user_bp
from routes.dashboard_routes import dashboard_bp
from routes.feedback_routes import feedback_routes
from routes.visualization_routes import visualization_bp

# Import database configuration
from models.db import db, init_db

# Import services
from utils.db_utils import init_db as init_db_utils, start_maintenance_task

# Create Flask app
app = Flask(__name__)

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Enable CORS
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(chat_bp, url_prefix='/api')
app.register_blueprint(model_routes, url_prefix='/api/models')
app.register_blueprint(file_bp, url_prefix='/api/files')
app.register_blueprint(user_bp, url_prefix='/api/users')
app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
app.register_blueprint(feedback_routes, url_prefix='/api/feedback')
app.register_blueprint(visualization_bp, url_prefix='/api/visualization')

# Initialize database
init_db(app)

# Start database maintenance task
start_maintenance_task()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - app.start_time
    })

# Store app start time
app.start_time = time.time()

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

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