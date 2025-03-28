"""
Main application entry point for the RR-Chatbot backend.
This module initializes the Flask application with all necessary configurations,
extensions, and blueprints. It also sets up logging, database connections,
and error handling.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from dotenv import load_dotenv
import redis
import logging
from logging.handlers import RotatingFileHandler
import traceback
from routes.chat_routes import chat_bp
from routes.auth_routes import auth_bp
from routes.user_routes import user_bp
from models.db import db
import tempfile
from utils.db_utils import get_db_connection

# Load environment variables
load_dotenv()

def setup_logging():
    """Configure logging with rotation and proper formatting."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure file handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10240000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def create_app():
    """Create and configure the Flask application."""
    logger = setup_logging()
    logger.info("Starting RR-Chatbot Backend...")
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    logger.info("Flask app initialized with CORS")
    
    # Configure app
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour

    # Configure file upload settings
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['TEMP_FOLDER'] = tempfile.gettempdir()
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    app.register_blueprint(chat_bp)  # Remove /api prefix to allow /public/chat
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(user_bp, url_prefix='/api/users')

    # Import and register data visualization routes
    from routes.data_visualization_routes import data_viz_bp
    app.register_blueprint(data_viz_bp, url_prefix='/api')  # Routes now have /data prefix

    # Import and register model routes
    from routes.model_routes import model_routes
    app.register_blueprint(model_routes, url_prefix='/api')
    
    # Import and register data modeling routes
    from routes.data_modeling_routes import data_modeling_routes
    app.register_blueprint(data_modeling_routes, url_prefix='/api')
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Ensure logs table exists (using raw SQL if needed)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create logs table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Logs table initialized")
        except Exception as e:
            logger.error(f"Error creating logs table: {str(e)}")
    
    # Configure Redis
    app.config['REDIS_HOST'] = os.getenv('REDIS_HOST', 'localhost')
    app.config['REDIS_PORT'] = int(os.getenv('REDIS_PORT', 6379))
    app.config['REDIS_DB'] = int(os.getenv('REDIS_DB', 0))
    logger.info(f"Redis config: {app.config['REDIS_HOST']}:{app.config['REDIS_PORT']}/{app.config['REDIS_DB']}")
    
    # Initialize Socket.IO
    socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"], async_mode='eventlet')
    logger.info("SocketIO initialized")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/visualizations', exist_ok=True)
    os.makedirs('static/dashboards', exist_ok=True)
    logger.info("Directories created")
    
    # Initialize Redis connection
    try:
        redis_client = redis.Redis(
            host=app.config['REDIS_HOST'],
            port=app.config['REDIS_PORT'],
            db=app.config['REDIS_DB'],
            socket_timeout=5
        )
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {str(e)}")
        # Don't raise the error, allow the app to start without Redis
    
    # Error handling middleware
    @app.errorhandler(Exception)
    def handle_error(error):
        logger.error(f"Unhandled error: {str(error)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(error),
            'message': 'An unexpected error occurred',
            'success': False
        }), 500

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        # Add additional headers for debugging
        if request.method == 'OPTIONS':
            logger.info(f"Handling OPTIONS request for: {request.path}")
        
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)