from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from dotenv import load_dotenv
from routes.chat_routes import chat_bp
from routes.monitoring_routes import monitoring_bp
from routes.data_routes import data_bp
from routes.visualization_routes import visualization_bp
from services.cache_service import cache_service
import multiprocessing
import redis
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Redis
app.config['REDIS_HOST'] = os.getenv('REDIS_HOST', 'localhost')
app.config['REDIS_PORT'] = int(os.getenv('REDIS_PORT', 6379))
app.config['REDIS_DB'] = int(os.getenv('REDIS_DB', 0))

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Register blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(data_bp)
app.register_blueprint(visualization_bp)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)
os.makedirs('static/dashboards', exist_ok=True)

# Initialize services
try:
    # Initialize Redis connection
    redis_client = redis.Redis(
        host=app.config['REDIS_HOST'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_DB']
    )
    redis_client.ping()  # Test connection
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Error connecting to Redis: {str(e)}")
    logger.warning("Continuing without Redis connection")

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "name": "RR Chatbot API",
        "version": "1.0.0",
        "status": "running"
    })

if __name__ == '__main__':
    # Add freeze_support for Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Import data_processing_service here to avoid circular imports
    from services.data_processing_service import data_processing_service
    
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True) 