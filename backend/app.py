from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)
os.makedirs('static/dashboards', exist_ok=True)

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
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True) 