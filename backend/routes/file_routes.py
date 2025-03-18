import os
import uuid
import magic
from flask import Blueprint, request, jsonify, current_app, send_file, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from models.db import db
from models.file import File
from services.file_service import process_file, get_file_metadata
import pandas as pd
import json
from services.data_analysis_service import DataAnalyzer, analyze_dataset
from services.visualization_service import generate_visualization, generate_dashboard
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
file_bp = Blueprint('file', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'txt', 'csv', 'xlsx', 'xls', 'doc', 'docx', 'ppt', 'pptx',
    'json', 'xml', 'md', 'html', 'htm', 'jpg', 'jpeg', 'png', 'gif'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@file_bp.route('/upload', methods=['POST'])
@jwt_required(optional=True)
def upload_file():
    """Upload a file"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file extension is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get current user if authenticated
        current_user_id = get_jwt_identity()
        
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Create user directory if authenticated
        if current_user_id:
            user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user_id))
            os.makedirs(user_dir, exist_ok=True)
            file_path = os.path.join(user_dir, unique_filename)
        else:
            # For unauthenticated users, store in a temporary directory
            temp_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Return file info
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': unique_filename,
            'original_filename': filename,
            'file_path': file_path
        }), 201
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze an uploaded file"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        file_path = data.get('file_path')
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'linear')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Analyze the dataset
        analysis_results = analyze_dataset(file_path, target_column, model_type)
        
        return jsonify({
            'message': 'File analyzed successfully',
            'results': analysis_results
        }), 200
    
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/preview', methods=['POST'])
def preview_file():
    """Preview an uploaded file"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Determine file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Read file based on extension
        if file_extension == '.csv':
            df = pd.read_csv(file_path, nrows=100)  # Read only first 100 rows for preview
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=100)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
            df = df.head(100)
        elif file_extension == '.txt':
            # Try to infer delimiter
            df = pd.read_csv(file_path, sep=None, engine='python', nrows=100)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Get basic info
        columns = df.columns.tolist()
        data_types = df.dtypes.astype(str).to_dict()
        preview_data = df.head(10).to_dict(orient='records')
        
        return jsonify({
            'message': 'File preview generated',
            'columns': columns,
            'data_types': data_types,
            'preview_data': preview_data,
            'row_count': len(df),
            'column_count': len(columns)
        }), 200
    
    except Exception as e:
        logger.error(f"Error previewing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/dashboard', methods=['POST'])
def create_dashboard():
    """Create a dashboard for an analyzed file"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        file_path = data.get('file_path')
        target_column = data.get('target_column')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Create analyzer
        analyzer = DataAnalyzer(file_path=file_path)
        analyzer.load_data()
        analyzer.explore_data()
        analyzer.clean_data()
        analyzer.engineer_features(target_column=target_column)
        analyzer.analyze_data()
        analyzer.create_visualizations()
        
        if target_column and target_column in analyzer.df.columns:
            analyzer.train_model()
        
        # Generate dashboard data
        dashboard_data = analyzer.generate_dashboard_data()
        
        return jsonify({
            'message': 'Dashboard created successfully',
            'dashboard_data': dashboard_data
        }), 200
    
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/', methods=['GET'])
@jwt_required(optional=True)
def list_files():
    """List all files for the current user"""
    try:
        # Get current user if authenticated
        current_user_id = get_jwt_identity()
        
        if current_user_id:
            # Get files for authenticated user
            user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user_id))
            os.makedirs(user_dir, exist_ok=True)
            
            files = []
            for filename in os.listdir(user_dir):
                file_path = os.path.join(user_dir, filename)
                if os.path.isfile(file_path):
                    # Get file info
                    file_info = {
                        'filename': filename,
                        'original_filename': filename.split('_', 1)[1] if '_' in filename else filename,
                        'file_path': file_path,
                        'size': os.path.getsize(file_path),
                        'created_at': os.path.getctime(file_path)
                    }
                    files.append(file_info)
            
            return jsonify({'files': files}), 200
        else:
            # For unauthenticated users, return empty list
            return jsonify({'files': []}), 200
    
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/<filename>', methods=['GET'])
@jwt_required(optional=True)
def download_file(filename):
    """Download a file"""
    try:
        # Get current user if authenticated
        current_user_id = get_jwt_identity()
        
        if current_user_id:
            # Get file for authenticated user
            user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user_id))
            return send_from_directory(user_dir, filename)
        else:
            # For unauthenticated users, check temp directory
            temp_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp')
            return send_from_directory(temp_dir, filename)
    
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/<filename>', methods=['DELETE'])
@jwt_required(optional=True)
def delete_file(filename):
    """Delete a file"""
    try:
        # Get current user if authenticated
        current_user_id = get_jwt_identity()
        
        if current_user_id:
            # Delete file for authenticated user
            user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user_id))
            file_path = os.path.join(user_dir, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return jsonify({'message': 'File deleted successfully'}), 200
            else:
                return jsonify({'error': 'File not found'}), 404
        else:
            # For unauthenticated users, check temp directory
            temp_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp')
            file_path = os.path.join(temp_dir, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return jsonify({'message': 'File deleted successfully'}), 200
            else:
                return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/visualize', methods=['POST'])
def visualize_file():
    """Generate visualizations for an uploaded file"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400
        
        file_path = data.get('file_path')
        viz_type = data.get('type', 'auto')
        params = data.get('params', {})
        
        if not file_path:
            return jsonify({'error': 'No file path provided', 'success': False}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found', 'success': False}), 404
        
        # Generate visualization
        result = generate_visualization(file_path, viz_type, params)
        
        return jsonify(result), 200 if result.get('success', False) else 500
    
    except Exception as e:
        logger.error(f"Error visualizing file: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@file_bp.route('/generate-dashboard', methods=['POST'])
def generate_file_dashboard():
    """Generate a comprehensive dashboard for an uploaded file"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400
        
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'No file path provided', 'success': False}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found', 'success': False}), 404
        
        # Generate dashboard
        result = generate_dashboard(file_path)
        
        return jsonify(result), 200 if result.get('success', False) else 500
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@file_bp.route('/visualizations/<filename>', methods=['GET'])
def get_visualization(filename):
    """Get a visualization image"""
    try:
        # Construct file path
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'visualizations', filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': f"Visualization not found: {filename}", 'success': False}), 404
        
        # Return the file
        return send_file(file_path, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Error retrieving visualization: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500 