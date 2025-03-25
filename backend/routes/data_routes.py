from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from datetime import datetime
import uuid
import magic
import logging

data_bp = Blueprint('data', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for files (replace with database in production)
files = {}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'pdf', 'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(file_path):
    """Detect file type using python-magic"""
    mime = magic.Magic()
    file_type = mime.from_file(file_path)
    return file_type

def get_file_preview(file_path, file_type):
    """Generate a preview of the file contents"""
    try:
        if file_type.endswith('csv'):
            df = pd.read_csv(file_path)
            return df.head().to_dict()
        elif file_type.endswith('json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {'data': data}
        elif file_type.endswith(('xlsx', 'xls')):
            df = pd.read_excel(file_path)
            return df.head().to_dict()
        elif file_type.endswith('txt'):
            with open(file_path, 'r') as f:
                return {'content': f.read(1000)}  # First 1000 characters
        else:
            return {'preview': 'Preview not available for this file type'}
    except Exception as e:
        logger.error(f'Error generating preview for {file_path}: {str(e)}')
        return {'error': 'Could not generate preview'}

@data_bp.route('/api/data/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            ext = filename.rsplit('.', 1)[1].lower()
            unique_filename = f'{file_id}.{ext}'
            
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            # Get file type and size
            file_type = get_file_type(file_path)
            file_size = os.path.getsize(file_path)
            
            # Generate preview
            preview = get_file_preview(file_path, file_type)
            
            # Store file metadata
            file_data = {
                'id': file_id,
                'name': filename,
                'type': ext,
                'mime_type': file_type,
                'size': file_size,
                'preview': preview,
                'uploaded_at': datetime.now().isoformat(),
                'path': file_path
            }
            files[file_id] = file_data
            
            return jsonify({
                'file': {
                    'id': file_id,
                    'name': filename,
                    'type': ext,
                    'size': file_size,
                    'uploaded_at': file_data['uploaded_at']
                }
            }), 201
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        logger.error(f'Error uploading file: {str(e)}')
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/data/files', methods=['GET'])
def get_files():
    """Get list of uploaded files"""
    try:
        file_list = []
        for file_id, file_data in files.items():
            file_list.append({
                'id': file_id,
                'name': file_data['name'],
                'type': file_data['type'],
                'size': file_data['size'],
                'uploaded_at': file_data['uploaded_at']
            })
        return jsonify({'files': file_list})
    except Exception as e:
        logger.error(f'Error getting files: {str(e)}')
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/data/files/<file_id>', methods=['GET'])
def get_file(file_id):
    """Get file details"""
    try:
        file_data = files.get(file_id)
        if not file_data:
            return jsonify({'error': 'File not found'}), 404
            
        return jsonify({
            'file': {
                'id': file_id,
                'name': file_data['name'],
                'type': file_data['type'],
                'mime_type': file_data['mime_type'],
                'size': file_data['size'],
                'preview': file_data['preview'],
                'uploaded_at': file_data['uploaded_at']
            }
        })
    except Exception as e:
        logger.error(f'Error getting file {file_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/data/files/<file_id>/download', methods=['GET'])
def download_file(file_id):
    """Download a file"""
    try:
        file_data = files.get(file_id)
        if not file_data:
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            file_data['path'],
            as_attachment=True,
            download_name=file_data['name']
        )
    except Exception as e:
        logger.error(f'Error downloading file {file_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/data/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete a file"""
    try:
        file_data = files.get(file_id)
        if not file_data:
            return jsonify({'error': 'File not found'}), 404
            
        # Delete the physical file
        if os.path.exists(file_data['path']):
            os.remove(file_data['path'])
            
        # Remove from storage
        del files[file_id]
        
        return jsonify({'message': 'File deleted successfully'})
    except Exception as e:
        logger.error(f'Error deleting file {file_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/data/files/<file_id>/preview', methods=['GET'])
def get_file_preview_route(file_id):
    """Get file preview"""
    try:
        file_data = files.get(file_id)
        if not file_data:
            return jsonify({'error': 'File not found'}), 404
            
        return jsonify({'preview': file_data['preview']})
    except Exception as e:
        logger.error(f'Error getting preview for file {file_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 