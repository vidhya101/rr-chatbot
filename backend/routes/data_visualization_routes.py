from flask import Blueprint, request, jsonify, current_app, send_file
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import json
import os
from werkzeug.utils import secure_filename
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
data_viz_bp = Blueprint('data_visualization', __name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@data_viz_bp.route('/data/test', methods=['GET'])
def test_route():
    """Test route to verify the server is working"""
    logger.info("Test route called")
    return jsonify({
        'success': True,
        'message': 'Data visualization API is working',
        'timestamp': datetime.utcnow().isoformat()
    })

@data_viz_bp.route('/data/upload', methods=['POST'])
def upload_file():
    """Upload a data file for visualization"""
    logger.info("Upload endpoint called")
    try:
        if 'file' not in request.files:
            logger.error("No file in request.files")
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        logger.info(f"Received file: {file.filename}")
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Secure the filename and generate unique ID
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_ext}"
        
        # Create upload directory if it doesn't exist
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
            
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        # Read and store file metadata
        try:
            # More robust CSV parsing with error handling
            if file_ext == 'csv':
                try:
                    # First attempt with standard parsing
                    df = pd.read_csv(file_path)
                except Exception as csv_err:
                    logger.info(f"Standard CSV parsing failed, trying with flexible options: {str(csv_err)}")
                    try:
                        # Second attempt with more flexible options
                        df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True, 
                                        on_bad_lines='skip', sep=None, engine='python')
                    except Exception as flex_err:
                        # Last attempt with very liberal options
                        logger.warning(f"Flexible CSV parsing failed, trying with minimal options: {str(flex_err)}")
                        df = pd.read_csv(file_path, sep=None, header=None, engine='python')
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            elif file_ext == 'json':
                df = pd.read_json(file_path)
            elif file_ext == 'parquet':
                df = pd.read_parquet(file_path)
            
            metadata = {
                'id': file_id,
                'name': filename,  # Match the property name expected by frontend
                'original_name': filename,
                'path': file_path,
                'type': file_ext,
                'size': os.path.getsize(file_path),  # Add file size for frontend
                'columns': list(df.columns),
                'rows': len(df),
                'uploaded_at': datetime.utcnow().isoformat()  # Match the property name expected by frontend
            }
            
            # Store metadata in Redis or database in production
            current_app.uploaded_files = getattr(current_app, 'uploaded_files', {})
            current_app.uploaded_files[file_id] = metadata
            
            logger.info(f"File uploaded successfully: {file_id}")
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'file': metadata
            })
            
        except Exception as e:
            # Clean up the file if metadata extraction fails
            logger.error(f"Error processing file metadata: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(f"Error processing file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_viz_bp.route('/data/files', methods=['GET'])
def get_uploaded_files():
    """Get list of uploaded files"""
    try:
        current_app.uploaded_files = getattr(current_app, 'uploaded_files', {})
        return jsonify({
            'success': True,
            'files': list(current_app.uploaded_files.values())
        })
    except Exception as e:
        logger.error(f"Error getting uploaded files: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_viz_bp.route('/data/visualize', methods=['POST'])
def visualize_data():
    """Generate visualization for uploaded file"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        file_id = data.get('fileId')
        viz_type = data.get('type', 'auto')
        columns = data.get('columns', {})
        
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'No file ID provided'
            }), 400
        
        current_app.uploaded_files = getattr(current_app, 'uploaded_files', {})
        file_info = current_app.uploaded_files.get(file_id)
        
        if not file_info:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Read the data file
        file_path = file_info['path']
        file_ext = file_info['type']
        
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)
        elif file_ext == 'parquet':
            df = pd.read_parquet(file_path)
        
        # Generate visualization
        fig = None
        if viz_type == 'auto':
            # Automatically determine the best visualization
            if len(df.select_dtypes(include=['number']).columns) >= 2:
                fig = px.scatter(df, x=df.select_dtypes(include=['number']).columns[0],
                               y=df.select_dtypes(include=['number']).columns[1])
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else None)
        elif viz_type == 'scatter':
            fig = px.scatter(df, x=columns.get('x'), y=columns.get('y'))
        elif viz_type == 'line':
            fig = px.line(df, x=columns.get('x'), y=columns.get('y'))
        elif viz_type == 'bar':
            fig = px.bar(df, x=columns.get('x'), y=columns.get('y'))
        elif viz_type == 'histogram':
            fig = px.histogram(df, x=columns.get('x'))
        elif viz_type == 'box':
            fig = px.box(df, x=columns.get('x'), y=columns.get('y'))
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid visualization type'
            }), 400
        
        # Save visualization
        viz_id = str(uuid.uuid4())
        viz_path = os.path.join('static/visualizations', f"{viz_id}.html")
        fig.write_html(viz_path)
        
        return jsonify({
            'success': True,
            'visualization': {
                'id': viz_id,
                'type': viz_type,
                'path': viz_path,
                'fileId': file_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_viz_bp.route('/data/visualization/<viz_id>', methods=['GET'])
def get_visualization(viz_id):
    """Get a specific visualization"""
    try:
        viz_path = os.path.join('static/visualizations', f"{viz_id}.html")
        if not os.path.exists(viz_path):
            return jsonify({
                'success': False,
                'error': 'Visualization not found'
            }), 404
        
        return send_file(viz_path)
        
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_viz_bp.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        if filename not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        # Read data
        df = pd.read_csv(uploaded_files[filename]['path'])
        
        # Generate summary
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'summary_stats': json.loads(df.describe().to_json()),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        return jsonify({'error': 'Failed to get data summary'}), 500

@data_viz_bp.route('/api/data/dashboard', methods=['POST'])
def create_dashboard():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        charts = data.get('charts', [])
        
        if not filename or not charts:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if filename not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        # Read data
        df = pd.read_csv(uploaded_files[filename]['path'])
        
        # Create dashboard
        dashboard_plots = []
        for chart in charts:
            chart_type = chart.get('type')
            x_column = chart.get('x_column')
            y_column = chart.get('y_column')
            
            if chart_type == 'line':
                fig = px.line(df, x=x_column, y=y_column)
            elif chart_type == 'bar':
                fig = px.bar(df, x=x_column, y=y_column)
            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_column, y=y_column)
            else:
                continue
            
            dashboard_plots.append({
                'plot': fig.to_json(),
                'type': chart_type
            })
        
        return jsonify({
            'dashboard': dashboard_plots
        })
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return jsonify({'error': 'Failed to create dashboard'}), 500 