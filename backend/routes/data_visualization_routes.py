from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
data_viz_bp = Blueprint('data_visualization', __name__)

# Store uploaded files and their metadata
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

uploaded_files = {}

@data_viz_bp.route('/api/data/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # Read and store file metadata
        df = pd.read_csv(filename)
        uploaded_files[file.filename] = {
            'path': filename,
            'columns': list(df.columns),
            'rows': len(df),
            'upload_time': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'metadata': uploaded_files[file.filename]
        })
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': 'Failed to upload file'}), 500

@data_viz_bp.route('/api/data/files', methods=['GET'])
def get_uploaded_files():
    try:
        return jsonify({
            'files': uploaded_files
        })
    except Exception as e:
        logger.error(f"Error getting uploaded files: {str(e)}")
        return jsonify({'error': 'Failed to get uploaded files'}), 500

@data_viz_bp.route('/api/data/visualize', methods=['POST'])
def create_visualization():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        chart_type = data.get('chart_type')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        
        if not all([filename, chart_type, x_column, y_column]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if filename not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        # Read data
        df = pd.read_csv(uploaded_files[filename]['path'])
        
        # Create visualization
        if chart_type == 'line':
            fig = px.line(df, x=x_column, y=y_column)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_column, y=y_column)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column)
        else:
            return jsonify({'error': 'Unsupported chart type'}), 400
        
        # Convert to JSON
        plot_json = fig.to_json()
        
        return jsonify({
            'plot': plot_json,
            'type': chart_type
        })
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return jsonify({'error': 'Failed to create visualization'}), 500

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