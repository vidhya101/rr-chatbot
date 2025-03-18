from flask import Blueprint, request, jsonify, current_app, send_file
import os
import logging
import json
from services.visualization_service import generate_visualization, generate_dashboard, suggest_chart_type, generate_plotly_visualization
from utils.db_utils import log_info, log_error
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/visualize', methods=['POST'])
def visualize():
    """Generate a visualization for a dataset"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Check for required fields
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({"success": False, "error": "No file path provided"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File not found: {file_path}"}), 404
        
        # Get visualization type and parameters
        viz_type = data.get('type', 'auto')
        params = data.get('params', {})
        
        # Log the request
        log_info(
            source='visualization',
            message=f"Visualization request for {file_path}",
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            details={
                'file_path': file_path,
                'viz_type': viz_type,
                'params': params
            }
        )
        
        # Generate visualization
        result = generate_visualization(file_path, viz_type, params)
        
        if result.get('success', False):
            return jsonify(result), 200
        else:
            log_error(
                source='visualization',
                message=f"Error generating visualization: {result.get('error', 'Unknown error')}",
                user_id=request.headers.get('X-User-ID', 'anonymous'),
                details={
                    'file_path': file_path,
                    'viz_type': viz_type,
                    'params': params
                }
            )
            return jsonify(result), 500
    
    except Exception as e:
        logger.error(f"Error in visualization route: {str(e)}")
        log_error(
            source='visualization',
            message=f"Error in visualization route: {str(e)}",
            user_id=request.headers.get('X-User-ID', 'anonymous')
        )
        return jsonify({
            "success": False,
            "error": "Failed to generate visualization",
            "message": str(e)
        }), 500

@visualization_bp.route('/dashboard', methods=['POST'])
def dashboard():
    """Generate a comprehensive dashboard for a dataset"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Check for required fields
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({"success": False, "error": "No file path provided"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File not found: {file_path}"}), 404
        
        # Log the request
        log_info(
            source='visualization',
            message=f"Dashboard request for {file_path}",
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            details={
                'file_path': file_path
            }
        )
        
        # Generate dashboard
        result = generate_dashboard(file_path)
        
        if result.get('success', False):
            return jsonify(result), 200
        else:
            log_error(
                source='visualization',
                message=f"Error generating dashboard: {result.get('error', 'Unknown error')}",
                user_id=request.headers.get('X-User-ID', 'anonymous'),
                details={
                    'file_path': file_path
                }
            )
            return jsonify(result), 500
    
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        log_error(
            source='visualization',
            message=f"Error in dashboard route: {str(e)}",
            user_id=request.headers.get('X-User-ID', 'anonymous')
        )
        return jsonify({
            "success": False,
            "error": "Failed to generate dashboard",
            "message": str(e)
        }), 500

@visualization_bp.route('/visualizations/<filename>', methods=['GET'])
def get_visualization(filename):
    """Get a visualization image by filename"""
    try:
        # Get visualization folder
        viz_folder = os.path.join(current_app.config.get('UPLOAD_FOLDER', 'uploads'), 'visualizations')
        
        # Construct file path
        file_path = os.path.join(viz_folder, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"Visualization not found: {filename}"}), 404
        
        # Return the file
        return send_file(file_path, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Error retrieving visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve visualization",
            "message": str(e)
        }), 500

@visualization_bp.route('/plotly', methods=['POST'])
def plotly_visualization():
    """Generate a Plotly visualization for a dataset"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Check for required fields
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({"success": False, "error": "No file path provided"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File not found: {file_path}"}), 404
        
        # Get visualization type and parameters
        viz_type = data.get('type', 'auto')
        params = data.get('params', {})
        
        # Log the request
        log_info(
            source='visualization',
            message=f"Plotly visualization request for {file_path}",
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            details={
                'file_path': file_path,
                'viz_type': viz_type,
                'params': params
            }
        )
        
        # Read the data
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Failed to read file: {str(e)}"
                }), 400
        
        # If viz_type is auto, suggest a chart type
        if viz_type == 'auto':
            viz_type = suggest_chart_type(df)
        
        # Generate Plotly visualization
        plotly_data = generate_plotly_visualization(df, viz_type, params)
        
        if plotly_data:
            return jsonify({
                "success": True,
                "visualization": {
                    "type": viz_type,
                    "title": params.get('title', f'{viz_type.capitalize()} Chart'),
                    "data": plotly_data
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to generate Plotly visualization of type {viz_type}"
            }), 500
    
    except Exception as e:
        logger.error(f"Error in Plotly visualization route: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to generate Plotly visualization",
            "message": str(e)
        }), 500

@visualization_bp.route('/suggest-chart', methods=['POST'])
def suggest_chart():
    """Suggest the best chart type for a dataset"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Check for required fields
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({"success": False, "error": "No file path provided"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File not found: {file_path}"}), 404
        
        # Read the data
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Failed to read file: {str(e)}"
                }), 400
        
        # Suggest chart type
        chart_type = suggest_chart_type(df)
        
        return jsonify({
            "success": True,
            "chart_type": chart_type
        }), 200
    
    except Exception as e:
        logger.error(f"Error in suggest chart route: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to suggest chart type",
            "message": str(e)
        }), 500 