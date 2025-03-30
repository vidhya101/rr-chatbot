from flask import Blueprint, jsonify, request
import logging
from services.data_analysis_service import analyze_product_recommendations
from utils.db_utils import log_info, log_error
from utils.data_utils import convert_to_serializable
import os
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
data_modeling_routes = Blueprint('data_modeling_routes', __name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return super().default(obj)

@data_modeling_routes.route('/modeling/analyze', methods=['POST'])
def analyze_data():
    """Analyze data file and create visualizations"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'file_id' not in data:
            return jsonify({
                'success': False,
                'error': 'File ID is required'
            }), 400
            
        file_id = data['file_id']
        
        # Get file path
        file_path = os.path.join('uploads', f'{file_id}.csv')
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {file_id}'
            }), 404
        
        # Log request
        try:
            log_info(
                user_id=request.headers.get('X-User-ID', 'anonymous'),
                source='data_modeling_routes.analyze_data',
                message=f'Analyzing data file: {file_id}',
                details={'file_id': file_id}
            )
        except Exception as log_err:
            logger.warning(f"Could not log info: {str(log_err)}")
        
        # Analyze data
        results = analyze_product_recommendations(file_path)
        
        # Check visualization count
        viz_count = results.get('visualization_count', 0)
        if viz_count < 12:
            logger.warning(f"Generated only {viz_count} visualizations, expected at least 12")
        
        # Convert results to JSON-serializable format
        serializable_results = convert_to_serializable(results)
        
        return jsonify({
            'success': True,
            'results': serializable_results,
            'visualization_count': viz_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_modeling_routes.route('/modeling/status/<file_id>', methods=['GET'])
def get_analysis_status(file_id):
    """Get the status of data analysis for a file"""
    try:
        # Here you would check the status of the analysis
        # For now, we'll return a mock status
        status = {
            'file_id': file_id,
            'status': 'completed',  # or 'in_progress', 'failed', etc.
            'progress': 100,  # percentage
            'message': 'Analysis completed successfully'
        }
        
        return jsonify({
            'success': True,
            'status': status
        }), 200
        
    except Exception as e:
        error_message = f"Error getting analysis status: {str(e)}"
        logger.error(error_message)
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500 