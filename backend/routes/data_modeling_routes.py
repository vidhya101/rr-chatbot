from flask import Blueprint, jsonify, request
import logging
from services.data_analysis_service import analyze_product_recommendations
from utils.db_utils import log_info, log_error
from utils.data_utils import convert_to_serializable
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
data_modeling_routes = Blueprint('data_modeling_routes', __name__)

@data_modeling_routes.route('/modeling/analyze', methods=['POST'])
def analyze_data():
    """Analyze data file and create dashboard"""
    try:
        # Get file path from request
        data = request.get_json()
        if not data or 'file_id' not in data:
            return jsonify({
                'success': False,
                'error': 'File ID is required'
            }), 400
            
        file_id = data['file_id']
        
        # Get file path from file ID (you'll need to implement this based on your file storage)
        file_path = os.path.join('uploads', f'{file_id}.csv')  # Example path
        
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
        
        # Convert results to serializable format
        serializable_results = convert_to_serializable(results)
        
        return jsonify({
            'success': True,
            'results': serializable_results
        }), 200
        
    except Exception as e:
        error_message = f"Error analyzing data: {str(e)}"
        logger.error(error_message)
        
        # Log error
        try:
            log_error(
                user_id=request.headers.get('X-User-ID', 'anonymous'),
                source='data_modeling_routes.analyze_data',
                message='Error analyzing data',
                details={'error': str(e)}
            )
        except Exception as log_err:
            logger.warning(f"Could not log error: {str(log_err)}")
        
        return jsonify({
            'success': False,
            'error': error_message
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