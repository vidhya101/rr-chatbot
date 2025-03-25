from flask import Blueprint, jsonify, request
import logging
from services.model_service import (
    get_available_models, 
    get_model_by_id,
    update_model_parameters
)
from utils.db_utils import log_info, log_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
model_routes = Blueprint('model_routes', __name__)

@model_routes.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    try:
        # Get models
        models = get_available_models()
        
        # Log request
        log_info(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.list_models',
            message='Listed models',
            details={'count': len(models)}
        )
        
        return jsonify({
            'success': True,
            'models': models
        }), 200
    
    except Exception as e:
        error_message = f"Error listing models: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.list_models',
            message='Error listing models',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@model_routes.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get details of a specific model"""
    try:
        model = get_model_by_id(model_id)
        
        if not model:
            return jsonify({
                'success': False,
                'error': f"Model {model_id} not found"
            }), 404
        
        # Log request
        log_info(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.get_model',
            message=f'Retrieved model {model_id}',
            details={'model': model}
        )
        
        return jsonify({
            'success': True,
            'model': model
        }), 200
    
    except Exception as e:
        error_message = f"Error getting model {model_id}: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.get_model',
            message=f'Error getting model {model_id}',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@model_routes.route('/models/<model_id>/parameters', methods=['PUT'])
def update_parameters(model_id):
    """Update parameters for a specific model"""
    try:
        # Get parameters from request
        parameters = request.get_json()
        
        if not parameters:
            return jsonify({
                'success': False,
                'error': "No parameters provided"
            }), 400
        
        # Update parameters
        success, message = update_model_parameters(model_id, parameters)
        
        if not success:
            return jsonify({
                'success': False,
                'error': message
            }), 404
        
        # Log request
        log_info(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.update_parameters',
            message=f'Updated parameters for model {model_id}',
            details={'parameters': parameters}
        )
        
        return jsonify({
            'success': True,
            'message': message
        }), 200
    
    except Exception as e:
        error_message = f"Error updating parameters for model {model_id}: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.update_parameters',
            message=f'Error updating parameters for model {model_id}',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500 