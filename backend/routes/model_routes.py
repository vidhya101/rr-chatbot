from flask import Blueprint, jsonify, request
import logging
from services.model_service import (
    get_available_models, 
    get_model_by_id, 
    check_ollama_status, 
    pull_ollama_model,
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
        # Get force refresh parameter
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        # Get models
        models = get_available_models(force_refresh=force_refresh)
        
        # Log request
        log_info(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.list_models',
            message='Listed models',
            details={'count': len(models), 'force_refresh': force_refresh}
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
    """Get model details by ID"""
    try:
        # Get model
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
            message=f"Got model {model_id}",
            details={'model_id': model_id}
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
            message=f"Error getting model {model_id}",
            details={'error': str(e), 'model_id': model_id}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@model_routes.route('/models/<model_id>/parameters', methods=['PUT'])
def update_parameters(model_id):
    """Update model parameters"""
    try:
        # Get parameters from request
        parameters = request.json
        
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
            message=f"Updated parameters for model {model_id}",
            details={'model_id': model_id, 'parameters': parameters}
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
            message=f"Error updating parameters for model {model_id}",
            details={'error': str(e), 'model_id': model_id}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@model_routes.route('/ollama/status', methods=['GET'])
def ollama_status():
    """Check OLLAMA server status"""
    try:
        # Check status
        status, message = check_ollama_status()
        
        # Log request
        log_info(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.ollama_status',
            message='Checked OLLAMA status',
            details={'status': status, 'message': message}
        )
        
        return jsonify({
            'success': True,
            'status': status,
            'message': message
        }), 200
    
    except Exception as e:
        error_message = f"Error checking OLLAMA status: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.ollama_status',
            message='Error checking OLLAMA status',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@model_routes.route('/ollama/pull', methods=['POST'])
def pull_model():
    """Pull a model from OLLAMA"""
    try:
        # Get model name from request
        data = request.json
        
        if not data or 'model' not in data:
            return jsonify({
                'success': False,
                'error': "Model name not provided"
            }), 400
        
        model_name = data['model']
        
        # Pull model
        success, message = pull_ollama_model(model_name)
        
        # Log request
        log_info(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.pull_model',
            message=f"Pulled model {model_name}",
            details={'model_name': model_name, 'success': success, 'message': message}
        )
        
        return jsonify({
            'success': success,
            'message': message
        }), 200 if success else 500
    
    except Exception as e:
        error_message = f"Error pulling model: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='model_routes.pull_model',
            message='Error pulling model',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500 