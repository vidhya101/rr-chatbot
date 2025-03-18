from flask import Blueprint, jsonify, request
import logging
from services.model_service import (
    get_available_models, 
    get_model_by_id, 
    check_ollama_status, 
    pull_ollama_model,
    update_model_parameters,
    model_service
)
from utils.db_utils import log_info, log_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
model_bp = Blueprint('model', __name__)

@model_bp.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    try:
        models = model_service.get_available_models()
        return jsonify({
            'success': True,
            'models': models,
            'current_model': model_service.get_current_model()
        })
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@model_bp.route('/models/refresh', methods=['POST'])
def refresh_models():
    """Refresh available models"""
    try:
        success = model_service.refresh_models()
        return jsonify({
            'success': success,
            'models': model_service.get_available_models() if success else []
        })
    except Exception as e:
        logger.error(f"Error refreshing models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@model_bp.route('/models/select', methods=['POST'])
def select_model():
    """Select a model to use"""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({
                'success': False,
                'error': 'Model name not provided'
            }), 400

        model_name = data['model_name']
        success = model_service.set_current_model(model_name)
        
        return jsonify({
            'success': success,
            'current_model': model_service.get_current_model()
        })
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@model_bp.route('/api-keys', methods=['POST'])
def update_api_key():
    """Update API key for a provider"""
    try:
        data = request.get_json()
        if not data or 'provider' not in data or 'api_key' not in data:
            return jsonify({
                'success': False,
                'error': 'Provider and API key required'
            }), 400

        provider = data['provider']
        api_key = data['api_key']
        
        model_service.update_api_key(provider, api_key)
        
        return jsonify({
            'success': True,
            'message': f'API key updated for {provider}'
        })
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@model_bp.route('/health', methods=['GET'])
def check_health():
    """Check health of model service"""
    try:
        health_status = model_service.check_model_health()
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@model_bp.route('/models/<model_id>', methods=['GET'])
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

@model_bp.route('/models/<model_id>/parameters', methods=['PUT'])
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

@model_bp.route('/ollama/status', methods=['GET'])
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

@model_bp.route('/ollama/pull', methods=['POST'])
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