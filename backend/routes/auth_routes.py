from flask import Blueprint, request, jsonify, g
from functools import wraps
from typing import Dict, Any
from pydantic import BaseModel, EmailStr, constr, validator
from services.auth_service import AuthService
from utils.exceptions import (
    AuthenticationError, AuthorizationError, RateLimitError,
    TokenError, ValidationError, DatabaseError
)

# Create Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Request validation models
class LoginRequest(BaseModel):
    username: constr(min_length=3, max_length=50)
    password: constr(min_length=8, max_length=100)

class RegisterRequest(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    password: constr(min_length=8, max_length=100)
    first_name: constr(max_length=50) = None
    last_name: constr(max_length=50) = None

class UpdateUserRequest(BaseModel):
    email: EmailStr = None
    first_name: constr(max_length=50) = None
    last_name: constr(max_length=50) = None
    profile: Dict[str, Any] = None

class PasswordChangeRequest(BaseModel):
    current_password: constr(min_length=8, max_length=100)
    new_password: constr(min_length=8, max_length=100)

    @validator('new_password')
    def passwords_must_be_different(cls, v, values):
        if 'current_password' in values and v == values['current_password']:
            raise ValueError('New password must be different from current password')
        return v

def validate_request(schema_cls):
    """Decorator for request validation"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            try:
                data = request.get_json()
                validated_data = schema_cls(**data).dict(exclude_none=True)
                return f(validated_data, *args, **kwargs)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        return decorated
    return decorator

def handle_errors(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AuthenticationError as e:
            return jsonify({'error': str(e)}), 401
        except AuthorizationError as e:
            return jsonify({'error': str(e)}), 403
        except RateLimitError as e:
            return jsonify({'error': str(e)}), 429
        except TokenError as e:
            return jsonify({'error': str(e)}), 401
        except ValidationError as e:
            return jsonify({'error': str(e)}), 400
        except DatabaseError as e:
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            return jsonify({'error': 'Internal server error'}), 500
    return decorated

# Routes
@auth_bp.route('/login', methods=['POST'])
@handle_errors
@validate_request(LoginRequest)
def login(data):
    """Login user and return tokens"""
    auth_service = AuthService()
    result = auth_service.authenticate_user(
        username=data['username'],
        password=data['password']
    )
    return jsonify(result), 200

@auth_bp.route('/register', methods=['POST'])
@handle_errors
@validate_request(RegisterRequest)
def register(data):
    """Register new user"""
    auth_service = AuthService()
    user = auth_service.create_user(**data)
    return jsonify(user), 201

@auth_bp.route('/refresh', methods=['POST'])
@handle_errors
def refresh_token():
    """Refresh access token"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise AuthenticationError('No refresh token provided')
    
    refresh_token = auth_header.split(' ')[1]
    auth_service = AuthService()
    result = auth_service.refresh_access_token(refresh_token)
    return jsonify(result), 200

@auth_bp.route('/me', methods=['GET'])
@handle_errors
def get_current_user():
    """Get current user information"""
    if not hasattr(g, 'user'):
        raise AuthenticationError('Not authenticated')
    return jsonify(g.user), 200

@auth_bp.route('/me', methods=['PUT'])
@handle_errors
@validate_request(UpdateUserRequest)
def update_current_user(data):
    """Update current user information"""
    if not hasattr(g, 'user'):
        raise AuthenticationError('Not authenticated')
    
    auth_service = AuthService()
    user = auth_service.update_user(g.user['id'], data)
    return jsonify(user), 200

@auth_bp.route('/me/password', methods=['PUT'])
@handle_errors
@validate_request(PasswordChangeRequest)
def change_password(data):
    """Change user password"""
    if not hasattr(g, 'user'):
        raise AuthenticationError('Not authenticated')
    
    auth_service = AuthService()
    auth_service.change_password(
        g.user['id'],
        data['current_password'],
        data['new_password']
    )
    return jsonify({'message': 'Password changed successfully'}), 200

@auth_bp.route('/me', methods=['DELETE'])
@handle_errors
def delete_current_user():
    """Delete current user"""
    if not hasattr(g, 'user'):
        raise AuthenticationError('Not authenticated')
    
    auth_service = AuthService()
    auth_service.delete_user(g.user['id'])
    return jsonify({'message': 'User deleted successfully'}), 200

# Admin routes
@auth_bp.route('/users', methods=['GET'])
@handle_errors
def list_users():
    """List all users (admin only)"""
    if not hasattr(g, 'user') or g.user['role'] != 'admin':
        raise AuthorizationError('Admin access required')
    
    auth_service = AuthService()
    users = auth_service.list_users()
    return jsonify(users), 200

@auth_bp.route('/users/<user_id>', methods=['GET'])
@handle_errors
def get_user(user_id):
    """Get user by ID (admin only)"""
    if not hasattr(g, 'user') or g.user['role'] != 'admin':
        raise AuthorizationError('Admin access required')
    
    auth_service = AuthService()
    user = auth_service._get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user), 200

@auth_bp.route('/users/<user_id>', methods=['PUT'])
@handle_errors
@validate_request(UpdateUserRequest)
def update_user(data, user_id):
    """Update user by ID (admin only)"""
    if not hasattr(g, 'user') or g.user['role'] != 'admin':
        raise AuthorizationError('Admin access required')
    
    auth_service = AuthService()
    user = auth_service.update_user(user_id, data)
    return jsonify(user), 200

@auth_bp.route('/users/<user_id>', methods=['DELETE'])
@handle_errors
def delete_user(user_id):
    """Delete user by ID (admin only)"""
    if not hasattr(g, 'user') or g.user['role'] != 'admin':
        raise AuthorizationError('Admin access required')
    
    auth_service = AuthService()
    auth_service.delete_user(user_id)
    return jsonify({'message': 'User deleted successfully'}), 200 