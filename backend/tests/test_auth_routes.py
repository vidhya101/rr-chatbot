import pytest
from flask import Flask, g
from unittest.mock import Mock, patch
from routes.auth_routes import auth_bp
from services.auth_service import AuthService

@pytest.fixture
def app():
    """Create Flask test app"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(auth_bp)
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def mock_auth_service():
    """Create mock AuthService"""
    with patch('routes.auth_routes.AuthService') as mock:
        yield mock

@pytest.fixture
def mock_user():
    """Create mock user data"""
    return {
        'id': '1',
        'username': 'testuser',
        'email': 'test@example.com',
        'role': 'user',
        'is_active': True,
        'is_verified': True,
        'first_name': 'Test',
        'last_name': 'User'
    }

def test_login_success(client, mock_auth_service, mock_user, mock_tokens):
    """Test successful login"""
    mock_auth_service.return_value.authenticate_user.return_value = {
        'user': mock_user,
        'access_token': mock_tokens['access_token'],
        'refresh_token': mock_tokens['refresh_token'],
        'session_id': mock_tokens['session_id']
    }
    
    response = client.post('/api/auth/login', json={
        'username': mock_user['username'],
        'password': 'password'
    })
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['user']['username'] == mock_user['username']
    assert data['access_token'] == mock_tokens['access_token']
    assert data['refresh_token'] == mock_tokens['refresh_token']
    assert data['session_id'] == mock_tokens['session_id']

def test_login_invalid_request(client):
    """Test login with invalid request data"""
    response = client.post('/api/auth/login', json={
        'username': 'te',  # Too short
        'password': 'pass'  # Too short
    })
    
    assert response.status_code == 400

def test_register_success(client, mock_auth_service, mock_user):
    """Test successful user registration"""
    mock_auth_service.return_value.create_user.return_value = mock_user
    
    response = client.post('/api/auth/register', json={
        'username': mock_user['username'],
        'email': mock_user['email'],
        'password': 'password123',
        'first_name': mock_user['first_name'],
        'last_name': mock_user['last_name']
    })
    
    assert response.status_code == 201
    data = response.get_json()
    assert data['username'] == mock_user['username']
    assert data['email'] == mock_user['email']

def test_register_invalid_email(client):
    """Test registration with invalid email"""
    response = client.post('/api/auth/register', json={
        'username': 'testuser',
        'email': 'invalid-email',
        'password': 'password123'
    })
    
    assert response.status_code == 400

def test_refresh_token_success(client, mock_auth_service, mock_tokens):
    """Test successful token refresh"""
    mock_auth_service.return_value.refresh_access_token.return_value = {
        'access_token': mock_tokens['access_token']
    }
    
    response = client.post(
        '/api/auth/refresh',
        headers={'Authorization': f"Bearer {mock_tokens['refresh_token']}"}
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['access_token'] == mock_tokens['access_token']

def test_refresh_token_missing_header(client):
    """Test token refresh without Authorization header"""
    response = client.post('/api/auth/refresh')
    
    assert response.status_code == 401

def test_get_current_user_success(client, mock_user):
    """Test getting current user information"""
    with client.application.app_context():
        g.user = mock_user
        response = client.get('/api/auth/me')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['username'] == mock_user['username']

def test_get_current_user_unauthorized(client):
    """Test getting current user without authentication"""
    response = client.get('/api/auth/me')
    
    assert response.status_code == 401

def test_update_current_user_success(client, mock_auth_service, mock_user):
    """Test updating current user information"""
    updated_user = {
        **mock_user,
        'email': 'newemail@example.com'
    }
    mock_auth_service.return_value.update_user.return_value = updated_user
    
    with client.application.app_context():
        g.user = mock_user
        response = client.put('/api/auth/me', json={
            'email': 'newemail@example.com'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['email'] == 'newemail@example.com'

def test_update_current_user_unauthorized(client):
    """Test updating current user without authentication"""
    response = client.put('/api/auth/me', json={
        'email': 'newemail@example.com'
    })
    
    assert response.status_code == 401

def test_change_password_success(client, mock_auth_service, mock_user):
    """Test successful password change"""
    mock_auth_service.return_value.change_password.return_value = True
    
    with client.application.app_context():
        g.user = mock_user
        response = client.put('/api/auth/me/password', json={
            'current_password': 'oldpassword',
            'new_password': 'newpassword123'
        })
        
        assert response.status_code == 200

def test_change_password_unauthorized(client):
    """Test password change without authentication"""
    response = client.put('/api/auth/me/password', json={
        'current_password': 'oldpassword',
        'new_password': 'newpassword123'
    })
    
    assert response.status_code == 401

def test_delete_current_user_success(client, mock_auth_service, mock_user):
    """Test successful user deletion"""
    mock_auth_service.return_value.delete_user.return_value = True
    
    with client.application.app_context():
        g.user = mock_user
        response = client.delete('/api/auth/me')
        
        assert response.status_code == 200

def test_delete_current_user_unauthorized(client):
    """Test user deletion without authentication"""
    response = client.delete('/api/auth/me')
    
    assert response.status_code == 401

def test_list_users_admin(client, mock_auth_service, mock_user, mock_admin_user):
    """Test listing users as admin"""
    mock_auth_service.return_value.list_users.return_value = [mock_user]
    
    with client.application.app_context():
        g.user = mock_admin_user
        response = client.get('/api/auth/users')
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 1
        assert data[0]['username'] == mock_user['username']

def test_list_users_unauthorized(client, mock_user):
    """Test listing users without admin role"""
    with client.application.app_context():
        g.user = mock_user  # Regular user role
        response = client.get('/api/auth/users')
        
        assert response.status_code == 403

def test_get_user_admin(client, mock_auth_service, mock_user, mock_admin_user):
    """Test getting user by ID as admin"""
    mock_auth_service.return_value._get_user_by_id.return_value = mock_user
    
    with client.application.app_context():
        g.user = mock_admin_user
        response = client.get(f"/api/auth/users/{mock_user['id']}")
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['username'] == mock_user['username']

def test_get_user_not_found(client, mock_auth_service, mock_admin_user):
    """Test getting non-existent user as admin"""
    mock_auth_service.return_value._get_user_by_id.return_value = None
    
    with client.application.app_context():
        g.user = mock_admin_user
        response = client.get('/api/auth/users/999')
        
        assert response.status_code == 404

def test_update_user_admin(client, mock_auth_service, mock_user, mock_admin_user):
    """Test updating user as admin"""
    updated_user = {
        **mock_user,
        'email': 'updated@example.com'
    }
    mock_auth_service.return_value.update_user.return_value = updated_user
    
    with client.application.app_context():
        g.user = mock_admin_user
        response = client.put(f"/api/auth/users/{mock_user['id']}", json={
            'email': 'updated@example.com'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['email'] == 'updated@example.com'

def test_delete_user_admin(client, mock_auth_service, mock_user, mock_admin_user):
    """Test deleting user as admin"""
    mock_auth_service.return_value.delete_user.return_value = True
    
    with client.application.app_context():
        g.user = mock_admin_user
        response = client.delete(f"/api/auth/users/{mock_user['id']}")
        
        assert response.status_code == 200 