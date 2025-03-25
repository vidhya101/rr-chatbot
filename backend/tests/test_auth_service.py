import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from services.auth_service import AuthService, User
from utils.exceptions import (
    AuthenticationError, AuthorizationError, RateLimitError,
    TokenError, ValidationError, DatabaseError, ResourceCleanupError
)

@pytest.fixture
def auth_service(mock_db_session, mock_redis):
    """Create AuthService instance with mocked dependencies"""
    return AuthService(redis_url=None, db_session=mock_db_session)

@pytest.fixture
def mock_user():
    """Create a mock user"""
    return {
        'id': '1',
        'username': 'testuser',
        'email': 'test@example.com',
        'password': b'$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY.5AQGHXyOQGd2',  # 'password'
        'role': 'user',
        'is_active': True,
        'is_verified': True,
        'first_name': 'Test',
        'last_name': 'User',
        'profile': {},
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
        'last_login': None
    }

def test_hash_password(auth_service):
    """Test password hashing"""
    password = "testpassword"
    hashed = auth_service.hash_password(password)
    assert isinstance(hashed, bytes)
    assert auth_service.verify_password(password, hashed)
    assert not auth_service.verify_password("wrongpassword", hashed)

def test_authenticate_user_success(auth_service, mock_user, mock_tokens):
    """Test successful user authentication"""
    auth_service.db.query.return_value.filter.return_value.first.return_value = Mock(
        to_dict=lambda: mock_user
    )
    auth_service._generate_token = Mock(side_effect=[
        mock_tokens['access_token'],
        mock_tokens['refresh_token']
    ])
    
    result = auth_service.authenticate_user("testuser", "password")
    
    assert result['user']['username'] == 'testuser'
    assert result['access_token'] == mock_tokens['access_token']
    assert result['refresh_token'] == mock_tokens['refresh_token']
    assert 'session_id' in result

def test_authenticate_user_invalid_credentials(auth_service):
    """Test authentication with invalid credentials"""
    auth_service.db.query.return_value.filter.return_value.first.return_value = None
    
    with pytest.raises(AuthenticationError):
        auth_service.authenticate_user("wronguser", "wrongpass")

def test_authenticate_user_rate_limit(auth_service, mock_rate_limits):
    """Test authentication rate limiting"""
    auth_service.redis.get.return_value = str(mock_rate_limits['login']['calls'])
    
    with pytest.raises(RateLimitError):
        auth_service.authenticate_user("testuser", "password")

def test_refresh_access_token_success(auth_service, mock_user, mock_tokens):
    """Test successful token refresh"""
    auth_service.redis.get.return_value = mock_user['id']
    auth_service.db.query.return_value.filter.return_value.first.return_value = Mock(
        to_dict=lambda: mock_user
    )
    auth_service._generate_token = Mock(return_value=mock_tokens['access_token'])
    
    result = auth_service.refresh_access_token(mock_tokens['refresh_token'])
    
    assert result['access_token'] == mock_tokens['access_token']

def test_refresh_access_token_invalid(auth_service):
    """Test token refresh with invalid token"""
    auth_service.redis.get.return_value = None
    
    with pytest.raises(TokenError):
        auth_service.refresh_access_token("invalid_refresh_token")

def test_create_user_success(auth_service, mock_user):
    """Test successful user creation"""
    auth_service.db.query.return_value.filter.return_value.first.return_value = None
    auth_service.db.add = Mock()
    auth_service.db.commit = Mock()
    
    new_user = Mock(to_dict=lambda: mock_user)
    auth_service.db.add.return_value = None
    
    with patch('services.auth_service.User') as mock_user_class:
        mock_user_class.return_value = new_user
        result = auth_service.create_user(
            username=mock_user['username'],
            email=mock_user['email'],
            password="password"
        )
    
    assert result['username'] == mock_user['username']
    assert result['email'] == mock_user['email']
    auth_service.db.add.assert_called_once()
    auth_service.db.commit.assert_called_once()

def test_create_user_duplicate(auth_service):
    """Test user creation with duplicate username"""
    auth_service.db.query.return_value.filter.return_value.first.return_value = Mock()
    
    with pytest.raises(ValidationError):
        auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password"
        )

def test_update_user_success(auth_service, mock_user):
    """Test successful user update"""
    user = Mock(to_dict=lambda: mock_user)
    auth_service.db.query.return_value.filter.return_value.first.return_value = user
    auth_service.db.commit = Mock()
    
    updates = {
        'email': 'newemail@example.com',
        'first_name': 'NewName'
    }
    
    result = auth_service.update_user(mock_user['id'], updates)
    
    assert result['email'] == 'newemail@example.com'
    assert result['first_name'] == 'NewName'
    auth_service.db.commit.assert_called_once()

def test_update_user_not_found(auth_service):
    """Test user update with non-existent user"""
    auth_service.db.query.return_value.filter.return_value.first.return_value = None
    
    with pytest.raises(ValidationError):
        auth_service.update_user('999', {'email': 'new@example.com'})

def test_delete_user_success(auth_service, mock_user):
    """Test successful user deletion"""
    user = Mock(to_dict=lambda: mock_user)
    auth_service.db.query.return_value.filter.return_value.first.return_value = user
    auth_service.db.commit = Mock()
    
    result = auth_service.delete_user(mock_user['id'])
    
    assert result is True
    assert user.is_active is False
    auth_service.db.commit.assert_called_once()

def test_delete_user_not_found(auth_service):
    """Test user deletion with non-existent user"""
    auth_service.db.query.return_value.filter.return_value.first.return_value = None
    
    with pytest.raises(ValidationError):
        auth_service.delete_user('999')

def test_change_password_success(auth_service, mock_user):
    """Test successful password change"""
    user = Mock(
        password=mock_user['password'],
        to_dict=lambda: mock_user
    )
    auth_service.db.query.return_value.filter.return_value.first.return_value = user
    auth_service.db.commit = Mock()
    
    result = auth_service.change_password(
        mock_user['id'],
        'password',  # Current password
        'newpassword'  # New password
    )
    
    assert result is True
    auth_service.db.commit.assert_called_once()

def test_change_password_invalid_current(auth_service, mock_user):
    """Test password change with invalid current password"""
    user = Mock(
        password=mock_user['password'],
        to_dict=lambda: mock_user
    )
    auth_service.db.query.return_value.filter.return_value.first.return_value = user
    
    with pytest.raises(ValidationError):
        auth_service.change_password(
            mock_user['id'],
            'wrongpassword',  # Wrong current password
            'newpassword'
        )

def test_list_users_success(auth_service, mock_user):
    """Test successful user listing"""
    users = [Mock(to_dict=lambda: mock_user)]
    auth_service.db.query.return_value.filter.return_value.all.return_value = users
    
    result = auth_service.list_users()
    
    assert len(result) == 1
    assert result[0]['username'] == mock_user['username']

def test_list_users_empty(auth_service):
    """Test user listing with no users"""
    auth_service.db.query.return_value.filter.return_value.all.return_value = []
    
    result = auth_service.list_users()
    
    assert len(result) == 0

def test_cleanup_resources(auth_service):
    """Test resource cleanup"""
    auth_service.redis.close = Mock()
    auth_service.db.close = Mock()
    
    auth_service._cleanup_resources()
    
    auth_service.redis.close.assert_called_once()
    auth_service.db.close.assert_called_once()

def test_cleanup_resources_error(auth_service):
    """Test resource cleanup with error"""
    auth_service.redis.close = Mock(side_effect=Exception("Redis error"))
    auth_service.db.close = Mock()
    
    with pytest.raises(ResourceCleanupError):
        auth_service._cleanup_resources()

def test_context_manager(auth_service):
    """Test context manager functionality"""
    auth_service._cleanup_resources = Mock()
    
    with auth_service:
        pass
    
    auth_service._cleanup_resources.assert_called_once() 