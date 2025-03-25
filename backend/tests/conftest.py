import pytest
from datetime import datetime
from unittest.mock import Mock
from flask import Flask
from routes.auth_routes import auth_bp
from .config import (
    TEST_DATABASE_URL, TEST_REDIS_URL, TEST_JWT_SECRET,
    TEST_ACCESS_TOKEN_EXPIRES, TEST_REFRESH_TOKEN_EXPIRES,
    TEST_RATE_LIMITS, TEST_ROLES, TEST_CIRCUIT_BREAKER,
    TEST_RETRY_CONFIG, TEST_VALIDATION
)

@pytest.fixture
def app():
    """Create Flask test app"""
    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': TEST_DATABASE_URL,
        'REDIS_URL': TEST_REDIS_URL,
        'JWT_SECRET_KEY': TEST_JWT_SECRET,
        'JWT_ACCESS_TOKEN_EXPIRES': TEST_ACCESS_TOKEN_EXPIRES,
        'JWT_REFRESH_TOKEN_EXPIRES': TEST_REFRESH_TOKEN_EXPIRES,
        'RATE_LIMITS': TEST_RATE_LIMITS,
        'ROLES': TEST_ROLES,
        'CIRCUIT_BREAKER': TEST_CIRCUIT_BREAKER,
        'RETRY_CONFIG': TEST_RETRY_CONFIG,
        'VALIDATION': TEST_VALIDATION
    })
    app.register_blueprint(auth_bp)
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def mock_db_session():
    """Create mock database session"""
    session = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    return session

@pytest.fixture
def mock_redis():
    """Create mock Redis client"""
    redis = Mock()
    redis.get = Mock()
    redis.set = Mock()
    redis.setex = Mock()
    redis.delete = Mock()
    redis.close = Mock()
    redis.pipeline = Mock(return_value=Mock())
    return redis

@pytest.fixture
def mock_user():
    """Create mock user data"""
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

@pytest.fixture
def mock_admin_user(mock_user):
    """Create mock admin user data"""
    return {
        **mock_user,
        'role': 'admin'
    }

@pytest.fixture
def mock_tokens():
    """Create mock authentication tokens"""
    return {
        'access_token': 'mock_access_token',
        'refresh_token': 'mock_refresh_token',
        'session_id': 'mock_session_id'
    }

@pytest.fixture
def mock_auth_headers(mock_tokens):
    """Create mock authentication headers"""
    return {
        'Authorization': f"Bearer {mock_tokens['access_token']}"
    }

@pytest.fixture
def mock_rate_limits():
    """Create mock rate limit configuration"""
    return TEST_RATE_LIMITS

@pytest.fixture
def mock_roles():
    """Create mock role configuration"""
    return TEST_ROLES

@pytest.fixture
def mock_audit_log():
    """Create mock audit log entry"""
    return {
        'timestamp': datetime.utcnow(),
        'user_id': '1',
        'action': 'login',
        'resource': 'auth',
        'status': 'success',
        'ip_address': '127.0.0.1',
        'user_agent': 'Mozilla/5.0',
        'details': {}
    }

@pytest.fixture
def mock_session():
    """Create mock user session"""
    return {
        'session_id': 'mock_session_id',
        'created_at': datetime.utcnow(),
        'last_activity': datetime.utcnow()
    }

@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker configuration"""
    return TEST_CIRCUIT_BREAKER

@pytest.fixture
def mock_retry_config():
    """Create mock retry configuration"""
    return TEST_RETRY_CONFIG

@pytest.fixture
def mock_validation():
    """Create mock validation configuration"""
    return TEST_VALIDATION 