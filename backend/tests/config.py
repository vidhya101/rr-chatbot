"""Test configuration settings"""

import os
from datetime import timedelta

# Test database
TEST_DATABASE_URL = 'sqlite:///:memory:'

# Test Redis
TEST_REDIS_URL = 'redis://localhost:6379/1'

# Test JWT settings
TEST_JWT_SECRET = 'test-jwt-secret'
TEST_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
TEST_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

# Test rate limits
TEST_RATE_LIMITS = {
    'login': {'calls': 5, 'period': 300},
    'refresh_token': {'calls': 10, 'period': 300},
    'api': {'calls': 100, 'period': 60}
}

# Test roles
TEST_ROLES = {
    'admin': {
        'name': 'admin',
        'permissions': ['read', 'write', 'delete', 'manage_users'],
        'priority': 100
    },
    'moderator': {
        'name': 'moderator',
        'permissions': ['read', 'write'],
        'priority': 50
    },
    'user': {
        'name': 'user',
        'permissions': ['read'],
        'priority': 10
    }
}

# Test upload settings
TEST_UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
TEST_MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Test CORS settings
TEST_CORS_ORIGINS = ['http://localhost:3000']

# Test logging settings
TEST_LOG_LEVEL = 'DEBUG'
TEST_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Test security settings
TEST_PASSWORD_SALT_ROUNDS = 4  # Lower for faster tests
TEST_SESSION_TIMEOUT = timedelta(minutes=30)
TEST_FAILED_LOGIN_DELAY = 0  # No delay in tests

# Test metrics settings
TEST_METRICS_ENABLED = False
TEST_METRICS_PORT = 9100

# Test circuit breaker settings
TEST_CIRCUIT_BREAKER = {
    'failure_threshold': 3,
    'reset_timeout': 30.0
}

# Test retry settings
TEST_RETRY_CONFIG = {
    'max_attempts': 3,
    'delay': 0.1,
    'backoff': 2.0
}

# Test cleanup settings
TEST_CLEANUP_INTERVAL = 300  # 5 minutes
TEST_SESSION_CLEANUP_THRESHOLD = timedelta(hours=1)
TEST_AUDIT_LOG_RETENTION = timedelta(days=7)

# Test feature flags
TEST_FEATURES = {
    'email_verification': False,
    'two_factor_auth': False,
    'password_reset': True,
    'social_login': False
}

# Test notification settings
TEST_NOTIFICATIONS = {
    'enabled': False,
    'email': {
        'enabled': False,
        'from_address': 'test@example.com',
        'smtp_host': 'localhost',
        'smtp_port': 1025
    },
    'sms': {
        'enabled': False,
        'provider': 'test'
    }
}

# Test cache settings
TEST_CACHE = {
    'type': 'memory',
    'ttl': 300,
    'max_size': 1000
}

# Test monitoring settings
TEST_MONITORING = {
    'enabled': False,
    'interval': 60,
    'retention': {
        'metrics': timedelta(days=7),
        'logs': timedelta(days=30),
        'traces': timedelta(days=3)
    }
}

# Test validation settings
TEST_VALIDATION = {
    'username': {
        'min_length': 3,
        'max_length': 50,
        'pattern': r'^[a-zA-Z0-9_-]+$'
    },
    'password': {
        'min_length': 8,
        'max_length': 100,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_number': True,
        'require_special': True
    },
    'email': {
        'max_length': 120,
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    }
} 