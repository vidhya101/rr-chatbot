import os
import logging
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from threading import Lock
from functools import wraps
from flask import request, g, current_app
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from models.user import User
from database import get_db
from utils.retry import retry, RetryContext
from utils.circuit_breaker import CircuitBreaker
from utils.exceptions import (
    AuthenticationError, AuthorizationError, RateLimitError,
    TokenError, ValidationError, ResourceCleanupError, DatabaseError
)

try:
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserRole:
    """Role configuration"""
    name: str
    permissions: List[str]
    priority: int

@dataclass
class AuditLog:
    """Audit log entry"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    status: str
    ip_address: str
    user_agent: str
    details: Dict[str, Any]

class AuthService:
    """Service for authentication, authorization, and security"""
    
    def __init__(self, redis_url: Optional[str] = None, db_session: Optional[Session] = None):
        """Initialize authentication service"""
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis = None
        if REDIS_AVAILABLE:
            try:
                self.redis = Redis.from_url(self.redis_url, decode_responses=True)
                self.redis.ping()  # Test connection
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}. Running without Redis.")
        
        self.db = db_session or get_db()
        
        # JWT configuration
        self.jwt_secret = os.getenv('JWT_SECRET') or secrets.token_hex(32)
        self.access_token_expiry = timedelta(minutes=15)
        self.refresh_token_expiry = timedelta(days=7)
        
        # Rate limiting configuration (in-memory fallback when Redis is not available)
        self.rate_limits = {
            'login': {'calls': 5, 'period': 300},  # 5 calls per 5 minutes
            'refresh_token': {'calls': 10, 'period': 300},  # 10 calls per 5 minutes
            'api': {'calls': 100, 'period': 60}  # 100 calls per minute
        }
        self.rate_limit_store = {}
        
        # Role configuration
        self.roles = {
            'admin': UserRole(
                name='admin',
                permissions=['read', 'write', 'delete', 'manage_users'],
                priority=100
            ),
            'moderator': UserRole(
                name='moderator',
                permissions=['read', 'write'],
                priority=50
            ),
            'user': UserRole(
                name='user',
                permissions=['read'],
                priority=10
            )
        }
        
        # Session management (in-memory fallback when Redis is not available)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.sessions_lock = Lock()
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0
        )
    
    def hash_password(self, password: str) -> bytes:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt)
    
    def verify_password(self, password: str, hashed: bytes) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hashed)
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and generate tokens"""
        try:
            # Check rate limit
            if not self._check_rate_limit(f"login:{username}"):
                raise RateLimitError("Too many login attempts")
            
            # Get user from database (implement your user storage)
            user = self._get_user(username)
            if not user or not self.verify_password(password, user['password']):
                self._log_audit(
                    user_id=username,
                    action='login',
                    resource='auth',
                    status='failed',
                    details={'reason': 'Invalid credentials'}
                )
                raise AuthenticationError("Invalid credentials")
            
            # Generate tokens
            access_token = self._generate_token(user, 'access')
            refresh_token = self._generate_token(user, 'refresh')
            
            # Store refresh token
            self._store_refresh_token(refresh_token, user['id'])
            
            # Create session
            session_id = secrets.token_hex(16)
            self._create_session(user['id'], session_id)
            
            self._log_audit(
                user_id=user['id'],
                action='login',
                resource='auth',
                status='success',
                details={'session_id': session_id}
            )
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'session_id': session_id,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role']
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Generate new access token using refresh token"""
        try:
            # Check rate limit
            if not self._check_rate_limit(f"refresh_token:{refresh_token}"):
                raise RateLimitError("Too many refresh attempts")
            
            # Verify refresh token
            user_id = self._get_refresh_token(refresh_token)
            if not user_id:
                raise TokenError("Invalid refresh token")
            
            # Get user
            user = self._get_user_by_id(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Generate new access token
            access_token = self._generate_token(user, 'access')
            
            self._log_audit(
                user_id=user['id'],
                action='refresh_token',
                resource='auth',
                status='success',
                details={}
            )
            
            return {'access_token': access_token}
            
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            raise TokenError(f"Token refresh failed: {str(e)}")
    
    def require_auth(self, required_permissions: Optional[List[str]] = None):
        """Decorator for requiring authentication and permissions"""
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                # Get token from header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    raise AuthenticationError("No token provided")
                
                token = auth_header.split(' ')[1]
                
                # Verify token and get user
                try:
                    payload = jwt.decode(
                        token,
                        self.jwt_secret,
                        algorithms=['HS256']
                    )
                    user = self._get_user_by_id(payload['sub'])
                    if not user:
                        raise AuthenticationError("User not found")
                    
                    # Check if session is active
                    if not self._is_session_active(user['id'], payload.get('session_id')):
                        raise AuthenticationError("Session expired")
                    
                    # Check permissions
                    if required_permissions:
                        role = self.roles.get(user['role'])
                        if not role or not all(p in role.permissions for p in required_permissions):
                            raise AuthorizationError("Insufficient permissions")
                    
                    # Store user in request context
                    g.user = user
                    
                    return f(*args, **kwargs)
                    
                except jwt.ExpiredSignatureError:
                    raise TokenError("Token expired")
                except jwt.InvalidTokenError:
                    raise TokenError("Invalid token")
                
            return decorated
        return decorator
    
    def _generate_token(self, user: Dict[str, Any], token_type: str) -> str:
        """Generate JWT token"""
        now = datetime.utcnow()
        expiry = (
            now + self.refresh_token_expiry
            if token_type == 'refresh'
            else now + self.access_token_expiry
        )
        
        payload = {
            'sub': user['id'],
            'username': user['username'],
            'role': user['role'],
            'type': token_type,
            'iat': now,
            'exp': expiry
        }
        
        if token_type == 'access':
            payload['session_id'] = user.get('session_id')
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _store_refresh_token(self, refresh_token: str, user_id: str) -> None:
        """Store refresh token with fallback to memory if Redis is not available"""
        expiry = int(self.refresh_token_expiry.total_seconds())
        if self.redis:
            self.redis.setex(f"refresh_token:{refresh_token}", expiry, user_id)
        else:
            # In-memory storage with expiration
            expiry_time = datetime.utcnow() + self.refresh_token_expiry
            self.rate_limit_store[f"refresh_token:{refresh_token}"] = {
                'user_id': user_id,
                'expires_at': expiry_time
            }
    
    def _get_refresh_token(self, refresh_token: str) -> Optional[str]:
        """Get refresh token with fallback to memory if Redis is not available"""
        if self.redis:
            return self.redis.get(f"refresh_token:{refresh_token}")
        else:
            # Check in-memory storage
            token_data = self.rate_limit_store.get(f"refresh_token:{refresh_token}")
            if token_data and datetime.utcnow() < token_data['expires_at']:
                return token_data['user_id']
            return None
    
    def _check_rate_limit(self, key: str) -> bool:
        """Check rate limit with fallback to memory if Redis is not available"""
        limit_key = key.split(':')[0]  # Extract limit type (login, refresh_token, api)
        limit_config = self.rate_limits.get(limit_key, self.rate_limits['api'])
        
        if self.redis:
            # Use Redis for rate limiting
            current = self.redis.get(f"rate_limit:{key}")
            if not current:
                self.redis.setex(f"rate_limit:{key}", limit_config['period'], 1)
                return True
            
            count = int(current)
            if count >= limit_config['calls']:
                return False
            
            self.redis.incr(f"rate_limit:{key}")
            return True
        else:
            # In-memory rate limiting
            now = datetime.utcnow()
            if key not in self.rate_limit_store:
                self.rate_limit_store[key] = {
                    'count': 1,
                    'reset_at': now + timedelta(seconds=limit_config['period'])
                }
                return True
            
            limit_data = self.rate_limit_store[key]
            if now >= limit_data['reset_at']:
                # Reset counter if period has passed
                limit_data['count'] = 1
                limit_data['reset_at'] = now + timedelta(seconds=limit_config['period'])
                return True
            
            if limit_data['count'] >= limit_config['calls']:
                return False
            
            limit_data['count'] += 1
            return True
    
    def _create_session(self, user_id: str, session_id: str) -> None:
        """Create new user session"""
        with self.sessions_lock:
            self.active_sessions[user_id] = {
                'session_id': session_id,
                'created_at': datetime.utcnow(),
                'last_activity': datetime.utcnow()
            }
    
    def _is_session_active(self, user_id: str, session_id: str) -> bool:
        """Check if session is active"""
        with self.sessions_lock:
            session = self.active_sessions.get(user_id)
            if not session or session['session_id'] != session_id:
                return False
            
            # Update last activity
            session['last_activity'] = datetime.utcnow()
            return True
    
    def invalidate_session(self, user_id: str) -> None:
        """Invalidate user session"""
        with self.sessions_lock:
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
    
    def _log_audit(
        self, user_id: str, action: str, resource: str,
        status: str, details: Dict[str, Any]
    ) -> None:
        """Log audit event"""
        try:
            log_entry = AuditLog(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                action=action,
                resource=resource,
                status=status,
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                details=details
            )
            
            # Store audit log (implement your storage solution)
            self._store_audit_log(log_entry)
            
        except Exception as e:
            logger.error(f"Audit logging error: {str(e)}")
    
    def _store_audit_log(self, log_entry: AuditLog) -> None:
        """Store audit log entry in database"""
        try:
            # Convert to dictionary for JSON storage
            log_data = {
                'timestamp': log_entry.timestamp.isoformat(),
                'user_id': log_entry.user_id,
                'action': log_entry.action,
                'resource': log_entry.resource,
                'status': log_entry.status,
                'ip_address': log_entry.ip_address,
                'user_agent': log_entry.user_agent,
                'details': log_entry.details
            }
            
            # Store in Redis for quick access
            self.redis.lpush('audit_logs', str(log_data))
            self.redis.ltrim('audit_logs', 0, 9999)  # Keep last 10000 logs
            
            # Log to application logger
            logger.info(f"Audit log: {log_data}")
            
        except Exception as e:
            logger.error(f"Error storing audit log: {str(e)}")
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        try:
            with self.sessions_lock:
                current_time = datetime.utcnow()
                expired = [
                    user_id for user_id, session in self.active_sessions.items()
                    if (current_time - session['last_activity']) > self.access_token_expiry
                ]
                
                for user_id in expired:
                    del self.active_sessions[user_id]
                    
        except Exception as e:
            logger.error(f"Session cleanup error: {str(e)}")
            raise ResourceCleanupError(f"Failed to clean up sessions: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup_resources()
    
    def _cleanup_resources(self) -> None:
        """Clean up authentication resources"""
        try:
            # Clear sessions
            with self.sessions_lock:
                self.active_sessions.clear()
            
            # Close Redis connection
            if self.redis:
                self.redis.close()
            
            # Close database session
            self.db.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise ResourceCleanupError(f"Failed to clean up auth resources: {str(e)}")
    
    def _get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            user = self.db.query(User).filter(
                User.username == username,
                User.is_active == True
            ).first()
            
            return user.to_dict() if user else None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by username: {str(e)}")
            raise DatabaseError(f"Failed to get user: {str(e)}")
    
    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = self.db.query(User).filter(
                User.id == user_id,
                User.is_active == True
            ).first()
            
            return user.to_dict() if user else None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by ID: {str(e)}")
            raise DatabaseError(f"Failed to get user: {str(e)}")
    
    def create_user(
        self, username: str, email: str, password: str,
        role: str = 'user', **kwargs
    ) -> Dict[str, Any]:
        """Create new user"""
        try:
            # Check if username or email already exists
            existing = self.db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            if existing:
                raise ValidationError(
                    "Username or email already exists"
                    if existing.username == username
                    else "Email already exists"
                )
            
            # Create user
            user = User(
                username=username,
                email=email,
                password=self.hash_password(password),
                role=role,
                **kwargs
            )
            
            self.db.add(user)
            self.db.commit()
            
            self._log_audit(
                user_id=str(user.id),
                action='create_user',
                resource='user',
                status='success',
                details={'username': username, 'role': role}
            )
            
            return user.to_dict()
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error creating user: {str(e)}")
            raise DatabaseError(f"Failed to create user: {str(e)}")
    
    def update_user(
        self, user_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user information"""
        try:
            user = self.db.query(User).filter(
                User.id == user_id,
                User.is_active == True
            ).first()
            
            if not user:
                raise ValidationError("User not found")
            
            # Update allowed fields
            allowed_fields = {
                'email', 'first_name', 'last_name',
                'is_active', 'is_verified', 'profile'
            }
            
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(user, field, value)
            
            self.db.commit()
            
            self._log_audit(
                user_id=str(user.id),
                action='update_user',
                resource='user',
                status='success',
                details={'updated_fields': list(updates.keys())}
            )
            
            return user.to_dict()
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error updating user: {str(e)}")
            raise DatabaseError(f"Failed to update user: {str(e)}")
    
    def delete_user(self, user_id: str) -> bool:
        """Soft delete user"""
        try:
            user = self.db.query(User).filter(
                User.id == user_id,
                User.is_active == True
            ).first()
            
            if not user:
                raise ValidationError("User not found")
            
            # Soft delete
            user.is_active = False
            self.db.commit()
            
            # Invalidate session
            self.invalidate_session(str(user.id))
            
            self._log_audit(
                user_id=str(user.id),
                action='delete_user',
                resource='user',
                status='success',
                details={}
            )
            
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error deleting user: {str(e)}")
            raise DatabaseError(f"Failed to delete user: {str(e)}")
    
    def change_password(
        self, user_id: str, current_password: str, new_password: str
    ) -> bool:
        """Change user password"""
        try:
            user = self.db.query(User).filter(
                User.id == user_id,
                User.is_active == True
            ).first()
            
            if not user:
                raise ValidationError("User not found")
            
            # Verify current password
            if not self.verify_password(current_password, user.password):
                raise ValidationError("Current password is incorrect")
            
            # Update password
            user.password = self.hash_password(new_password)
            self.db.commit()
            
            # Invalidate all sessions
            self.invalidate_session(str(user.id))
            
            self._log_audit(
                user_id=str(user.id),
                action='change_password',
                resource='user',
                status='success',
                details={}
            )
            
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error changing password: {str(e)}")
            raise DatabaseError(f"Failed to change password: {str(e)}")
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (admin only)"""
        try:
            users = self.db.query(User).filter(
                User.is_active == True
            ).all()
            
            return [user.to_dict() for user in users]
            
        except SQLAlchemyError as e:
            logger.error(f"Database error listing users: {str(e)}")
            raise DatabaseError(f"Failed to list users: {str(e)}") 