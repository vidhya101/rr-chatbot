from datetime import datetime
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from models.db import db

class User(db.Model):
    """User model for authentication and profile information"""
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    profile_picture = db.Column(db.String(255), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    role = db.Column(db.String(20), default='user')  # 'user', 'admin', etc.
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Settings
    theme = db.Column(db.String(20), default='light')  # 'light', 'dark'
    language = db.Column(db.String(10), default='en')  # 'en', 'es', etc.
    notifications_enabled = db.Column(db.Boolean, default=True)
    default_model = db.Column(db.String(50), default='digitalogy')
    
    # Relationships
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')
    files = db.relationship('File', backref='user', lazy=True, cascade='all, delete-orphan')
    dashboards = db.relationship('Dashboard', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, username, email, password, **kwargs):
        self.username = username
        self.email = email
        self.set_password(password)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'profile_picture': self.profile_picture,
            'bio': self.bio,
            'role': self.role,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'theme': self.theme,
            'language': self.language,
            'notifications_enabled': self.notifications_enabled,
            'default_model': self.default_model
        }
    
    def __repr__(self):
        return f'<User {self.username}>' 