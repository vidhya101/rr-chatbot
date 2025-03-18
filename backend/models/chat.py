from datetime import datetime
import uuid
from models.db import db

class Chat(db.Model):
    """Chat model for storing conversation sessions"""
    __tablename__ = 'chats'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(255), nullable=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    model = db.Column(db.String(50), nullable=False, default='digitalogy')
    is_pinned = db.Column(db.Boolean, default=False)
    is_archived = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, user_id, model='digitalogy', title=None):
        self.user_id = user_id
        self.model = model
        self.title = title
    
    def to_dict(self):
        """Convert chat to dictionary for API responses"""
        return {
            'id': self.id,
            'title': self.title,
            'user_id': self.user_id,
            'model': self.model,
            'is_pinned': self.is_pinned,
            'is_archived': self.is_archived,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self.messages),
            'last_message': self.messages[-1].to_dict() if self.messages else None
        }
    
    def __repr__(self):
        return f'<Chat {self.id}>'


class Message(db.Model):
    """Message model for storing individual messages in a chat"""
    __tablename__ = 'messages'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = db.Column(db.String(36), db.ForeignKey('chats.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = db.Column(db.Text, nullable=False)
    tokens = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, chat_id, role, content, tokens=None):
        self.chat_id = chat_id
        self.role = role
        self.content = content
        self.tokens = tokens
    
    def to_dict(self):
        """Convert message to dictionary for API responses"""
        return {
            'id': self.id,
            'chat_id': self.chat_id,
            'role': self.role,
            'content': self.content,
            'tokens': self.tokens,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Message {self.id}>' 