from datetime import datetime
import uuid
import os
from models.db import db

class File(db.Model):
    """File model for storing uploaded files"""
    __tablename__ = 'files'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    mime_type = db.Column(db.String(100), nullable=False)
    is_processed = db.Column(db.Boolean, default=False)
    processing_status = db.Column(db.String(50), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    processing_error = db.Column(db.Text, nullable=True)
    file_metadata = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, user_id, filename, original_filename, file_path, file_type, file_size, mime_type, file_metadata=None):
        self.user_id = user_id
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_type = file_type
        self.file_size = file_size
        self.mime_type = mime_type
        self.file_metadata = file_metadata or {}
    
    def get_full_path(self, upload_folder):
        """Get the full path to the file"""
        return os.path.join(upload_folder, self.file_path)
    
    def to_dict(self):
        """Convert file to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'is_processed': self.is_processed,
            'processing_status': self.processing_status,
            'file_metadata': self.file_metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<File {self.filename}>' 