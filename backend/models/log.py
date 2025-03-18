from datetime import datetime, timedelta
import uuid
from models.db import db
import json

class Log(db.Model):
    """Log model for storing system logs, errors, and user activity"""
    __tablename__ = 'logs'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    level = db.Column(db.String(20), nullable=False)  # 'info', 'warning', 'error', 'debug'
    source = db.Column(db.String(50), nullable=False)  # 'system', 'user', 'api', 'ollama', 'mistral', etc.
    message = db.Column(db.Text, nullable=False)
    details = db.Column(db.JSON, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, level, source, message, user_id=None, details=None, ip_address=None, user_agent=None):
        self.level = level
        self.source = source
        self.message = message
        self.user_id = user_id
        self.details = details or {}
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    @classmethod
    def log_info(cls, source, message, user_id=None, details=None, ip_address=None, user_agent=None):
        """Log an info message"""
        log = cls('info', source, message, user_id, details, ip_address, user_agent)
        db.session.add(log)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error logging info: {str(e)}")
        return log
    
    @classmethod
    def log_warning(cls, source, message, user_id=None, details=None, ip_address=None, user_agent=None):
        """Log a warning message"""
        log = cls('warning', source, message, user_id, details, ip_address, user_agent)
        db.session.add(log)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error logging warning: {str(e)}")
        return log
    
    @classmethod
    def log_error(cls, source, message, user_id=None, details=None, ip_address=None, user_agent=None):
        """Log an error message"""
        log = cls('error', source, message, user_id, details, ip_address, user_agent)
        db.session.add(log)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error logging error: {str(e)}")
        return log
    
    @classmethod
    def log_debug(cls, source, message, user_id=None, details=None, ip_address=None, user_agent=None):
        """Log a debug message"""
        log = cls('debug', source, message, user_id, details, ip_address, user_agent)
        db.session.add(log)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error logging debug: {str(e)}")
        return log
    
    @classmethod
    def purge_old_logs(cls, days=30):
        """Purge logs older than the specified number of days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        try:
            # Archive logs before deleting
            cls.archive_logs(cutoff_date)
            
            # Delete old logs
            old_logs = cls.query.filter(cls.created_at < cutoff_date).delete()
            db.session.commit()
            return old_logs
        except Exception as e:
            db.session.rollback()
            print(f"Error purging old logs: {str(e)}")
            return 0
    
    @classmethod
    def archive_logs(cls, cutoff_date):
        """Archive logs older than the cutoff date"""
        try:
            # Get logs to archive
            logs_to_archive = cls.query.filter(cls.created_at < cutoff_date).all()
            
            if not logs_to_archive:
                return 0
            
            # Create archive
            archive = LogArchive(
                start_date=min(log.created_at for log in logs_to_archive),
                end_date=max(log.created_at for log in logs_to_archive),
                log_count=len(logs_to_archive),
                data=json.dumps([{
                    'id': log.id,
                    'user_id': log.user_id,
                    'level': log.level,
                    'source': log.source,
                    'message': log.message,
                    'details': log.details,
                    'ip_address': log.ip_address,
                    'user_agent': log.user_agent,
                    'created_at': log.created_at.isoformat()
                } for log in logs_to_archive])
            )
            
            db.session.add(archive)
            db.session.commit()
            return len(logs_to_archive)
        except Exception as e:
            db.session.rollback()
            print(f"Error archiving logs: {str(e)}")
            return 0
    
    def to_dict(self):
        """Convert log to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'level': self.level,
            'source': self.source,
            'message': self.message,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Log {self.id} {self.level}>'


class LogArchive(db.Model):
    """Log archive model for storing archived logs"""
    __tablename__ = 'log_archives'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    log_count = db.Column(db.Integer, nullable=False)
    data = db.Column(db.Text, nullable=False)  # JSON string of archived logs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, start_date, end_date, log_count, data):
        self.start_date = start_date
        self.end_date = end_date
        self.log_count = log_count
        self.data = data
    
    def to_dict(self):
        """Convert log archive to dictionary for API responses"""
        return {
            'id': self.id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'log_count': self.log_count,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<LogArchive {self.id} {self.log_count} logs>'


class UserSession(db.Model):
    """User session model for tracking user sessions"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    token = db.Column(db.String(255), nullable=False, unique=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    device_info = db.Column(db.JSON, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def __init__(self, user_id, token, expires_at, ip_address=None, user_agent=None, device_info=None):
        self.user_id = user_id
        self.token = token
        self.expires_at = expires_at
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.device_info = device_info or {}
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        db.session.commit()
    
    def is_expired(self):
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def invalidate(self):
        """Invalidate session"""
        self.is_active = False
        db.session.commit()
    
    @classmethod
    def cleanup_expired_sessions(cls):
        """Clean up expired sessions"""
        try:
            expired = cls.query.filter(cls.expires_at < datetime.utcnow()).delete()
            db.session.commit()
            return expired
        except Exception as e:
            db.session.rollback()
            print(f"Error cleaning up expired sessions: {str(e)}")
            return 0
    
    def to_dict(self):
        """Convert session to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'is_active': self.is_active,
            'last_activity': self.last_activity.isoformat(),
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }
    
    def __repr__(self):
        return f'<UserSession {self.id} {self.user_id}>' 