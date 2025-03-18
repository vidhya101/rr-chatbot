from datetime import datetime
import uuid
from models.db import db

class Dashboard(db.Model):
    """Dashboard model for storing user dashboards"""
    __tablename__ = 'dashboards'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    layout = db.Column(db.JSON, nullable=True)  # Layout configuration
    is_public = db.Column(db.Boolean, default=False)
    category = db.Column(db.String(50), default='personal')  # 'personal', 'work', 'analytics', etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    charts = db.relationship('Chart', backref='dashboard', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, user_id, title, description=None, layout=None, is_public=False, category='personal'):
        self.user_id = user_id
        self.title = title
        self.description = description
        self.layout = layout or {}
        self.is_public = is_public
        self.category = category
    
    def to_dict(self):
        """Convert dashboard to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'layout': self.layout,
            'is_public': self.is_public,
            'category': self.category,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'chart_count': len(self.charts)
        }
    
    def __repr__(self):
        return f'<Dashboard {self.title}>'


class Chart(db.Model):
    """Chart model for storing dashboard charts"""
    __tablename__ = 'charts'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dashboard_id = db.Column(db.String(36), db.ForeignKey('dashboards.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    chart_type = db.Column(db.String(50), nullable=False)  # 'bar', 'line', 'pie', etc.
    data_source = db.Column(db.JSON, nullable=False)  # Data source configuration
    config = db.Column(db.JSON, nullable=True)  # Chart configuration
    position = db.Column(db.JSON, nullable=True)  # Position in dashboard
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, dashboard_id, title, chart_type, data_source, description=None, config=None, position=None):
        self.dashboard_id = dashboard_id
        self.title = title
        self.description = description
        self.chart_type = chart_type
        self.data_source = data_source
        self.config = config or {}
        self.position = position or {'x': 0, 'y': 0, 'w': 6, 'h': 4}
    
    def to_dict(self):
        """Convert chart to dictionary for API responses"""
        return {
            'id': self.id,
            'dashboard_id': self.dashboard_id,
            'title': self.title,
            'description': self.description,
            'chart_type': self.chart_type,
            'data_source': self.data_source,
            'config': self.config,
            'position': self.position,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Chart {self.title}>' 