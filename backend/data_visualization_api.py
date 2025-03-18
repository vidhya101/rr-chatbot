from flask import Blueprint, request, jsonify, current_app, send_file, abort, g
from flask_cors import CORS
from flask_jwt_extended import jwt_required, get_jwt_identity
import os
import logging
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
import uuid
import traceback
from marshmallow import Schema, fields, validate, ValidationError
from flask_caching import Cache

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Create blueprint
data_viz_api = Blueprint('data_visualization_api', __name__)

# Enable CORS for this blueprint
CORS(data_viz_api)

# Initialize cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3600})

# Schema definitions for request validation
class HealthCheckSchema(Schema):
    pass

class LoadDataSchema(Schema):
    file = fields.Raw(required=True)
    file_type = fields.Str(validate=validate.OneOf(['csv', 'excel', 'json']), required=False, default='csv')
    options = fields.Dict(required=False)

class QuerySchema(Schema):
    query = fields.Str(required=True)
    parameters = fields.Dict(required=False)

class ExecuteQuerySchema(Schema):
    query = fields.Str(required=True)
    parameters = fields.Dict(required=False)
    timeout = fields.Int(required=False, default=30)

class VisualizationSchema(Schema):
    data_source = fields.Str(required=True)
    chart_type = fields.Str(validate=validate.OneOf(['bar', 'line', 'pie', 'scatter', 'heatmap', 'histogram', 'box', 'auto']), required=True)
    title = fields.Str(required=False)
    x_axis = fields.Str(required=False)
    y_axis = fields.Str(required=False)
    filters = fields.Dict(required=False)
    options = fields.Dict(required=False)

class InsightSchema(Schema):
    data_source = fields.Str(required=True)
    insight_type = fields.Str(validate=validate.OneOf(['correlation', 'outliers', 'trends', 'summary', 'auto']), required=True)
    options = fields.Dict(required=False)

class PreviewDataSchema(Schema):
    data_source = fields.Str(required=True)
    limit = fields.Int(required=False, default=100)
    offset = fields.Int(required=False, default=0)
    filters = fields.Dict(required=False)

class ExportDashboardSchema(Schema):
    dashboard_id = fields.Str(required=True)
    format = fields.Str(validate=validate.OneOf(['pdf', 'png', 'html', 'json']), required=True)
    options = fields.Dict(required=False)

# Utility functions
def validate_request(schema_class):
    """Decorator to validate request data against a schema"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            schema = schema_class()
            
            # Get request data based on content type and method
            if request.method == 'GET':
                data = request.args.to_dict()
            elif request.content_type == 'application/json':
                data = request.get_json() or {}
            elif request.content_type.startswith('multipart/form-data'):
                data = request.form.to_dict()
                if 'file' in request.files:
                    data['file'] = request.files['file']
            else:
                data = {}
            
            try:
                # Validate request data
                validated_data = schema.load(data)
                g.validated_data = validated_data
            except ValidationError as err:
                return jsonify({
                    'success': False,
                    'error': 'Validation error',
                    'details': err.messages
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def error_handler(f):
    """Decorator to handle exceptions and return appropriate error responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return appropriate error response
            if isinstance(e, ValidationError):
                return jsonify({
                    'success': False,
                    'error': 'Validation error',
                    'details': e.messages
                }), 400
            elif isinstance(e, FileNotFoundError):
                return jsonify({
                    'success': False,
                    'error': 'File not found',
                    'message': str(e)
                }), 404
            else:
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'message': str(e)
                }), 500
    return decorated_function

def rate_limit(limit=100, per=60):
    """Decorator to implement rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get user identity
            user_id = get_jwt_identity() if hasattr(g, 'jwt_identity') else request.remote_addr
            
            # Check rate limit
            key = f"rate_limit:{user_id}:{f.__name__}"
            current = cache.get(key) or 0
            
            if current >= limit:
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {limit} requests per {per} seconds'
                }), 429
            
            # Increment counter
            cache.set(key, current + 1, timeout=per)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_upload_path():
    """Get the upload directory path"""
    upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

def load_dataframe(file_path, file_type='csv', options=None):
    """Load a dataframe from a file"""
    options = options or {}
    
    if file_type == 'csv':
        return pd.read_csv(file_path, **options)
    elif file_type == 'excel':
        return pd.read_excel(file_path, **options)
    elif file_type == 'json':
        return pd.read_json(file_path, **options)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# API Endpoints
@data_viz_api.route('/health', methods=['GET'])
@error_handler
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'version': current_app.config.get('API_VERSION', '1.0.0')
    })

@data_viz_api.route('/load-data', methods=['POST'])
@jwt_required()
@validate_request(LoadDataSchema)
@error_handler
def load_data():
    """Load data from a file"""
    # Get validated data
    data = g.validated_data
    
    # Get file from request
    file = data['file']
    file_type = data.get('file_type', 'csv')
    options = data.get('options', {})
    
    # Save file
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(get_upload_path(), f"{file_id}_{filename}")
    file.save(file_path)
    
    # Load data to validate it
    try:
        df = load_dataframe(file_path, file_type, options)
        
        # Get basic stats
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_path': file_path,
            'file_name': filename,
            'file_type': file_type,
            'stats': stats
        })
    except Exception as e:
        # Remove file if loading fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

@data_viz_api.route('/process-query', methods=['POST'])
@jwt_required()
@validate_request(QuerySchema)
@error_handler
@rate_limit(limit=50, per=60)
def process_query():
    """Process a natural language query"""
    # Get validated data
    data = g.validated_data
    
    # Get query and parameters
    query = data['query']
    parameters = data.get('parameters', {})
    
    # Process query (this would be implemented in a service)
    # For now, we'll just return a mock response
    return jsonify({
        'success': True,
        'query': query,
        'processed_query': f"SELECT * FROM data WHERE {parameters.get('condition', 'true')}",
        'parameters': parameters
    })

@data_viz_api.route('/execute-query', methods=['POST'])
@jwt_required()
@validate_request(ExecuteQuerySchema)
@error_handler
@rate_limit(limit=20, per=60)
def execute_query():
    """Execute a SQL query"""
    # Get validated data
    data = g.validated_data
    
    # Get query, parameters, and timeout
    query = data['query']
    parameters = data.get('parameters', {})
    timeout = data.get('timeout', 30)
    
    # Execute query (this would be implemented in a service)
    # For now, we'll just return a mock response
    return jsonify({
        'success': True,
        'query': query,
        'parameters': parameters,
        'results': [
            {'id': 1, 'name': 'Example 1', 'value': 100},
            {'id': 2, 'name': 'Example 2', 'value': 200},
            {'id': 3, 'name': 'Example 3', 'value': 300}
        ],
        'execution_time': 0.1
    })

@data_viz_api.route('/visualize', methods=['POST'])
@jwt_required()
@validate_request(VisualizationSchema)
@error_handler
@cache.cached(timeout=3600, key_prefix=lambda: f"viz_{request.get_json()}")
def visualize():
    """Generate a visualization"""
    # Get validated data
    data = g.validated_data
    
    # Get visualization parameters
    data_source = data['data_source']
    chart_type = data['chart_type']
    title = data.get('title', f"{chart_type.capitalize()} Chart")
    x_axis = data.get('x_axis')
    y_axis = data.get('y_axis')
    filters = data.get('filters', {})
    options = data.get('options', {})
    
    # Check if data source exists
    if not os.path.exists(data_source):
        return jsonify({
            'success': False,
            'error': 'Data source not found',
            'message': f"File not found: {data_source}"
        }), 404
    
    # Load data
    try:
        file_type = data_source.split('.')[-1].lower()
        if file_type == 'xlsx' or file_type == 'xls':
            file_type = 'excel'
        
        df = load_dataframe(data_source, file_type)
        
        # Apply filters if any
        for column, value in filters.items():
            if column in df.columns:
                df = df[df[column] == value]
        
        # Generate visualization (this would be implemented in a service)
        # For now, we'll just return a mock response
        viz_id = str(uuid.uuid4())
        viz_path = f"/visualizations/{viz_id}.png"
        
        return jsonify({
            'success': True,
            'visualization_id': viz_id,
            'visualization_url': viz_path,
            'title': title,
            'chart_type': chart_type,
            'data_source': data_source,
            'parameters': {
                'x_axis': x_axis,
                'y_axis': y_axis,
                'filters': filters,
                'options': options
            },
            'created_at': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Visualization error',
            'message': str(e)
        }), 500

@data_viz_api.route('/insights', methods=['POST'])
@jwt_required()
@validate_request(InsightSchema)
@error_handler
def get_insights():
    """Get insights from data"""
    # Get validated data
    data = g.validated_data
    
    # Get insight parameters
    data_source = data['data_source']
    insight_type = data['insight_type']
    options = data.get('options', {})
    
    # Check if data source exists
    if not os.path.exists(data_source):
        return jsonify({
            'success': False,
            'error': 'Data source not found',
            'message': f"File not found: {data_source}"
        }), 404
    
    # Load data
    try:
        file_type = data_source.split('.')[-1].lower()
        if file_type == 'xlsx' or file_type == 'xls':
            file_type = 'excel'
        
        df = load_dataframe(data_source, file_type)
        
        # Generate insights based on type
        insights = []
        
        if insight_type == 'correlation' or insight_type == 'auto':
            # Calculate correlations for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr().round(2)
                
                # Find strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr = corr_matrix.iloc[i, j]
                        
                        if abs(corr) > 0.7:  # Strong correlation threshold
                            insights.append({
                                'type': 'correlation',
                                'description': f"Strong {'positive' if corr > 0 else 'negative'} correlation ({corr}) between {col1} and {col2}",
                                'strength': abs(corr),
                                'columns': [col1, col2]
                            })
        
        if insight_type == 'outliers' or insight_type == 'auto':
            # Detect outliers in numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            for col in numeric_df.columns:
                q1 = numeric_df[col].quantile(0.25)
                q3 = numeric_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
                
                if len(outliers) > 0:
                    insights.append({
                        'type': 'outliers',
                        'description': f"Found {len(outliers)} outliers in column {col}",
                        'column': col,
                        'count': len(outliers),
                        'percentage': round(len(outliers) / len(df) * 100, 2)
                    })
        
        if insight_type == 'summary' or insight_type == 'auto':
            # Generate summary statistics
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                summary = numeric_df.describe().round(2).to_dict()
                
                insights.append({
                    'type': 'summary',
                    'description': 'Summary statistics for numeric columns',
                    'statistics': summary
                })
        
        return jsonify({
            'success': True,
            'data_source': data_source,
            'insight_type': insight_type,
            'insights': insights,
            'created_at': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Insight error',
            'message': str(e)
        }), 500

@data_viz_api.route('/preview-data', methods=['POST'])
@jwt_required()
@validate_request(PreviewDataSchema)
@error_handler
def preview_data():
    """Preview data from a source"""
    # Get validated data
    data = g.validated_data
    
    # Get preview parameters
    data_source = data['data_source']
    limit = data.get('limit', 100)
    offset = data.get('offset', 0)
    filters = data.get('filters', {})
    
    # Check if data source exists
    if not os.path.exists(data_source):
        return jsonify({
            'success': False,
            'error': 'Data source not found',
            'message': f"File not found: {data_source}"
        }), 404
    
    # Load data
    try:
        file_type = data_source.split('.')[-1].lower()
        if file_type == 'xlsx' or file_type == 'xls':
            file_type = 'excel'
        
        df = load_dataframe(data_source, file_type)
        
        # Apply filters if any
        for column, value in filters.items():
            if column in df.columns:
                df = df[df[column] == value]
        
        # Get total count
        total_count = len(df)
        
        # Apply pagination
        df = df.iloc[offset:offset+limit]
        
        # Convert to dict
        records = df.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'data_source': data_source,
            'records': records,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'columns': df.columns.tolist(),
            'data_types': {col: str(df[col].dtype) for col in df.columns}
        })
    except Exception as e:
        logger.error(f"Error previewing data: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Preview error',
            'message': str(e)
        }), 500

@data_viz_api.route('/export-dashboard', methods=['POST'])
@jwt_required()
@validate_request(ExportDashboardSchema)
@error_handler
def export_dashboard():
    """Export a dashboard"""
    # Get validated data
    data = g.validated_data
    
    # Get export parameters
    dashboard_id = data['dashboard_id']
    export_format = data['format']
    options = data.get('options', {})
    
    # Generate export (this would be implemented in a service)
    # For now, we'll just return a mock response
    export_id = str(uuid.uuid4())
    export_path = f"/exports/{export_id}.{export_format}"
    
    return jsonify({
        'success': True,
        'dashboard_id': dashboard_id,
        'export_id': export_id,
        'export_url': export_path,
        'format': export_format,
        'created_at': datetime.utcnow().isoformat()
    })

def init_app(app):
    """Initialize the API with the Flask app"""
    # Initialize cache
    cache.init_app(app)
    
    # Register blueprint
    app.register_blueprint(data_viz_api, url_prefix='/api/data-viz')
    
    # Log initialization
    logger.info("Data Visualization API initialized") 