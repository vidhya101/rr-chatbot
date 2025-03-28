from flask import Blueprint, request, jsonify, current_app, send_file
import os
import logging
import json
from services.visualization_service import generate_visualization, generate_dashboard
from utils.db_utils import log_info, log_error
from utils.validation import validate_file_path, validate_visualization_params
from utils.retry import retry
from utils.circuit_breaker import CircuitBreaker
from utils.exceptions import ValidationError, VisualizationError, ExternalServiceError
from werkzeug.utils import secure_filename
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
from services.ml_service import MLService
from pathlib import Path
from services.data_processing_service import data_processing_service
from services.cache_service import cache_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
visualization_bp = Blueprint('visualization', __name__)

# Create circuit breaker for external service calls
circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)

# Initialize ML service
ml_service = MLService()

# In-memory storage for visualizations and dashboards (replace with database in production)
visualizations = {}
dashboards = {}
models = {}

ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'parquet'}
UPLOAD_FOLDER = 'uploads'
VISUALIZATION_FOLDER = 'static/visualizations'
DASHBOARD_FOLDER = 'static/dashboards'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@visualization_bp.route('/visualize', methods=['POST'])
@retry(max_attempts=3, delay=1.0, backoff=2.0)
@circuit_breaker
def visualize():
    """Generate a visualization for a dataset"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        file_path = data.get('file_path')
        viz_type = data.get('type', 'auto')
        params = data.get('params', {})
        
        # Validate inputs
        try:
            validate_file_path(file_path)
            validate_visualization_params(params)
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
        
        # Generate visualization
        result = generate_visualization(file_path, viz_type, params)
        
        if result.get('success', False):
            return jsonify(result), 200
        else:
            log_error(
                source='visualization',
                message=f"Error generating visualization: {result.get('error', 'Unknown error')}",
                user_id=request.headers.get('X-User-ID', 'anonymous'),
                details={
                    'file_path': file_path,
                    'viz_type': viz_type,
                    'params': params
                }
            )
            return jsonify(result), 500
    
    except ValidationError as e:
        logger.error(f"Validation error in visualization route: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Validation error",
            "message": str(e)
        }), 400
    except ExternalServiceError as e:
        logger.error(f"External service error in visualization route: {str(e)}")
        return jsonify({
            "success": False,
            "error": "External service error",
            "message": str(e)
        }), 503
    except Exception as e:
        logger.error(f"Error in visualization route: {str(e)}")
        log_error(
            source='visualization',
            message=f"Error in visualization route: {str(e)}",
            user_id=request.headers.get('X-User-ID', 'anonymous')
        )
        return jsonify({
            "success": False,
            "error": "Failed to generate visualization",
            "message": str(e)
        }), 500

@visualization_bp.route('/dashboard', methods=['POST'])
@retry(max_attempts=3, delay=1.0, backoff=2.0)
@circuit_breaker
def dashboard():
    """Generate a comprehensive dashboard for a dataset"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        file_path = data.get('file_path')
        
        # Validate inputs
        try:
            validate_file_path(file_path)
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
        
        # Log the request
        log_info(
            source='visualization',
            message=f"Dashboard request for {file_path}",
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            details={
                'file_path': file_path
            }
        )
        
        # Generate dashboard
        result = generate_dashboard(file_path)
        
        if result.get('success', False):
            return jsonify(result), 200
        else:
            log_error(
                source='visualization',
                message=f"Error generating dashboard: {result.get('error', 'Unknown error')}",
                user_id=request.headers.get('X-User-ID', 'anonymous'),
                details={
                    'file_path': file_path
                }
            )
            return jsonify(result), 500
    
    except ValidationError as e:
        logger.error(f"Validation error in dashboard route: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Validation error",
            "message": str(e)
        }), 400
    except ExternalServiceError as e:
        logger.error(f"External service error in dashboard route: {str(e)}")
        return jsonify({
            "success": False,
            "error": "External service error",
            "message": str(e)
        }), 503
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        log_error(
            source='visualization',
            message=f"Error in dashboard route: {str(e)}",
            user_id=request.headers.get('X-User-ID', 'anonymous')
        )
        return jsonify({
            "success": False,
            "error": "Failed to generate dashboard",
            "message": str(e)
        }), 500

@visualization_bp.route('/visualizations/<filename>', methods=['GET'])
@retry(max_attempts=2, delay=0.5, backoff=1.5)
def get_visualization(filename):
    """Get a visualization image by filename"""
    try:
        # Get visualization folder
        viz_folder = os.path.join(current_app.config.get('UPLOAD_FOLDER', 'uploads'), 'visualizations')
        
        # Construct file path
        file_path = os.path.join(viz_folder, filename)
        
        # Validate file path
        try:
            validate_file_path(file_path)
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 404
        
        # Return the file
        return send_file(file_path, mimetype='image/png')
    
    except ValidationError as e:
        logger.error(f"Validation error retrieving visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Validation error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error retrieving visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve visualization",
            "message": str(e)
        }), 500

@visualization_bp.route('/api/visualizations', methods=['GET'])
async def get_visualizations():
    """Get all visualizations with caching"""
    try:
        # Try to get from cache
        cached_visualizations = cache_service.get('all_visualizations')
        if cached_visualizations:
            return jsonify(cached_visualizations)

        # If not in cache, get from storage and cache
        visualizations_path = Path(VISUALIZATION_FOLDER)
        visualizations = []
        for viz_file in visualizations_path.glob('*.html'):
            viz_id = viz_file.stem
            viz_meta_file = viz_file.with_suffix('.json')
            if viz_meta_file.exists():
                with open(viz_meta_file, 'r') as f:
                    metadata = json.load(f)
                    visualizations.append({
                        'id': viz_id,
                        **metadata
                    })

        # Cache the results
        cache_service.set('all_visualizations', visualizations, expiration=3600)
        return jsonify(visualizations)

    except Exception as e:
        logger.error(f"Error getting visualizations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/visualizations', methods=['POST'])
async def create_visualization():
    """Create a new visualization with optimized data processing"""
    try:
        data = request.get_json()
        name = data.get('name')
        viz_type = data.get('type')
        data_file = data.get('dataFile')
        x_axis = data.get('xAxis')
        y_axis = data.get('yAxis')
        
        if not all([name, viz_type, data_file, x_axis, y_axis]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Load and process data
        file_path = os.path.join(UPLOAD_FOLDER, data_file)
        df = await data_processing_service.load_data(file_path)
        
        # Generate visualization
        viz_id = str(uuid.uuid4())
        fig = None

        if viz_type == 'bar':
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif viz_type == 'line':
            fig = px.line(df, x=x_axis, y=y_axis)
        elif viz_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif viz_type == 'pie':
            fig = px.pie(df, values=y_axis, names=x_axis)
        else:
            return jsonify({'error': 'Invalid visualization type'}), 400

        # Save visualization
        viz_path = os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.html")
        fig.write_html(viz_path)

        # Save metadata
        metadata = {
            'name': name,
            'type': viz_type,
            'dataFile': data_file,
            'xAxis': x_axis,
            'yAxis': y_axis,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.json"), 'w') as f:
            json.dump(metadata, f)

        # Clear visualizations cache
        cache_service.delete('all_visualizations')

        return jsonify({
            'id': viz_id,
            **metadata
        })

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/visualizations/<viz_id>', methods=['GET'])
async def get_visualization_by_id(viz_id):
    """Get a specific visualization by ID"""
    try:
        # Try to get from cache
        cache_key = f'visualization_{viz_id}'
        cached_viz = cache_service.get(cache_key)
        if cached_viz:
            return jsonify(cached_viz)

        # If not in cache, get from storage
        viz_path = Path(VISUALIZATION_FOLDER) / f'{viz_id}.html'
        if not viz_path.exists():
            return jsonify({
                "success": False,
                "error": "Visualization not found"
            }), 404

        # Read visualization data
        with open(viz_path, 'r') as f:
            viz_data = f.read()

        # Cache the result
        cache_service.set(cache_key, viz_data, expiration=3600)  # Cache for 1 hour

        return jsonify({
            "success": True,
            "data": viz_data
        })

    except Exception as e:
        logger.error(f"Error retrieving visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve visualization",
            "message": str(e)
        }), 500

@visualization_bp.route('/api/visualizations/<viz_id>', methods=['DELETE'])
async def delete_visualization(viz_id):
    """Delete a visualization and clear cache"""
    try:
        viz_path = os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.html")
        meta_path = os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.json")

        if not os.path.exists(viz_path):
            return jsonify({'error': 'Visualization not found'}), 404

        # Delete files
        os.remove(viz_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

        # Clear caches
        cache_service.delete(f'visualization:{viz_id}')
        cache_service.delete('all_visualizations')

        return jsonify({'message': 'Visualization deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/dashboards', methods=['GET'])
async def get_dashboards():
    """Get all dashboards with caching"""
    try:
        # Try to get from cache
        cached_dashboards = cache_service.get('all_dashboards')
        if cached_dashboards:
            return jsonify(cached_dashboards)

        dashboards_path = Path(DASHBOARD_FOLDER)
        dashboards = []
        for dash_file in dashboards_path.glob('*.html'):
            dash_id = dash_file.stem
            dash_meta_file = dash_file.with_suffix('.json')
            if dash_meta_file.exists():
                with open(dash_meta_file, 'r') as f:
                    metadata = json.load(f)
                    dashboards.append({
                        'id': dash_id,
                        **metadata
                    })

        # Cache the results
        cache_service.set('all_dashboards', dashboards, expiration=3600)
        return jsonify(dashboards)

    except Exception as e:
        logger.error(f"Error getting dashboards: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/dashboards', methods=['POST'])
async def create_dashboard():
    """Create a new dashboard with optimized processing"""
    try:
        data = request.get_json()
        name = data.get('name')
        visualizations = data.get('visualizations', [])

        if not name or not visualizations:
            return jsonify({'error': 'Missing required fields'}), 400

        # Verify all visualizations exist
        for viz_id in visualizations:
            viz_path = os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.html")
            if not os.path.exists(viz_path):
                return jsonify({'error': f'Visualization {viz_id} not found'}), 404

        # Create dashboard
        dash_id = str(uuid.uuid4())
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{name}</title>
            <style>
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    padding: 20px;
                }}
                .visualization-container {{
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                    background: white;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-grid">
        """

        for viz_id in visualizations:
            with open(os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.html"), 'r') as f:
                viz_content = f.read()
                dashboard_html += f"""
                <div class="visualization-container">
                    {viz_content}
                </div>
                """

        dashboard_html += """
            </div>
        </body>
        </html>
        """

        # Save dashboard
        dash_path = os.path.join(DASHBOARD_FOLDER, f"{dash_id}.html")
        with open(dash_path, 'w') as f:
            f.write(dashboard_html)

        # Save metadata
        metadata = {
            'name': name,
            'visualizations': visualizations,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(DASHBOARD_FOLDER, f"{dash_id}.json"), 'w') as f:
            json.dump(metadata, f)

        # Clear dashboards cache
        cache_service.delete('all_dashboards')

        return jsonify({
            'id': dash_id,
            **metadata
        })

    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/dashboards/<dashboard_id>', methods=['GET'])
async def get_dashboard(dashboard_id):
    """Get a specific dashboard with caching"""
    try:
        # Try to get from cache
        cached_dash = cache_service.get(f'dashboard:{dashboard_id}')
        if cached_dash:
            return jsonify(cached_dash)

        dash_path = os.path.join(DASHBOARD_FOLDER, f"{dashboard_id}.html")
        meta_path = os.path.join(DASHBOARD_FOLDER, f"{dashboard_id}.json")

        if not os.path.exists(dash_path) or not os.path.exists(meta_path):
            return jsonify({'error': 'Dashboard not found'}), 404

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        result = {
            'id': dashboard_id,
            'html_path': dash_path,
            **metadata
        }

        # Cache the result
        cache_service.set(f'dashboard:{dashboard_id}', result, expiration=3600)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/dashboards/<dashboard_id>', methods=['DELETE'])
async def delete_dashboard(dashboard_id):
    """Delete a dashboard and clear cache"""
    try:
        dash_path = os.path.join(DASHBOARD_FOLDER, f"{dashboard_id}.html")
        meta_path = os.path.join(DASHBOARD_FOLDER, f"{dashboard_id}.json")

        if not os.path.exists(dash_path):
            return jsonify({'error': 'Dashboard not found'}), 404

        # Delete files
        os.remove(dash_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

        # Clear caches
        cache_service.delete(f'dashboard:{dashboard_id}')
        cache_service.delete('all_dashboards')

        return jsonify({'message': 'Dashboard deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/visualizations/advanced', methods=['POST'])
async def create_advanced_visualization():
    """Create an advanced visualization with ML insights"""
    try:
        data = request.get_json()
        name = data.get('name')
        viz_type = data.get('type')
        data_file = data.get('dataFile')
        features = data.get('features', [])
        target = data.get('target')
        analysis_type = data.get('analysisType', 'correlation')
        
        if not all([name, viz_type, data_file]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Load and process data
        file_path = os.path.join(UPLOAD_FOLDER, data_file)
        df = await data_processing_service.load_data(file_path)
        
        # Generate visualization based on analysis type
        viz_id = str(uuid.uuid4())
        fig = None

        if analysis_type == 'correlation':
            corr_matrix = df[features].corr()
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Correlation"),
                          x=features,
                          y=features)
        elif analysis_type == 'distribution':
            fig = px.histogram(df, x=target, marginal="box")
        elif analysis_type == 'scatter_matrix':
            fig = px.scatter_matrix(df, dimensions=features)
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400

        # Save visualization
        viz_path = os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.html")
        fig.write_html(viz_path)

        # Save metadata
        metadata = {
            'name': name,
            'type': viz_type,
            'dataFile': data_file,
            'features': features,
            'target': target,
            'analysisType': analysis_type,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(VISUALIZATION_FOLDER, f"{viz_id}.json"), 'w') as f:
            json.dump(metadata, f)

        # Clear visualizations cache
        cache_service.delete('all_visualizations')

        return jsonify({
            'id': viz_id,
            **metadata
        })

    except Exception as e:
        logger.error(f"Error creating advanced visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/analysis/time-series', methods=['POST'])
def analyze_time_series():
    """Analyze and forecast time series data"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        file_id = data.get('fileId')
        date_col = data.get('dateColumn')
        value_col = data.get('valueColumn')
        periods = data.get('forecastPeriods', 30)

        # Load data
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Perform forecast
        result = ml_service.forecast_time_series(df, date_col, value_col, periods)
        
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/analysis/text', methods=['POST'])
def analyze_text():
    """Perform advanced text analysis"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        text = data.get('text')
        analysis_type = data.get('type', 'all')

        result = ml_service.analyze_text(text, analysis_type)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/models/train', methods=['POST'])
def train_model():
    """Train ML model on data"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        file_id = data.get('fileId')
        target_col = data.get('targetColumn')
        model_type = data.get('modelType', 'auto')
        params = data.get('parameters')

        # Load data
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Preprocess data
        df = ml_service.preprocess_data(df, target_col)

        # Train model
        result = ml_service.train_model(df, target_col, model_type, params)
        
        # Store model info
        model_id = str(uuid.uuid4())
        models[model_id] = {
            'id': model_id,
            'type': result['model_type'],
            'score': result['score'],
            'created_at': datetime.now().isoformat()
        }

        return jsonify({
            'model': models[model_id]
        }), 201

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/dashboards/advanced', methods=['POST'])
def create_advanced_dashboard():
    """Create advanced dashboard with ML insights"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        file_id = data.get('fileId')
        dashboard_type = data.get('type')
        params = data.get('params', {})

        # Load data
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Create visualizations based on dashboard type
        dashboard_id = str(uuid.uuid4())
        visualizations_data = []

        if dashboard_type == 'comprehensive':
            # Add correlation matrix
            corr_viz = ml_service.create_advanced_visualization(df, 'correlation_matrix', params)
            visualizations_data.append({
                'type': 'correlation_matrix',
                'plot': corr_viz
            })

            # Add distribution plots
            dist_viz = ml_service.create_advanced_visualization(df, 'distribution', params)
            visualizations_data.append({
                'type': 'distribution',
                'plot': dist_viz
            })

            # Add time series analysis if date column exists
            if params.get('date_col') and params.get('value_col'):
                ts_viz = ml_service.create_advanced_visualization(df, 'time_series', params)
                visualizations_data.append({
                    'type': 'time_series',
                    'plot': ts_viz
                })

        elif dashboard_type == 'network':
            network_viz = ml_service.create_advanced_visualization(df, 'network', params)
            visualizations_data.append({
                'type': 'network',
                'plot': network_viz
            })

        # Create dashboard HTML
        dashboard_html = []
        dashboard_html.append('<html><head><title>Advanced Dashboard</title></head><body>')
        dashboard_html.append(f'<h1>Advanced Analytics Dashboard</h1>')
        
        # Create a grid layout
        dashboard_html.append('<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; padding: 20px;">')
        
        for viz in visualizations_data:
            dashboard_html.append(f'<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">')
            dashboard_html.append(f'<h3>{viz["type"].replace("_", " ").title()}</h3>')
            dashboard_html.append(f'<div id="{viz["type"]}"></div>')
            dashboard_html.append(f'<script>{viz["plot"]}</script>')
            dashboard_html.append('</div>')
        
        dashboard_html.append('</div></body></html>')

        # Save dashboard
        dashboard_path = os.path.join('static', 'dashboards', f'{dashboard_id}.html')
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        with open(dashboard_path, 'w') as f:
            f.write('\n'.join(dashboard_html))

        dashboard = {
            'id': dashboard_id,
            'type': dashboard_type,
            'visualizations': visualizations_data,
            'created_at': datetime.now().isoformat(),
            'dashboard_url': f'/static/dashboards/{dashboard_id}.html'
        }

        dashboards[dashboard_id] = dashboard
        return jsonify({'dashboard': dashboard}), 201

    except Exception as e:
        logger.error(f"Error creating advanced dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Ensure required directories exist
os.makedirs(os.path.join('static', 'visualizations'), exist_ok=True)
os.makedirs(os.path.join('static', 'dashboards'), exist_ok=True) 