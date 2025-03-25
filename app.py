from flask import Flask, request, jsonify
from flask_cors import CORS
from blueprints.data_visualization import data_viz_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(data_viz_bp, url_prefix='/api/data')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

@data_viz_bp.route('/datasets/<dataset_id>/interactive-visualize', methods=['POST'])
def create_interactive_visualization(dataset_id):
    """
    Create an interactive visualization with advanced features
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            visualization_type:
              type: string
              description: Type of visualization to create
            params:
              type: object
              description: Additional parameters for the visualization
    responses:
      200:
        description: Interactive visualization details
      404:
        description: Dataset not found
    """
    try:
        data = request.get_json()
        visualization_type = data.get('visualization_type')
        params = data.get('params', {})
        
        if not visualization_type:
            return jsonify({'success': False, 'error': 'Visualization type must be specified'}), 400
        
        result = data_viz_api.createInteractiveVisualization(dataset_id, visualization_type, params)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating interactive visualization: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/advanced-analyze', methods=['POST'])
def perform_advanced_analysis(dataset_id):
    """
    Perform advanced statistical analysis on a dataset
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            analysis_config:
              type: object
              description: Configuration for the analysis
    responses:
      200:
        description: Advanced analysis results
      404:
        description: Dataset not found
    """
    try:
        data = request.get_json()
        analysis_config = data.get('analysis_config', {})
        
        result = data_viz_api.performAdvancedAnalysis(dataset_id, analysis_config)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error performing advanced analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/report', methods=['POST'])
def generate_data_report(dataset_id):
    """
    Generate a comprehensive data report
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            report_config:
              type: object
              description: Configuration for the report
    responses:
      200:
        description: Report details
      404:
        description: Dataset not found
    """
    try:
        data = request.get_json()
        report_config = data.get('report_config', {})
        
        result = data_viz_api.generateDataReport(dataset_id, report_config)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating data report: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/dashboard', methods=['POST'])
def create_dashboard(dataset_id):
    """
    Create a dashboard with multiple visualizations
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            visualizations:
              type: array
              description: Array of visualization configurations
    responses:
      200:
        description: Dashboard details
      404:
        description: Dataset not found
    """
    try:
        data = request.get_json()
        visualizations = data.get('visualizations', [])
        
        if not visualizations:
            return jsonify({'success': False, 'error': 'At least one visualization must be specified'}), 400
        
        result = data_viz_api.createDashboard(dataset_id, visualizations)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualizations/<visualization_id>/export', methods=['GET'])
def export_visualization(dataset_id, visualization_id):
    """
    Export a visualization in various formats
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: visualization_id
        in: path
        type: string
        required: true
        description: ID of the visualization
      - name: format
        in: query
        type: string
        enum: [png, svg, pdf, html]
        default: png
        description: Export format
    responses:
      200:
        description: Export details
      404:
        description: Dataset or visualization not found
    """
    try:
        format = request.args.get('format', 'png')
        
        result = data_viz_api.exportVisualization(dataset_id, visualization_id, format)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error exporting visualization: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualization-configs', methods=['POST'])
def save_visualization_config(dataset_id):
    """
    Save a visualization configuration for later use
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            config:
              type: object
              description: Visualization configuration to save
    responses:
      200:
        description: Saved configuration details
      404:
        description: Dataset not found
    """
    try:
        data = request.get_json()
        visualization_config = data.get('config', {})
        
        if not visualization_config:
            return jsonify({'success': False, 'error': 'Configuration must be specified'}), 400
        
        result = data_viz_api.saveVisualizationConfig(dataset_id, visualization_config)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error saving visualization config: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/visualization-templates', methods=['GET'])
def get_visualization_templates():
    """
    Get available visualization templates
    ---
    responses:
      200:
        description: List of available templates
    """
    try:
        result = data_viz_api.getVisualizationTemplates()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting visualization templates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/templates/<template_id>/apply', methods=['POST'])
def apply_visualization_template(dataset_id, template_id):
    """
    Apply a visualization template to a dataset
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: template_id
        in: path
        type: string
        required: true
        description: ID of the template to apply
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            params:
              type: object
              description: Additional parameters for template customization
    responses:
      200:
        description: Applied template details
      404:
        description: Dataset or template not found
    """
    try:
        data = request.get_json()
        params = data.get('params', {})
        
        result = data_viz_api.applyVisualizationTemplate(dataset_id, template_id, params)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error applying visualization template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualizations/<visualization_id>/share', methods=['POST'])
def share_visualization(dataset_id, visualization_id):
    """
    Share a visualization
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: visualization_id
        in: path
        type: string
        required: true
        description: ID of the visualization
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            share_config:
              type: object
              description: Sharing configuration
    responses:
      200:
        description: Sharing details
      404:
        description: Dataset or visualization not found
    """
    try:
        data = request.get_json()
        share_config = data.get('share_config', {})
        
        if not share_config:
            return jsonify({'success': False, 'error': 'Sharing configuration must be specified'}), 400
        
        result = data_viz_api.shareVisualization(dataset_id, visualization_id, share_config)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error sharing visualization: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualizations/<visualization_id>/analytics', methods=['GET'])
def get_visualization_analytics(dataset_id, visualization_id):
    """
    Get analytics for a visualization
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: visualization_id
        in: path
        type: string
        required: true
        description: ID of the visualization
    responses:
      200:
        description: Visualization analytics
      404:
        description: Dataset or visualization not found
    """
    try:
        result = data_viz_api.getVisualizationAnalytics(dataset_id, visualization_id)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting visualization analytics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualizations/<visualization_id>/schedule', methods=['POST'])
def schedule_visualization_update(dataset_id, visualization_id):
    """
    Schedule updates for a visualization
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: visualization_id
        in: path
        type: string
        required: true
        description: ID of the visualization
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            schedule_config:
              type: object
              description: Schedule configuration
    responses:
      200:
        description: Schedule details
      404:
        description: Dataset or visualization not found
    """
    try:
        data = request.get_json()
        schedule_config = data.get('schedule_config', {})
        
        if not schedule_config:
            return jsonify({'success': False, 'error': 'Schedule configuration must be specified'}), 400
        
        result = data_viz_api.scheduleVisualizationUpdate(dataset_id, visualization_id, schedule_config)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error scheduling visualization update: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualizations/<visualization_id>/schedule', methods=['GET'])
def get_scheduled_updates(dataset_id, visualization_id):
    """
    Get scheduled updates for a visualization
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: visualization_id
        in: path
        type: string
        required: true
        description: ID of the visualization
    responses:
      200:
        description: List of scheduled updates
      404:
        description: Dataset or visualization not found
    """
    try:
        result = data_viz_api.getScheduledUpdates(dataset_id, visualization_id)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting scheduled updates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/visualizations/<visualization_id>/schedule/<schedule_id>', methods=['DELETE'])
def cancel_scheduled_update(dataset_id, visualization_id, schedule_id):
    """
    Cancel a scheduled visualization update
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
      - name: visualization_id
        in: path
        type: string
        required: true
        description: ID of the visualization
      - name: schedule_id
        in: path
        type: string
        required: true
        description: ID of the schedule to cancel
    responses:
      200:
        description: Cancellation status
      404:
        description: Dataset, visualization, or schedule not found
    """
    try:
        result = data_viz_api.cancelScheduledUpdate(dataset_id, visualization_id, schedule_id)
        
        if not result['success']:
            return jsonify(result), 404 if 'not found' in result.get('error', '') else 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error canceling scheduled update: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

from prometheus_client import generate_latest
import psutil
import os
from datetime import datetime

@app.route('/api/system/health')
def system_health():
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get database connection count (example using SQLAlchemy)
        active_connections = db.session.execute('SELECT count(*) FROM pg_stat_activity').scalar()
        
        # Check database health
        try:
            db.session.execute('SELECT 1')
            database_status = 'ok'
        except Exception as e:
            database_status = 'error'
            logger.error(f"Database health check failed: {str(e)}")
        
        # Check system health
        system_status = 'ok'
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            system_status = 'warning'
        
        health_data = {
            'status': system_status,
            'database': database_status,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'active_connections': active_connections,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return jsonify(health_data)
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return jsonify({'error': 'Failed to get system health'}), 500

@app.route('/metrics')
def metrics():
    try:
        return generate_latest()
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        return jsonify({'error': 'Failed to generate metrics'}), 500 