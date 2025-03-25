from flask import Blueprint, jsonify
import psutil
from datetime import datetime
import logging
from models.db import db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
monitoring_bp = Blueprint('monitoring', __name__)

@monitoring_bp.route('/api/system/health', methods=['GET'])
def system_health():
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get database connection count
        try:
            db.session.execute('SELECT 1')
            database_status = 'ok'
            active_connections = db.session.execute('SELECT count(*) FROM pg_stat_activity').scalar()
        except Exception as e:
            database_status = 'error'
            active_connections = 0
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

@monitoring_bp.route('/api/system/metrics/detailed', methods=['GET'])
def detailed_metrics():
    try:
        metrics = {
            'network': psutil.net_io_counters()._asdict(),
            'disk_io': psutil.disk_io_counters()._asdict(),
            'memory_detailed': psutil.virtual_memory()._asdict(),
            'swap': psutil.swap_memory()._asdict(),
            'cpu_times': psutil.cpu_times()._asdict(),
            'cpu_freq': psutil.cpu_freq()._asdict() if hasattr(psutil, 'cpu_freq') else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting detailed metrics: {str(e)}")
        return jsonify({'error': 'Failed to get detailed metrics'}), 500

@monitoring_bp.route('/api/system/performance', methods=['GET'])
def performance_metrics():
    try:
        # Mock performance data (replace with actual metrics in production)
        performance_data = {
            'slow_queries': [
                {
                    'query': 'SELECT * FROM large_table',
                    'calls': 150,
                    'total_time': 45.2,
                    'mean_time': 0.3
                }
            ],
            'endpoint_performance': {
                'endpoints': [
                    {
                        'path': '/api/data',
                        'method': 'GET',
                        'avg_duration': 0.15,
                        'requests': 1000
                    }
                ]
            }
        }
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({'error': 'Failed to get performance metrics'}), 500

@monitoring_bp.route('/api/system/alerts', methods=['GET'])
def system_alerts():
    try:
        # Mock alerts data (replace with actual alerts in production)
        alerts_data = {
            'alerts': [
                {
                    'type': 'warning',
                    'component': 'CPU',
                    'message': 'High CPU usage detected',
                    'timestamp': datetime.utcnow().isoformat()
                }
            ]
        }
        return jsonify(alerts_data)
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        return jsonify({'error': 'Failed to get alerts'}), 500 