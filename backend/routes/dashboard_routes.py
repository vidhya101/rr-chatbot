from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.db import db
from models.dashboard import Dashboard, Chart
import uuid

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboards', methods=['GET'])
@jwt_required()
def get_dashboards():
    """Get all dashboards for the current user"""
    current_user_id = get_jwt_identity()
    
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    category = request.args.get('category')
    
    # Query dashboards
    query = Dashboard.query.filter_by(user_id=current_user_id)
    
    # Filter by category if specified
    if category:
        query = query.filter_by(category=category)
    
    # Apply sorting
    sort_by = request.args.get('sort_by', 'updated_at')
    sort_order = request.args.get('sort_order', 'desc')
    
    if sort_order == 'desc':
        query = query.order_by(getattr(Dashboard, sort_by).desc())
    else:
        query = query.order_by(getattr(Dashboard, sort_by).asc())
    
    # Paginate results
    dashboards_pagination = query.paginate(page=page, per_page=per_page)
    
    # Format response
    dashboards = [dashboard.to_dict() for dashboard in dashboards_pagination.items]
    
    return jsonify({
        'dashboards': dashboards,
        'total': dashboards_pagination.total,
        'pages': dashboards_pagination.pages,
        'page': page,
        'per_page': per_page
    }), 200


@dashboard_bp.route('/dashboards/<dashboard_id>', methods=['GET'])
@jwt_required()
def get_dashboard(dashboard_id):
    """Get a specific dashboard by ID"""
    current_user_id = get_jwt_identity()
    
    # Find dashboard
    dashboard = Dashboard.query.filter_by(id=dashboard_id).first()
    if not dashboard:
        return jsonify({'error': 'Dashboard not found'}), 404
    
    # Check if user has access to dashboard
    if dashboard.user_id != current_user_id and not dashboard.is_public:
        return jsonify({'error': 'Access denied'}), 403
    
    # Get charts
    charts = [chart.to_dict() for chart in dashboard.charts]
    
    # Format response
    dashboard_data = dashboard.to_dict()
    dashboard_data['charts'] = charts
    
    return jsonify(dashboard_data), 200


@dashboard_bp.route('/dashboards', methods=['POST'])
@jwt_required()
def create_dashboard():
    """Create a new dashboard"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    if 'title' not in data:
        return jsonify({'error': 'Title is required'}), 400
    
    # Create dashboard
    try:
        dashboard = Dashboard(
            user_id=current_user_id,
            title=data['title'],
            description=data.get('description'),
            layout=data.get('layout'),
            is_public=data.get('is_public', False),
            category=data.get('category', 'personal')
        )
        
        db.session.add(dashboard)
        db.session.commit()
        
        return jsonify({
            'message': 'Dashboard created successfully',
            'dashboard': dashboard.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/dashboards/<dashboard_id>', methods=['PUT'])
@jwt_required()
def update_dashboard(dashboard_id):
    """Update a dashboard"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Find dashboard
    dashboard = Dashboard.query.filter_by(id=dashboard_id, user_id=current_user_id).first()
    if not dashboard:
        return jsonify({'error': 'Dashboard not found'}), 404
    
    # Update fields
    if 'title' in data:
        dashboard.title = data['title']
    
    if 'description' in data:
        dashboard.description = data['description']
    
    if 'layout' in data:
        dashboard.layout = data['layout']
    
    if 'is_public' in data:
        dashboard.is_public = data['is_public']
    
    if 'category' in data:
        dashboard.category = data['category']
    
    # Save changes
    try:
        db.session.commit()
        
        return jsonify({
            'message': 'Dashboard updated successfully',
            'dashboard': dashboard.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/dashboards/<dashboard_id>', methods=['DELETE'])
@jwt_required()
def delete_dashboard(dashboard_id):
    """Delete a dashboard"""
    current_user_id = get_jwt_identity()
    
    # Find dashboard
    dashboard = Dashboard.query.filter_by(id=dashboard_id, user_id=current_user_id).first()
    if not dashboard:
        return jsonify({'error': 'Dashboard not found'}), 404
    
    # Delete dashboard
    try:
        db.session.delete(dashboard)
        db.session.commit()
        
        return jsonify({
            'message': 'Dashboard deleted successfully'
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/dashboards/<dashboard_id>/charts', methods=['POST'])
@jwt_required()
def create_chart(dashboard_id):
    """Create a new chart in a dashboard"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Find dashboard
    dashboard = Dashboard.query.filter_by(id=dashboard_id, user_id=current_user_id).first()
    if not dashboard:
        return jsonify({'error': 'Dashboard not found'}), 404
    
    # Validate required fields
    required_fields = ['title', 'chart_type', 'data_source']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Create chart
    try:
        chart = Chart(
            dashboard_id=dashboard.id,
            title=data['title'],
            description=data.get('description'),
            chart_type=data['chart_type'],
            data_source=data['data_source'],
            config=data.get('config'),
            position=data.get('position')
        )
        
        db.session.add(chart)
        db.session.commit()
        
        return jsonify({
            'message': 'Chart created successfully',
            'chart': chart.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/charts/<chart_id>', methods=['PUT'])
@jwt_required()
def update_chart(chart_id):
    """Update a chart"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Find chart
    chart = Chart.query.join(Dashboard).filter(
        Chart.id == chart_id,
        Dashboard.user_id == current_user_id
    ).first()
    
    if not chart:
        return jsonify({'error': 'Chart not found'}), 404
    
    # Update fields
    if 'title' in data:
        chart.title = data['title']
    
    if 'description' in data:
        chart.description = data['description']
    
    if 'chart_type' in data:
        chart.chart_type = data['chart_type']
    
    if 'data_source' in data:
        chart.data_source = data['data_source']
    
    if 'config' in data:
        chart.config = data['config']
    
    if 'position' in data:
        chart.position = data['position']
    
    # Save changes
    try:
        db.session.commit()
        
        return jsonify({
            'message': 'Chart updated successfully',
            'chart': chart.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/charts/<chart_id>', methods=['DELETE'])
@jwt_required()
def delete_chart(chart_id):
    """Delete a chart"""
    current_user_id = get_jwt_identity()
    
    # Find chart
    chart = Chart.query.join(Dashboard).filter(
        Chart.id == chart_id,
        Dashboard.user_id == current_user_id
    ).first()
    
    if not chart:
        return jsonify({'error': 'Chart not found'}), 404
    
    # Delete chart
    try:
        db.session.delete(chart)
        db.session.commit()
        
        return jsonify({
            'message': 'Chart deleted successfully'
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    """Get dashboard statistics"""
    current_user_id = get_jwt_identity()
    
    # Get dashboard count
    dashboard_count = Dashboard.query.filter_by(user_id=current_user_id).count()
    
    # Get chart count
    chart_count = Chart.query.join(Dashboard).filter(Dashboard.user_id == current_user_id).count()
    
    # Get category counts
    category_counts = db.session.query(
        Dashboard.category, db.func.count(Dashboard.id)
    ).filter(
        Dashboard.user_id == current_user_id
    ).group_by(Dashboard.category).all()
    
    # Format category counts
    categories = {category: count for category, count in category_counts}
    
    # Get chart type counts
    chart_type_counts = db.session.query(
        Chart.chart_type, db.func.count(Chart.id)
    ).join(Dashboard).filter(
        Dashboard.user_id == current_user_id
    ).group_by(Chart.chart_type).all()
    
    # Format chart type counts
    chart_types = {chart_type: count for chart_type, count in chart_type_counts}
    
    return jsonify({
        'dashboard_count': dashboard_count,
        'chart_count': chart_count,
        'categories': categories,
        'chart_types': chart_types
    }), 200 