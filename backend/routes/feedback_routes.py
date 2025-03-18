from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
import json
import time
from datetime import datetime
import uuid
import sqlite3
from utils.db_utils import log_info, log_error, get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
feedback_routes = Blueprint('feedback_routes', __name__)

@feedback_routes.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for an AI response"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': "No data provided"
            }), 400
        
        # Extract feedback data
        message_id = data.get('messageId')
        chat_id = data.get('chatId')
        feedback_type = data.get('type')  # 'positive' or 'negative'
        comment = data.get('comment', '')
        user_id = data.get('userId', request.headers.get('X-User-ID', 'anonymous'))
        
        # Validate required fields
        if not message_id or not feedback_type:
            return jsonify({
                'success': False,
                'error': "Missing required fields: messageId and type are required"
            }), 400
        
        # Validate feedback type
        if feedback_type not in ['positive', 'negative']:
            return jsonify({
                'success': False,
                'error': "Invalid feedback type. Must be 'positive' or 'negative'"
            }), 400
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Generate ID
        feedback_id = str(uuid.uuid4())
        
        # Insert feedback
        cursor.execute('''
        INSERT INTO feedback (
            id, 
            message_id, 
            chat_id, 
            user_id, 
            feedback_type, 
            comment, 
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            message_id,
            chat_id,
            user_id,
            feedback_type,
            comment,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Log feedback
        log_info(
            user_id=user_id,
            source='feedback_routes.submit_feedback',
            message=f"Feedback submitted: {feedback_type}",
            details={
                'feedback_id': feedback_id,
                'message_id': message_id,
                'chat_id': chat_id,
                'feedback_type': feedback_type,
                'has_comment': bool(comment)
            }
        )
        
        return jsonify({
            'success': True,
            'message': "Feedback submitted successfully",
            'feedbackId': feedback_id
        }), 201
    
    except Exception as e:
        error_message = f"Error submitting feedback: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=request.headers.get('X-User-ID', 'anonymous'),
            source='feedback_routes.submit_feedback',
            message='Error submitting feedback',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@feedback_routes.route('/feedback/stats', methods=['GET'])
@jwt_required()
def get_feedback_stats():
    """Get feedback statistics"""
    try:
        # Get current user
        current_user_id = get_jwt_identity()
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get feedback stats
        cursor.execute('''
        SELECT 
            feedback_type, 
            COUNT(*) as count 
        FROM feedback 
        GROUP BY feedback_type
        ''')
        
        stats = {}
        for row in cursor.fetchall():
            stats[row['feedback_type']] = row['count']
        
        # Get recent feedback
        cursor.execute('''
        SELECT 
            id,
            message_id,
            chat_id,
            user_id,
            feedback_type,
            comment,
            created_at
        FROM feedback
        ORDER BY created_at DESC
        LIMIT 10
        ''')
        
        recent = []
        for row in cursor.fetchall():
            recent.append({
                'id': row['id'],
                'messageId': row['message_id'],
                'chatId': row['chat_id'],
                'userId': row['user_id'],
                'type': row['feedback_type'],
                'comment': row['comment'],
                'createdAt': row['created_at']
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent': recent
        }), 200
    
    except Exception as e:
        error_message = f"Error getting feedback stats: {str(e)}"
        logger.error(error_message)
        
        # Log error
        log_error(
            user_id=get_jwt_identity() if get_jwt_identity() else 'anonymous',
            source='feedback_routes.get_feedback_stats',
            message='Error getting feedback stats',
            details={'error': str(e)}
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500 