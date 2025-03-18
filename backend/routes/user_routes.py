import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from models.db import db
from models.user import User
from models.chat import Chat, Message

# Create blueprint
user_bp = Blueprint('user', __name__)

@user_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    current_user_id = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify(user.to_dict()), 200


@user_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Find user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Update fields
    updatable_fields = ['first_name', 'last_name', 'bio']
    for field in updatable_fields:
        if field in data:
            setattr(user, field, data[field])
    
    # Save changes
    try:
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@user_bp.route('/profile/picture', methods=['POST'])
@jwt_required()
def upload_profile_picture():
    """Upload profile picture"""
    current_user_id = get_jwt_identity()
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is an image
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return jsonify({'error': 'File must be an image (PNG, JPG, JPEG, GIF)'}), 400
    
    try:
        # Find user
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Secure filename and generate unique name
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"profile_{uuid.uuid4().hex}.{file_extension}"
        
        # Create profile pictures directory if it doesn't exist
        profile_pics_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'profile_pictures')
        os.makedirs(profile_pics_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(profile_pics_dir, unique_filename)
        file.save(file_path)
        
        # Delete old profile picture if exists
        if user.profile_picture:
            old_pic_path = os.path.join(current_app.config['UPLOAD_FOLDER'], user.profile_picture)
            if os.path.exists(old_pic_path):
                os.remove(old_pic_path)
        
        # Update user profile picture
        user.profile_picture = os.path.join('profile_pictures', unique_filename)
        db.session.commit()
        
        return jsonify({
            'message': 'Profile picture updated successfully',
            'user': user.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@user_bp.route('/settings', methods=['GET'])
@jwt_required()
def get_settings():
    """Get user settings"""
    current_user_id = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Return settings
    settings = {
        'theme': user.theme,
        'language': user.language,
        'notifications_enabled': user.notifications_enabled,
        'default_model': user.default_model
    }
    
    return jsonify(settings), 200


@user_bp.route('/settings', methods=['PUT'])
@jwt_required()
def update_settings():
    """Update user settings"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Find user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Update settings
    if 'theme' in data:
        user.theme = data['theme']
    
    if 'language' in data:
        user.language = data['language']
    
    if 'notifications_enabled' in data:
        user.notifications_enabled = data['notifications_enabled']
    
    if 'default_model' in data:
        user.default_model = data['default_model']
    
    # Save changes
    try:
        db.session.commit()
        
        return jsonify({
            'message': 'Settings updated successfully',
            'settings': {
                'theme': user.theme,
                'language': user.language,
                'notifications_enabled': user.notifications_enabled,
                'default_model': user.default_model
            }
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@user_bp.route('/password', methods=['PUT'])
@jwt_required()
def change_password():
    """Change user password"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    if 'current_password' not in data or 'new_password' not in data:
        return jsonify({'error': 'Current password and new password are required'}), 400
    
    # Validate password length
    if len(data['new_password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400
    
    # Find user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Check current password
    if not user.check_password(data['current_password']):
        return jsonify({'error': 'Current password is incorrect'}), 401
    
    # Update password
    try:
        user.set_password(data['new_password'])
        db.session.commit()
        
        return jsonify({
            'message': 'Password changed successfully'
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@user_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """Get user statistics"""
    current_user_id = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get chat statistics
    chat_count = Chat.query.filter_by(user_id=current_user_id).count()
    message_count = Message.query.join(Chat).filter(Chat.user_id == current_user_id).count()
    
    # Get user message count
    user_message_count = Message.query.join(Chat).filter(
        Chat.user_id == current_user_id,
        Message.role == 'user'
    ).count()
    
    # Get AI message count
    ai_message_count = Message.query.join(Chat).filter(
        Chat.user_id == current_user_id,
        Message.role == 'assistant'
    ).count()
    
    # Get total tokens
    total_tokens = db.session.query(db.func.sum(Message.tokens)).join(Chat).filter(
        Chat.user_id == current_user_id
    ).scalar() or 0
    
    # Get active days (days with at least one message)
    active_days = db.session.query(db.func.date(Message.created_at)).join(Chat).filter(
        Chat.user_id == current_user_id
    ).distinct().count()
    
    return jsonify({
        'chat_count': chat_count,
        'message_count': message_count,
        'user_message_count': user_message_count,
        'ai_message_count': ai_message_count,
        'total_tokens': total_tokens,
        'active_days': active_days,
        'account_age_days': (db.func.current_timestamp() - user.created_at).days,
        'last_active': user.last_login.isoformat() if user.last_login else None
    }), 200 