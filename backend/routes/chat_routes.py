from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.db import db
from models.chat import Chat, Message
from models.user import User
from models.log import Log
from services.ai_service import generate_response, count_tokens
from services.model_service import get_available_models, check_ollama_status
from utils.ai_utils import generate_optimized_response, log_api_usage, rate_limiter
import uuid
from datetime import datetime
import logging
import time
import threading
import queue
import concurrent.futures
import json
import re
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

# For Mistral AI
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logging.warning("Mistral AI client not available. Install with: pip install mistralai")

# Load environment variables
load_dotenv()

# API Keys
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Thread pool for concurrent processing
chat_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Response cache for quick responses
response_cache = {}

@chat_bp.route('/models', methods=['GET'])
def get_models():
    """Get available AI models"""
    try:
        # Check if Ollama is healthy
        ollama_healthy, status_message = check_ollama_status()
        
        # Get models with force refresh if requested
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        models = get_available_models(force_refresh=force_refresh)
        
        # Log the request
        Log.log_info('api', 'Models requested', 
                    details={'ollama_healthy': ollama_healthy, 'model_count': len(models)},
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'))
        
        return jsonify({
            "success": True,
            "models": models,
            "ollama_status": "online" if ollama_healthy else "offline",
            "status_message": status_message
        }), 200
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        Log.log_error('api', f"Error getting models: {str(e)}", 
                     ip_address=request.remote_addr,
                     user_agent=request.headers.get('User-Agent'))
        return jsonify({
            "success": False,
            "error": "Failed to get models", 
            "details": str(e)
        }), 500

@chat_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    """Process a chat message and get AI response"""
    start_time = time.time()
    
    try:
        # Get current user
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        # Check rate limiting
        if rate_limiter.is_rate_limited(request.remote_addr):
            Log.log_warning('chat', "Rate limit exceeded", 
                           user_id=current_user_id,
                           ip_address=request.remote_addr)
            return jsonify({
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "rate_limited": True
            }), 429
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        message = data.get('message')
        chat_id = data.get('chatId')
        model = data.get('model', 'mistral')  # Default to mistral if not specified
        model_parameters = data.get('parameters', {})
        
        if not message:
            return jsonify({"success": False, "error": "No message provided"}), 400
        
        # Log the request
        Log.log_info('chat', f"Chat request from user {user.username}", 
                    user_id=current_user_id,
                    details={'model': model, 'message_length': len(message)},
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'))
        
        # Get or create chat
        chat = None
        if chat_id:
            chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
        
        if not chat:
            # Create new chat
            chat = Chat(
                user_id=current_user_id,
                model=model,
                title=message[:30] + "..." if len(message) > 30 else message,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.session.add(chat)
            db.session.commit()
        
        # Create user message
        user_message = Message(
            chat_id=chat.id,
            role="user",
            content=message,
            created_at=datetime.utcnow()
        )
        db.session.add(user_message)
        db.session.commit()
        
        # Get chat history for context
        chat_messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
        messages_for_ai = [{"role": msg.role, "content": msg.content} for msg in chat_messages]
        
        # Add system message if not present
        if not any(msg.get('role') == 'system' for msg in messages_for_ai):
            system_message = {
                "role": "system",
                "content": "You are a helpful, friendly, and engaging assistant. Be conversational and use a variety of sentence structures. Show enthusiasm and personality in your responses. Be concise but thorough, and use emojis occasionally where appropriate. If you don't know something, be honest about it."
            }
            messages_for_ai.insert(0, system_message)
        
        # Generate AI response with optimized handling
        try:
            # Determine fallback model
            fallback_model = None
            if model.startswith('llama'):
                fallback_model = 'mistral'
            elif model.startswith('mistral'):
                fallback_model = 'llama2'
            
            # Use optimized response generation with fallback
            ai_response = generate_optimized_response(
                messages_for_ai, 
                model, 
                timeout=45,  # Increased timeout
                fallback_model=fallback_model
            )
            
            # Log API usage
            tokens = count_tokens(message) + count_tokens(ai_response)
            log_api_usage(
                user_id=current_user_id,
                model=model,
                tokens=tokens,
                success=True,
                ip_address=request.remote_addr
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            Log.log_error('chat', f"Error generating response: {str(e)}", 
                         user_id=current_user_id,
                         details={'model': model},
                         ip_address=request.remote_addr)
            
            # Log API usage with error
            log_api_usage(
                user_id=current_user_id,
                model=model,
                tokens=count_tokens(message),
                success=False,
                error=str(e),
                ip_address=request.remote_addr
            )
            
            # Provide a friendly error message
            ai_response = "I'm sorry, but I encountered an error processing your message. This might be due to high demand or a temporary issue with the AI service. Could you try again in a moment or with a different model?"
        
        # Create AI message
        ai_message = Message(
            chat_id=chat.id,
            role="assistant",
            content=ai_response,
            created_at=datetime.utcnow()
        )
        db.session.add(ai_message)
        
        # Update chat
        chat.updated_at = datetime.utcnow()
        chat.model = model  # Update the model used
        db.session.commit()
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log the response
        Log.log_info('chat', f"Chat response sent to user {user.username}", 
                    user_id=current_user_id,
                    details={
                        'model': model, 
                        'response_length': len(ai_response),
                        'response_time': response_time
                    },
                    ip_address=request.remote_addr)
        
        # Generate suggested follow-up questions
        suggested_questions = []
        try:
            # Simple heuristic to generate follow-up questions
            if len(ai_response) > 100:
                # Add a system message asking for follow-up questions
                question_prompt = messages_for_ai + [
                    {"role": "assistant", "content": ai_response},
                    {"role": "system", "content": "Based on this conversation, suggest 3 short follow-up questions the user might want to ask. Return ONLY a JSON array of strings with no explanation."}
                ]
                
                # Generate questions in a separate thread to not block the response
                def generate_questions():
                    try:
                        questions_response = generate_response(question_prompt, model)
                        # Try to parse as JSON
                        try:
                            # Extract JSON array if it's embedded in text
                            json_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', questions_response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                try:
                                    questions = json.loads(json_str)
                                    # Store in database for future retrieval
                                    chat.metadata = json.dumps({"suggested_questions": questions})
                                    db.session.commit()
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse extracted JSON: {json_str}")
                            else:
                                logger.warning("No JSON array found in response")
                        except Exception as e:
                            logger.warning(f"Failed to parse suggested questions as JSON: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error generating suggested questions: {str(e)}")
                
                # Start in background
                threading.Thread(target=generate_questions).start()
        except Exception as e:
            logger.error(f"Error setting up suggested questions: {str(e)}")
        
        return jsonify({
            "success": True,
            "chatId": chat.id,
            "message": ai_response,
            "responseTime": response_time,
            "suggestedQuestions": suggested_questions
        }), 200
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        Log.log_error('chat', f"Error in chat: {str(e)}", 
                     user_id=get_jwt_identity() if get_jwt_identity() else None,
                     ip_address=request.remote_addr)
        return jsonify({
            "success": False,
            "error": "Failed to process chat message",
            "message": "I'm sorry, but I encountered an error processing your message. Please try again in a moment."
        }), 500

@chat_bp.route('/chats', methods=['GET'])
@jwt_required()
def get_chats():
    """Get all chats for the current user"""
    try:
        current_user_id = get_jwt_identity()
        
        # Get all chats for the user
        chats = Chat.query.filter_by(user_id=current_user_id).order_by(Chat.updated_at.desc()).all()
        
        chats_data = []
        for chat in chats:
            # Get the last message for preview
            last_message = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at.desc()).first()
            
            chats_data.append({
                "id": chat.id,
                "title": chat.title,
                "lastMessage": last_message.content if last_message else "",
                "updatedAt": chat.updated_at.isoformat()
            })
        
        return jsonify({"chats": chats_data}), 200
    
    except Exception as e:
        logger.error(f"Error getting chats: {str(e)}")
        return jsonify({"error": "Failed to get chats"}), 500

@chat_bp.route('/chats/<int:chat_id>', methods=['GET'])
@jwt_required()
def get_chat(chat_id):
    """Get a specific chat with all messages"""
    try:
        current_user_id = get_jwt_identity()
        
        # Get the chat
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Get all messages for the chat
        messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
        
        messages_data = []
        for message in messages:
            messages_data.append({
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "createdAt": message.created_at.isoformat()
            })
        
        chat_data = {
            "id": chat.id,
            "title": chat.title,
            "messages": messages_data,
            "createdAt": chat.created_at.isoformat(),
            "updatedAt": chat.updated_at.isoformat()
        }
        
        return jsonify({"chat": chat_data}), 200
    
    except Exception as e:
        logger.error(f"Error getting chat: {str(e)}")
        return jsonify({"error": "Failed to get chat"}), 500

@chat_bp.route('/chats', methods=['POST'])
@jwt_required()
def create_chat():
    """Create a new chat"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate model
    model = data.get('model', 'digitalogy')
    
    # Create chat
    try:
        chat = Chat(
            user_id=current_user_id,
            model=model,
            title=data.get('title', 'New Chat')
        )
        
        db.session.add(chat)
        db.session.commit()
        
        return jsonify({
            'message': 'Chat created successfully',
            'chat': chat.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/chats/<chat_id>', methods=['PUT'])
@jwt_required()
def update_chat(chat_id):
    """Update a chat"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Find chat
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Update fields
    if 'title' in data:
        chat.title = data['title']
    
    if 'is_pinned' in data:
        chat.is_pinned = data['is_pinned']
    
    if 'is_archived' in data:
        chat.is_archived = data['is_archived']
    
    # Save changes
    try:
        db.session.commit()
        
        return jsonify({
            'message': 'Chat updated successfully',
            'chat': chat.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/chats/<int:chat_id>', methods=['DELETE'])
@jwt_required()
def delete_chat(chat_id):
    """Delete a specific chat"""
    try:
        current_user_id = get_jwt_identity()
        
        # Get the chat
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Delete all messages for the chat
        Message.query.filter_by(chat_id=chat.id).delete()
        
        # Delete the chat
        current_app.db.session.delete(chat)
        current_app.db.session.commit()
        
        return jsonify({"message": "Chat deleted successfully"}), 200
    
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        return jsonify({"error": "Failed to delete chat"}), 500

@chat_bp.route('/chats/<int:chat_id>/title', methods=['PUT'])
@jwt_required()
def update_chat_title(chat_id):
    """Update the title of a chat"""
    try:
        current_user_id = get_jwt_identity()
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        new_title = data.get('title')
        
        if not new_title:
            return jsonify({"error": "No title provided"}), 400
        
        # Get the chat
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Update the title
        chat.title = new_title
        chat.updated_at = datetime.utcnow()
        current_app.db.session.commit()
        
        return jsonify({"message": "Chat title updated successfully"}), 200
    
    except Exception as e:
        logger.error(f"Error updating chat title: {str(e)}")
        return jsonify({"error": "Failed to update chat title"}), 500

@chat_bp.route('/chats/<chat_id>/messages', methods=['POST'])
@jwt_required()
def send_message(chat_id):
    """Send a message to a chat and get AI response"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate message
    if 'content' not in data:
        return jsonify({'error': 'Message content is required'}), 400
    
    # Find chat
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Get user
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Count tokens in user message
    user_message_tokens = count_tokens(data['content'])
    
    # Create user message
    try:
        user_message = Message(
            chat_id=chat.id,
            role='user',
            content=data['content'],
            tokens=user_message_tokens
        )
        
        db.session.add(user_message)
        db.session.commit()
        
        # Get chat history for context
        messages = [
            {'role': msg.role, 'content': msg.content}
            for msg in chat.messages
        ]
        
        # Generate AI response
        ai_response_content = generate_response(
            messages=messages,
            model=chat.model,
            user=user.to_dict()
        )
        
        # Count tokens in AI response
        ai_response_tokens = count_tokens(ai_response_content)
        
        # Create AI message
        ai_message = Message(
            chat_id=chat.id,
            role='assistant',
            content=ai_response_content,
            tokens=ai_response_tokens
        )
        
        db.session.add(ai_message)
        db.session.commit()
        
        # Update chat title if it's a new chat
        if chat.title == 'New Chat' and len(chat.messages) <= 3:
            # Generate a title based on the first user message
            chat.title = data['content'][:30] + '...' if len(data['content']) > 30 else data['content']
            db.session.commit()
        
        return jsonify({
            'user_message': user_message.to_dict(),
            'ai_message': ai_message.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/chats/<chat_id>/messages', methods=['GET'])
@jwt_required()
def get_messages(chat_id):
    """Get all messages for a chat"""
    current_user_id = get_jwt_identity()
    
    # Find chat
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user_id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    # Query messages
    messages_pagination = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).paginate(page=page, per_page=per_page)
    
    # Format response
    messages = [message.to_dict() for message in messages_pagination.items]
    
    return jsonify({
        'messages': messages,
        'total': messages_pagination.total,
        'pages': messages_pagination.pages,
        'page': page,
        'per_page': per_page
    }), 200

@chat_bp.route('/public/chat', methods=['POST'])
def public_chat():
    """Public chat endpoint that doesn't require authentication"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        message = data.get('message')
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        chat_history = data.get('chatHistory', [])
        model = data.get('model', 'mistral')
        
        # Log the request
        Log.log_info('public_chat', "Public chat request", 
                    details={'model': model, 'message_length': len(message)},
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'))
        
        # Create a system message that sets the tone for the AI
        system_message = """You are a friendly, helpful, and engaging AI assistant. 
        Your responses should be conversational, warm, and natural-sounding.
        Use a variety of sentence structures and occasionally include emojis where appropriate.
        Show enthusiasm and personality in your responses, and feel free to use light humor.
        Be concise but thorough, and always focus on being helpful and informative.
        If you're unsure about something, be honest about your limitations.
        Remember to be empathetic and understanding of the user's needs."""
        
        # Format the chat history for the AI
        formatted_history = []
        if chat_history:
            for chat in chat_history:
                role = chat.get('role', '')
                content = chat.get('message', '')
                if role and content:
                    formatted_history.append({"role": role, "content": content})
        
        # Add the system message at the beginning
        if formatted_history and formatted_history[0].get('role') != 'system':
            formatted_history.insert(0, {"role": "system", "content": system_message})
        else:
            formatted_history = [{"role": "system", "content": system_message}]
        
        # Add the current message
        formatted_history.append({"role": "user", "content": message})
        
        # Check cache for quick response
        cache_key = f"{model}:{hash(str(formatted_history))}"
        if cache_key in response_cache:
            response_text = response_cache[cache_key]
            logger.info(f"Using cached response for {cache_key}")
        else:
            # Generate a response using the AI service with timeout
            try:
                # Submit task to thread pool with timeout
                future = chat_executor.submit(generate_response, formatted_history, model)
                response_text = future.result(timeout=30)  # 30 second timeout
                
                # Cache the response
                response_cache[cache_key] = response_text
                
                # Limit cache size
                if len(response_cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(response_cache.keys())[:100]
                    for key in keys_to_remove:
                        response_cache.pop(key, None)
                
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout generating response for model {model}")
                Log.log_error('public_chat', f"Timeout generating response", 
                             details={'model': model},
                             ip_address=request.remote_addr)
                
                # Provide a fallback response
                response_text = "I'm sorry, but I'm taking longer than expected to process your request. Could you try again with a simpler question or a different model?"
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log the response
        Log.log_info('public_chat', "Public chat response sent", 
                    details={
                        'model': model, 
                        'response_length': len(response_text),
                        'response_time': response_time
                    },
                    ip_address=request.remote_addr)
        
        return jsonify({
            "message": response_text,
            "responseTime": response_time
        })
    
    except Exception as e:
        logger.error(f"Error in public chat: {str(e)}")
        Log.log_error('public_chat', f"Error in public chat: {str(e)}", 
                     ip_address=request.remote_addr)
        return jsonify({
            "error": "Something went wrong processing your request",
            "message": "Sorry about that! I encountered an unexpected issue. Could you try again?"
        }), 500

@chat_bp.route('/simple-chat', methods=['POST'])
def simple_chat():
    """Simple chat endpoint that doesn't require authentication"""
    try:
        # Check if form data or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle form data with files
            data_json = request.form.get('data')
            if not data_json:
                return jsonify({"success": False, "error": "No data provided"}), 400
            
            data = json.loads(data_json)
            message = data.get('message')
            chat_history = data.get('chatHistory', [])
            model = data.get('model', 'mistral')
            
            # Handle file uploads
            files = []
            for key in request.files:
                file = request.files[key]
                if file and file.filename:
                    # Save file
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    
                    # Create uploads directory if it doesn't exist
                    uploads_dir = os.path.join(current_app.config.get('UPLOAD_FOLDER', 'uploads'), 'temp')
                    os.makedirs(uploads_dir, exist_ok=True)
                    
                    file_path = os.path.join(uploads_dir, unique_filename)
                    file.save(file_path)
                    
                    files.append({
                        'filename': unique_filename,
                        'original_filename': filename,
                        'path': file_path
                    })
            
            # If files were uploaded, add them to the message
            if files:
                # Check if the message is about visualization
                if message and ('visualize' in message.lower() or 'visualization' in message.lower() or 
                               'chart' in message.lower() or 'graph' in message.lower() or 
                               'plot' in message.lower() or 'dashboard' in message.lower()):
                    # Generate visualizations for the files
                    visualizations = []
                    for file_info in files:
                        try:
                            # Generate dashboard for the file
                            result = generate_dashboard(file_info['path'])
                            if result.get('success', False):
                                visualizations.extend(result.get('visualizations', []))
                                
                                # Create a response with visualization links
                                viz_links = []
                                for viz in result.get('visualizations', []):
                                    viz_links.append(f"![{viz.get('title', 'Visualization')}]({viz.get('url', '')})")
                                
                                # Create a markdown response with visualization links
                                response = f"I've analyzed your file '{file_info['original_filename']}' and created some visualizations for you:\n\n"
                                response += "\n\n".join(viz_links)
                                response += f"\n\nHere's a summary of your data:\n\n"
                                
                                # Add data summary
                                stats = result.get('stats', {})
                                response += f"- Rows: {stats.get('rows', 'N/A')}\n"
                                response += f"- Columns: {stats.get('columns', 'N/A')}\n"
                                response += f"- Numeric columns: {', '.join(stats.get('numeric_columns', []))}\n"
                                response += f"- Categorical columns: {', '.join(stats.get('categorical_columns', []))}\n"
                                
                                # Add insights based on visualizations
                                response += f"\n\nBased on the visualizations, here are some insights:\n\n"
                                
                                # Add insights for correlation heatmap
                                if any('correlation' in viz.get('type', '') for viz in result.get('visualizations', [])):
                                    response += "- The correlation heatmap shows relationships between numeric variables. Strong positive correlations appear in dark red, while strong negative correlations appear in dark blue.\n"
                                
                                # Add insights for histograms
                                if any('histogram' in viz.get('type', '') for viz in result.get('visualizations', [])):
                                    response += "- The histograms show the distribution of numeric variables. You can see the shape, central tendency, and spread of each variable.\n"
                                
                                # Add insights for categorical data
                                if any('categorical' in viz.get('type', '') for viz in result.get('visualizations', [])):
                                    response += "- The bar charts show the frequency of different categories in your categorical variables.\n"
                                
                                # Add call to action
                                response += f"\n\nWould you like me to create any specific visualizations for this data? For example, I can create scatter plots, line charts, or box plots for specific variables."
                                
                                return jsonify({"success": True, "message": response}), 200
                        except Exception as e:
                            logger.error(f"Error generating visualizations: {str(e)}")
                
                # Add file information to the message
                file_info_text = "I've uploaded the following files:\n"
                for file_info in files:
                    file_info_text += f"- {file_info['original_filename']}\n"
                
                if message:
                    message = f"{message}\n\n{file_info_text}"
                else:
                    message = file_info_text
        else:
            # Handle JSON data
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400
            
            message = data.get('message')
            chat_history = data.get('chatHistory', [])
            model = data.get('model', 'mistral')
        
        if not message:
            return jsonify({"success": False, "error": "No message provided"}), 400
        
        # Format chat history for AI
        messages_for_ai = []
        
        # Add system message if not present
        if not any(msg.get('role') == 'system' for msg in chat_history):
            system_message = {
                "role": "system",
                "content": "You are a helpful, friendly, and engaging assistant. Be conversational and use a variety of sentence structures. Show enthusiasm and personality in your responses. Be concise but thorough, and use emojis occasionally where appropriate. If you don't know something, be honest about it."
            }
            messages_for_ai.append(system_message)
        
        # Add chat history
        for msg in chat_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role and content:
                messages_for_ai.append({
                    "role": role,
                    "content": content
                })
        
        # Add current message
        messages_for_ai.append({
            "role": "user",
            "content": message
        })
        
        # Generate AI response with optimized handling
        try:
            # Determine fallback model
            fallback_model = None
            if model.startswith('llama'):
                fallback_model = 'mistral'
            elif model.startswith('mistral'):
                fallback_model = 'llama2'
            
            # Check if Claude is available and preferred
            if 'claude' in model.lower() and is_claude_available():
                ai_response = generate_claude_response(messages_for_ai, model)
            else:
                # Use optimized response generation with fallback
                ai_response = generate_optimized_response(
                    messages_for_ai, 
                    model, 
                    timeout=45,  # Increased timeout
                    fallback_model=fallback_model
                )
            
            # Log API usage
            log_api_usage(
                user_id='anonymous',
                model=model,
                tokens=count_tokens(message) + count_tokens(ai_response),
                success=True,
                ip_address=request.remote_addr
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Log API usage with error
            log_api_usage(
                user_id='anonymous',
                model=model,
                tokens=count_tokens(message),
                success=False,
                error=str(e),
                ip_address=request.remote_addr
            )
            
            # Provide a friendly error message
            ai_response = "I'm sorry, but I encountered an error processing your message. This might be due to high demand or a temporary issue with the AI service. Could you try again in a moment or with a different model?"
        
        return jsonify({
            "success": True,
            "message": ai_response,
            "model": model
        }), 200
    
    except Exception as e:
        logger.error(f"Error in simple chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to process chat message",
            "message": "I'm sorry, but I encountered an error processing your message. Please try again in a moment."
        }), 500 