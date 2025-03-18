import numpy as np
import time
import threading
import queue
import logging
import json
import os
import hashlib
from collections import OrderedDict, deque
import sqlite3
from datetime import datetime, timedelta
import re
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize thread pool
ai_executor = ThreadPoolExecutor(max_workers=4)

# Response wrapper class to store metadata
class ResponseWrapper:
    """Wrapper class for AI responses to store metadata"""
    def __init__(self, content, original_query=None):
        self.content = content
        self.original_query = original_query
    
    def __str__(self):
        return self.content

# LRU Cache for responses
class LRUCache:
    """LRU Cache implementation for caching AI responses"""
    def __init__(self, capacity=1000):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        
        # If at capacity, remove least recently used
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        # Add new item
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)

# Initialize response cache
response_cache = LRUCache(capacity=1000)

# Semantic similarity calculation for better response matching
def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts using simple TF-IDF"""
    # Tokenize and lowercase
    def tokenize(text):
        # Simple tokenization by splitting on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # Create vocabulary
    vocab = set(tokens1 + tokens2)
    
    # Calculate term frequency
    def term_frequency(tokens):
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        return tf
    
    tf1 = term_frequency(tokens1)
    tf2 = term_frequency(tokens2)
    
    # Calculate dot product
    dot_product = sum(tf1.get(token, 0) * tf2.get(token, 0) for token in vocab)
    
    # Calculate magnitude
    magnitude1 = sum(tf1.get(token, 0) ** 2 for token in vocab) ** 0.5
    magnitude2 = sum(tf2.get(token, 0) ** 2 for token in vocab) ** 0.5
    
    # Avoid division by zero
    if magnitude1 * magnitude2 == 0:
        return 0
    
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)

# Generate cache key for messages
def generate_cache_key(messages, model):
    """Generate a cache key for a set of messages and model"""
    # Extract only the last few messages for the key to avoid cache misses due to long history
    last_messages = messages[-3:] if len(messages) > 3 else messages
    
    # Convert to string and hash
    message_str = json.dumps(last_messages, sort_keys=True)
    hash_obj = hashlib.md5(message_str.encode())
    
    return f"{model}:{hash_obj.hexdigest()}"

# Check cache for similar queries
def check_cache_for_similar(messages, model, threshold=0.85):
    """Check cache for similar queries above a similarity threshold"""
    # Generate key for exact match
    key = generate_cache_key(messages, model)
    
    # Check for exact match
    exact_match = response_cache.get(key)
    if exact_match:
        logger.info(f"Cache hit for {key}")
        return exact_match
    
    # If no exact match, check for similar queries
    if len(messages) == 0:
        return None
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            last_user_message = msg.get('content', '')
            break
    
    if not last_user_message:
        return None
    
    # Check all cache entries for similar queries
    for cache_key, cache_value in response_cache.cache.items():
        # Only check same model
        if not cache_key.startswith(f"{model}:"):
            continue
        
        # Extract the messages from the cache key
        try:
            # Reconstruct the messages from the cache key
            cache_model, cache_hash = cache_key.split(':')
            
            # If we have cached the original messages
            if hasattr(cache_value, 'original_query') and cache_value.original_query:
                cached_last_user_message = None
                for msg in reversed(cache_value.original_query):
                    if msg.get('role') == 'user':
                        cached_last_user_message = msg.get('content', '')
                        break
                
                if cached_last_user_message:
                    # Calculate similarity
                    similarity = calculate_similarity(last_user_message, cached_last_user_message)
                    
                    if similarity >= threshold:
                        logger.info(f"Similar cache hit for {key} (similarity: {similarity:.2f})")
                        return cache_value
        except Exception as e:
            logger.error(f"Error checking cache similarity: {str(e)}")
    
    return None

# Optimized response generation with fallbacks
def generate_optimized_response(messages, model, timeout=30, fallback_model=None):
    """Generate an optimized response with caching and fallbacks"""
    # Check cache first
    cache_result = check_cache_for_similar(messages, model)
    if cache_result:
        return cache_result.content if hasattr(cache_result, 'content') else cache_result
    
    # If not in cache, generate response with timeout
    try:
        # Import here to avoid circular imports
        from services.ai_service import generate_response
        
        # Submit task to thread pool with timeout
        future = ai_executor.submit(generate_response, messages, model)
        response_text = future.result(timeout=timeout)
        
        # Wrap the response in our wrapper class
        response_obj = ResponseWrapper(response_text, original_query=messages)
        
        # Cache the response
        cache_key = generate_cache_key(messages, model)
        response_cache.put(cache_key, response_obj)
        
        return response_text
    
    except TimeoutError:
        logger.error(f"Timeout generating response for model {model}")
        
        # Try fallback model if provided
        if fallback_model and fallback_model != model:
            logger.info(f"Trying fallback model {fallback_model}")
            try:
                # Import here to avoid circular imports
                from services.ai_service import generate_response
                
                # Submit task to thread pool with shorter timeout
                future = ai_executor.submit(generate_response, messages, fallback_model)
                response_text = future.result(timeout=timeout * 0.8)  # Shorter timeout for fallback
                
                return response_text
            except Exception as e:
                logger.error(f"Error with fallback model: {str(e)}")
        
        # If all else fails, return a simple response
        return generate_simple_response(messages)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return generate_simple_response(messages)

# Simple response generation for fallbacks
def generate_simple_response(messages):
    """Generate a simple response when all other methods fail"""
    try:
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                last_user_message = msg.get('content', '')
                break
        
        if not last_user_message:
            return "I'm sorry, I couldn't understand your request. Could you please try again?"
        
        # Simple keyword-based responses
        last_user_message = last_user_message.lower()
        
        if "hello" in last_user_message or "hi" in last_user_message:
            return "Hello! How can I help you today?"
        
        if "how are you" in last_user_message:
            return "I'm functioning well, thank you for asking! How can I assist you?"
        
        if "help" in last_user_message:
            return "I'd be happy to help! I can assist with data analysis, answer questions, or have a conversation. What would you like to know?"
        
        if "thank" in last_user_message:
            return "You're welcome! Is there anything else I can help you with?"
        
        if "bye" in last_user_message or "goodbye" in last_user_message:
            return "Goodbye! Feel free to come back if you have more questions."
        
        # Default response
        return "I'm currently experiencing some technical difficulties, but I'm still here to help. Could you try rephrasing your question or asking something else?"
    
    except Exception as e:
        logger.error(f"Error generating simple response: {str(e)}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."

# Rate limiting to prevent abuse
class RateLimiter:
    """Rate limiter to prevent abuse"""
    def __init__(self, max_requests=60, time_window=60):
        self.max_requests = max_requests  # Max requests per time window
        self.time_window = time_window  # Time window in seconds
        self.requests = {}  # Dictionary to track requests by IP
    
    def is_rate_limited(self, ip_address):
        """Check if an IP address is rate limited"""
        current_time = time.time()
        
        # Clean up old entries
        for ip in list(self.requests.keys()):
            self.requests[ip] = [t for t in self.requests[ip] if current_time - t < self.time_window]
            if not self.requests[ip]:
                del self.requests[ip]
        
        # Check if IP is in the dictionary
        if ip_address not in self.requests:
            self.requests[ip_address] = [current_time]
            return False
        
        # Check if IP has exceeded rate limit
        if len(self.requests[ip_address]) >= self.max_requests:
            return True
        
        # Add current request
        self.requests[ip_address].append(current_time)
        return False

# Initialize rate limiter
rate_limiter = RateLimiter()

# Function to log API usage to SQLite
def log_api_usage(user_id, model, tokens, success, error=None, ip_address=None):
    """Log API usage to SQLite database"""
    try:
        # Connect to database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_usage (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            model TEXT NOT NULL,
            tokens INTEGER,
            success INTEGER NOT NULL,
            error TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Generate ID
        import uuid
        log_id = str(uuid.uuid4())
        
        # Insert log
        cursor.execute('''
        INSERT INTO api_usage (id, user_id, model, tokens, success, error, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (log_id, user_id, model, tokens, 1 if success else 0, error, ip_address))
        
        conn.commit()
        conn.close()
    
    except Exception as e:
        logger.error(f"Error logging API usage: {str(e)}")

# Initialize cache
def init_cache():
    """Initialize cache from database if available"""
    try:
        # Connect to database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS response_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            original_query TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Get recent cache entries
        cursor.execute('''
        SELECT key, value, original_query FROM response_cache
        WHERE created_at > ?
        ORDER BY created_at DESC
        LIMIT 100
        ''', ((datetime.now() - timedelta(days=1)).isoformat(),))
        
        # Load into memory cache
        for key, value, original_query in cursor.fetchall():
            try:
                # Create a response wrapper
                response_obj = ResponseWrapper(value)
                if original_query:
                    response_obj.original_query = json.loads(original_query)
                response_cache.put(key, response_obj)
            except Exception as e:
                logger.error(f"Error loading cache entry: {str(e)}")
        
        conn.close()
        
        logger.info(f"Loaded {len(response_cache)} cache entries from database")
    
    except Exception as e:
        logger.error(f"Error initializing cache: {str(e)}")

# Save cache to database periodically
def save_cache_to_db():
    """Save cache to database periodically"""
    try:
        # Connect to database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS response_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            original_query TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Save each cache entry
        for key, value in response_cache.cache.items():
            try:
                # Extract content and original query
                content = value.content if hasattr(value, 'content') else str(value)
                original_query = json.dumps(value.original_query) if hasattr(value, 'original_query') else None
                
                # Insert or replace
                cursor.execute('''
                INSERT OR REPLACE INTO response_cache (key, value, original_query)
                VALUES (?, ?, ?)
                ''', (key, content, original_query))
            except Exception as e:
                logger.error(f"Error saving cache entry: {str(e)}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(response_cache)} cache entries to database")
    
    except Exception as e:
        logger.error(f"Error saving cache to database: {str(e)}")

# Start cache saving thread
def start_cache_saving_thread():
    """Start a thread to periodically save cache to database"""
    def cache_saver():
        while True:
            try:
                # Sleep for 10 minutes
                time.sleep(600)
                
                # Save cache to database
                save_cache_to_db()
            except Exception as e:
                logger.error(f"Error in cache saver thread: {str(e)}")
    
    # Start thread
    cache_thread = threading.Thread(target=cache_saver, daemon=True)
    cache_thread.start()
    
    logger.info("Cache saving thread started")

# Initialize cache and start saving thread
init_cache()
start_cache_saving_thread() 