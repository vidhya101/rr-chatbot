import sqlite3
import os
import json
import time
import threading
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.environ.get('DB_PATH', 'app.db')

def get_db_connection():
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        profile_picture TEXT,
        bio TEXT,
        role TEXT DEFAULT 'user',
        is_active INTEGER DEFAULT 1,
        last_login TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        theme TEXT DEFAULT 'light',
        language TEXT DEFAULT 'en',
        notifications_enabled INTEGER DEFAULT 1,
        default_model TEXT DEFAULT 'mistral'
    )
    ''')
    
    # Create chats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT,
        user_id TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT 'mistral',
        is_pinned INTEGER DEFAULT 0,
        is_archived INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tokens INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
    )
    ''')
    
    # Create files table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        mime_type TEXT NOT NULL,
        is_processed INTEGER DEFAULT 0,
        processing_status TEXT DEFAULT 'pending',
        processing_error TEXT,
        file_metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        level TEXT NOT NULL,
        source TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT,
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
    )
    ''')
    
    # Create log_archives table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS log_archives (
        id TEXT PRIMARY KEY,
        start_date TIMESTAMP NOT NULL,
        end_date TIMESTAMP NOT NULL,
        log_count INTEGER NOT NULL,
        data TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user_sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        token TEXT NOT NULL UNIQUE,
        ip_address TEXT,
        user_agent TEXT,
        device_info TEXT,
        is_active INTEGER DEFAULT 1,
        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        message_id TEXT,
        is_positive INTEGER NOT NULL,
        content TEXT,
        model TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
    )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (chat_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats (user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs (created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions (expires_at)')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

def archive_old_logs(days=30):
    """Archive logs older than the specified number of days"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    # Get logs to archive
    cursor.execute('SELECT * FROM logs WHERE created_at < ?', (cutoff_date,))
    logs_to_archive = cursor.fetchall()
    
    if not logs_to_archive:
        conn.close()
        return 0
    
    # Create archive
    archive_id = generate_id()
    start_date = min(log['created_at'] for log in logs_to_archive)
    end_date = max(log['created_at'] for log in logs_to_archive)
    log_count = len(logs_to_archive)
    
    # Convert logs to JSON
    logs_json = json.dumps([dict(log) for log in logs_to_archive])
    
    # Insert archive
    cursor.execute('''
    INSERT INTO log_archives (id, start_date, end_date, log_count, data, created_at)
    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (archive_id, start_date, end_date, log_count, logs_json))
    
    # Delete archived logs
    cursor.execute('DELETE FROM logs WHERE created_at < ?', (cutoff_date,))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Archived {log_count} logs from {start_date} to {end_date}")
    return log_count

def purge_expired_sessions():
    """Purge expired user sessions"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete expired sessions
    cursor.execute('DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP')
    deleted_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    if deleted_count > 0:
        logger.info(f"Purged {deleted_count} expired sessions")
    
    return deleted_count

def generate_id():
    """Generate a unique ID for database records"""
    import uuid
    return str(uuid.uuid4())

def start_maintenance_task():
    """Start a background thread for database maintenance tasks"""
    def maintenance_worker():
        """Worker function for database maintenance"""
        while True:
            try:
                # Archive old logs (older than 30 days)
                archived_logs = archive_old_logs(days=30)
                if archived_logs > 0:
                    logger.info(f"Archived {archived_logs} old logs")
                
                # Purge expired sessions
                purged_sessions = purge_expired_sessions()
                if purged_sessions > 0:
                    logger.info(f"Purged {purged_sessions} expired sessions")
                
            except Exception as e:
                logger.error(f"Error in database maintenance task: {str(e)}")
            
            # Sleep for 24 hours
            time.sleep(86400)  # 24 hours
    
    # Start the maintenance thread
    maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
    maintenance_thread.start()
    logger.info("Database maintenance task started")

# Log functions
def log_info(source, message, user_id=None, details=None, ip_address=None, user_agent=None):
    """Log an info message"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    log_id = generate_id()
    details_json = json.dumps(details) if details else None
    
    cursor.execute('''
    INSERT INTO logs (id, user_id, level, source, message, details, ip_address, user_agent)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (log_id, user_id, 'info', source, message, details_json, ip_address, user_agent))
    
    conn.commit()
    conn.close()
    
    return log_id

def log_error(source, message, user_id=None, details=None, ip_address=None, user_agent=None):
    """Log an error message"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    log_id = generate_id()
    details_json = json.dumps(details) if details else None
    
    cursor.execute('''
    INSERT INTO logs (id, user_id, level, source, message, details, ip_address, user_agent)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (log_id, user_id, 'error', source, message, details_json, ip_address, user_agent))
    
    conn.commit()
    conn.close()
    
    return log_id

def log_warning(source, message, user_id=None, details=None, ip_address=None, user_agent=None):
    """Log a warning message"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    log_id = generate_id()
    details_json = json.dumps(details) if details else None
    
    cursor.execute('''
    INSERT INTO logs (id, user_id, level, source, message, details, ip_address, user_agent)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (log_id, user_id, 'warning', source, message, details_json, ip_address, user_agent))
    
    conn.commit()
    conn.close()
    
    return log_id

# Initialize database when module is imported
if not os.path.exists(DB_PATH):
    init_db()
    # Start maintenance task
    start_maintenance_task() 