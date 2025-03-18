from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import threading
import time
from datetime import datetime, timedelta

# Initialize SQLAlchemy
db = SQLAlchemy()

def init_db(app):
    """Initialize the database with the Flask app"""
    db.init_app(app)
    Migrate(app, db)
    
    # Import models to ensure they're registered with SQLAlchemy
    from models.user import User
    from models.chat import Chat, Message
    from models.file import File
    from models.dashboard import Dashboard
    from models.log import Log, LogArchive, UserSession
    
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()
        
        # Start background task for database maintenance
        if not app.config.get('TESTING', False):
            start_db_maintenance_task(app)

def start_db_maintenance_task(app):
    """Start a background thread for database maintenance tasks"""
    def maintenance_worker():
        """Worker function for database maintenance"""
        from models.log import Log, UserSession
        
        with app.app_context():
            while True:
                try:
                    # Purge old logs (older than 30 days)
                    purged_logs = Log.purge_old_logs(days=30)
                    if purged_logs > 0:
                        print(f"Purged {purged_logs} old logs")
                    
                    # Clean up expired sessions
                    expired_sessions = UserSession.cleanup_expired_sessions()
                    if expired_sessions > 0:
                        print(f"Cleaned up {expired_sessions} expired sessions")
                    
                except Exception as e:
                    print(f"Error in database maintenance task: {str(e)}")
                
                # Sleep for 24 hours
                time.sleep(86400)  # 24 hours
    
    # Start the maintenance thread
    maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
    maintenance_thread.start()
    print("Database maintenance task started") 