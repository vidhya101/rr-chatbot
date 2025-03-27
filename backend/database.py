"""
Database configuration and management for the RR-Chatbot application.
This module handles database connections, session management, and migrations.
"""

import os
from typing import Generator, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        """Initialize database manager with configuration."""
        self.engine = self._create_engine()
        self.SessionLocal = self._create_session_factory()
        self.db_session = scoped_session(self.SessionLocal)
        self.Base = declarative_base()
        
        # Register event listeners
        self._register_event_listeners()
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with connection pooling."""
        return create_engine(
            Config.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections after 30 minutes
            connect_args={'check_same_thread': False} if Config.IS_WINDOWS else {}
        )
    
    def _create_session_factory(self) -> sessionmaker:
        """Create session factory with configuration."""
        return sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    
    def _register_event_listeners(self) -> None:
        """Register database event listeners."""
        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance."""
            if Config.DATABASE_URL.startswith('sqlite'):
                cursor = dbapi_connection.cursor()
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA synchronous=NORMAL')
                cursor.execute('PRAGMA temp_store=MEMORY')
                cursor.close()
    
    @contextmanager
    def get_db(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
    
    def init_db(self) -> None:
        """Initialize database schema."""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {str(e)}")
            raise
    
    def drop_db(self) -> None:
        """Drop all tables (use with caution)."""
        try:
            self.Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {str(e)}")
            raise
    
    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            with self.get_db() as db:
                db.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False

# Create global database manager instance
db_manager = DatabaseManager()

# Export commonly used functions and classes
get_db = db_manager.get_db
init_db = db_manager.init_db
drop_db = db_manager.drop_db
check_connection = db_manager.check_connection
Base = db_manager.Base  # Export Base for models 