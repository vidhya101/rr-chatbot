import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

# Database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False}  # Required for SQLite
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

# Create thread-safe session
db_session = scoped_session(SessionLocal)

# Base class for models
Base = declarative_base()

@contextmanager
def get_db():
    """Get database session with automatic cleanup"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def init_db():
    """Initialize database schema"""
    Base.metadata.create_all(bind=engine)

def drop_db():
    """Drop all tables"""
    Base.metadata.drop_all(bind=engine) 