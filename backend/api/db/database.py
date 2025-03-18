import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./app.db")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    poolclass=NullPool if "sqlite" in DATABASE_URL else None
)

# Create async session factory
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Create declarative base
Base = declarative_base()

async def init_db():
    """Initialize the database"""
    try:
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

async def get_db():
    """Get a database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def close_db_connection():
    """Close database connection"""
    await engine.dispose()
    logger.info("Database connection closed") 