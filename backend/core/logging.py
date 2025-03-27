"""
Logging configuration for the RR-Chatbot application.
This module sets up logging with proper formatting, rotation, and handlers.
"""

import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
from config import Config

class LogManager:
    """Manages application logging configuration."""
    
    def __init__(self):
        """Initialize logging manager."""
        self.log_dir = Config.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Configure logging with handlers and formatters."""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create handlers
        self._setup_file_handlers(file_formatter)
        self._setup_console_handler(console_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Disable propagation to avoid duplicate logs
        for logger_name in ['werkzeug', 'sqlalchemy', 'flask']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def _setup_file_handlers(self, formatter: logging.Formatter) -> None:
        """Set up file handlers for different log levels."""
        # Main application log
        app_handler = RotatingFileHandler(
            self.log_dir / 'app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setFormatter(formatter)
        app_handler.setLevel(logging.INFO)
        
        # Error log
        error_handler = TimedRotatingFileHandler(
            self.log_dir / 'error.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(app_handler)
        root_logger.addHandler(error_handler)
    
    def _setup_console_handler(self, formatter: logging.Formatter) -> None:
        """Set up console handler for development."""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name."""
        return logging.getLogger(name)
    
    def set_level(self, level: int) -> None:
        """Set the logging level for all handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)

# Create global log manager instance
log_manager = LogManager()

# Export commonly used function
get_logger = log_manager.get_logger 