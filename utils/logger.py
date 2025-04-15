import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name="app", log_level=None):
    """Configure and return a logger instance"""
    # Get log level from environment or parameter
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level (with fallback to INFO)
    try:
        logger.setLevel(getattr(logging, log_level))
        logger.info(f"Log level set to {log_level}")
    except AttributeError:
        logger.setLevel(logging.INFO)
        logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs with rotation
    try:
        os.makedirs("logs", exist_ok=True)
        log_file_path = os.path.join("logs", f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file_path, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - [%(funcName)s] - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Log file created at: {log_file_path}")
    except Exception as e:
        logger.warning(f"Failed to create log file: {e}")
    
    return logger

# Default application logger
app_logger = setup_logger("app")

# Specialized loggers
db_logger = setup_logger("mongodb")
embedding_logger = setup_logger("embedding")
code_analyzer_logger = setup_logger("code_analyzer")
feedback_logger = setup_logger("feedback")
api_logger = setup_logger("api") 