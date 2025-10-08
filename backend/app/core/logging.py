import logging
import sys
import os
from typing import Optional
from datetime import datetime
from pathlib import Path
from .config import settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """Setup centralized logging configuration."""
    
    # Use provided log level or default from settings
    level = log_level or settings.log_level
    
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logs directory structure
    setup_rag_logging_directories()
    
    # Add file handler for persistent logs
    try:
        log_file_path = os.path.join("logs", "app.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - [%(funcName)s] - %(message)s'
        ))
        
        # Get root logger and add file handler
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
    except Exception as e:
        logging.warning(f"Failed to create log file: {e}")


def setup_rag_logging_directories():
    """Setup directory structure for RAG-specific logging."""
    # Use absolute path to ensure we're creating directories in the correct location
    base_logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
    base_logs_dir.mkdir(exist_ok=True)
    
    # Create RAG-specific subdirectories
    rag_dirs = [
        "rag/embeddings",
        "rag/llm_responses", 
        "rag/augmentation",
        "rag/similarity",
        "rag/retrieval"
    ]
    
    for dir_path in rag_dirs:
        full_path = base_logs_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def create_rag_logger(logger_name: str, log_file: str, log_level: int = logging.INFO) -> logging.Logger:
    """Create a specialized RAG logger with file output."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create file handler with fixed filename (no timestamp)
    log_filename = f"{log_file.split('/')[-1]}.log"
    # Use absolute path to ensure we're writing to the correct location
    log_dir = Path(__file__).parent.parent.parent.parent / "logs" / "rag" / log_file.split('/')[0]
    
    # Ensure directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_filename
    file_handler = logging.FileHandler(log_path, mode='w')  # 'w' mode to overwrite file
    file_handler.setLevel(log_level)
    
    # Create formatter with detailed timestamp
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger


# Create common loggers
app_logger = get_logger("app")
db_logger = get_logger("database")
service_logger = get_logger("services")

def clear_rag_logs():
    """Clear all RAG log files before starting a new evaluation session."""
    # Use absolute path to ensure we're clearing the correct files
    base_logs_dir = Path(__file__).parent.parent.parent.parent / "logs" / "rag"
    
    rag_log_files = [
        base_logs_dir / "embeddings" / "embedding_generation.log",
        base_logs_dir / "llm_responses" / "llm_generation.log", 
        base_logs_dir / "augmentation" / "context_augmentation.log",
        base_logs_dir / "similarity" / "similarity_computation.log",
        base_logs_dir / "retrieval" / "context_retrieval.log"
    ]
    
    for log_path in rag_log_files:
        try:
            if log_path.exists():
                log_path.unlink()  # Delete the file
                print(f"Cleared log file: {log_path}")
        except Exception as e:
            print(f"Failed to clear log file {log_path}: {e}")


def initialize_rag_loggers():
    """Initialize RAG loggers with fresh file handlers."""
    # Clear existing handlers
    for logger_name in ["rag.embeddings", "rag.llm_responses", "rag.augmentation", "rag.similarity", "rag.retrieval"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Create fresh loggers
    global embedding_logger, llm_response_logger, augmentation_logger, similarity_logger, retrieval_logger
    embedding_logger = create_rag_logger("rag.embeddings", "embeddings/embedding_generation")
    llm_response_logger = create_rag_logger("rag.llm_responses", "llm_responses/llm_generation")
    augmentation_logger = create_rag_logger("rag.augmentation", "augmentation/context_augmentation")
    similarity_logger = create_rag_logger("rag.similarity", "similarity/similarity_computation")
    retrieval_logger = create_rag_logger("rag.retrieval", "retrieval/context_retrieval")


# Create specialized RAG loggers
embedding_logger = create_rag_logger("rag.embeddings", "embeddings/embedding_generation")
llm_response_logger = create_rag_logger("rag.llm_responses", "llm_responses/llm_generation")
augmentation_logger = create_rag_logger("rag.augmentation", "augmentation/context_augmentation")
similarity_logger = create_rag_logger("rag.similarity", "similarity/similarity_computation")
retrieval_logger = create_rag_logger("rag.retrieval", "retrieval/context_retrieval")
