"""
RAG Logging Framework

This module provides specialized logging decorators and utilities for RAG (Retrieval-Augmented Generation) operations.
It includes separate logging for embeddings, LLM responses, augmentation, similarity computation, and retrieval.
"""

import functools
import time
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

from .logging import (
    embedding_logger, 
    llm_response_logger, 
    augmentation_logger, 
    similarity_logger, 
    retrieval_logger,
    clear_rag_logs,
    initialize_rag_loggers,
    create_rag_logger
)


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text to prevent encoding issues."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)


def start_fresh_rag_session():
    """Clear all RAG logs and initialize fresh loggers for a new evaluation session."""
    print("Starting fresh RAG session - clearing previous logs...")
    clear_rag_logs()
    initialize_rag_loggers()
    
    # Update the imported loggers in the logging module
    import sys
    logging_module = sys.modules['app.core.logging']
    
    # Recreate loggers with fresh file handlers
    logging_module.embedding_logger = create_rag_logger("rag.embeddings", "embeddings/embedding_generation")
    logging_module.llm_response_logger = create_rag_logger("rag.llm_responses", "llm_responses/llm_generation")
    logging_module.augmentation_logger = create_rag_logger("rag.augmentation", "augmentation/context_augmentation")
    logging_module.similarity_logger = create_rag_logger("rag.similarity", "similarity/similarity_computation")
    logging_module.retrieval_logger = create_rag_logger("rag.retrieval", "retrieval/context_retrieval")
    
    # Update the local references as well
    global embedding_logger, llm_response_logger, augmentation_logger, similarity_logger, retrieval_logger
    embedding_logger = logging_module.embedding_logger
    llm_response_logger = logging_module.llm_response_logger
    augmentation_logger = logging_module.augmentation_logger
    similarity_logger = logging_module.similarity_logger
    retrieval_logger = logging_module.retrieval_logger
    
    print("RAG session initialized with fresh log files.")


def log_embedding_operation(operation_type: str = "embedding_generation"):
    """Decorator for logging embedding operations with detailed context."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
            
            # Extract context from arguments (skip 'self' for methods)
            if args and hasattr(args[0], '__class__') and hasattr(args[0], '__dict__'):
                # This is likely a method call, skip 'self'
                text = args[1] if len(args) > 1 else kwargs.get('text', '')
            else:
                text = args[0] if args else kwargs.get('text', '')
            model = kwargs.get('model', 'unknown')
            
            embedding_logger.info(f"[{operation_id}] Starting {operation_type}")
            embedding_logger.info(f"[{operation_id}] Input text length: {len(text)} characters")
            embedding_logger.info(f"[{operation_id}] Model: {model}")
            embedding_logger.info(f"[{operation_id}] Start time: {start_time.isoformat()}")
            
            # Log text preview (first 200 chars)
            text_preview = text[:200] + "..." if len(text) > 200 else text
            embedding_logger.debug(f"[{operation_id}] Text preview: {text_preview}")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log success details
                if hasattr(result, 'shape'):
                    embedding_logger.info(f"[{operation_id}] Embedding generated successfully")
                    embedding_logger.info(f"[{operation_id}] Vector dimension: {result.shape[0]}")
                    embedding_logger.info(f"[{operation_id}] Duration: {duration:.3f}s")
                    embedding_logger.info(f"[{operation_id}] End time: {end_time.isoformat()}")
                    
                    # Log embedding statistics
                    embedding_logger.debug(f"[{operation_id}] Embedding stats - Min: {result.min():.4f}, Max: {result.max():.4f}, Mean: {result.mean():.4f}")
                else:
                    embedding_logger.info(f"[{operation_id}] Operation completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                embedding_logger.error(f"[{operation_id}] Operation failed after {duration:.3f}s")
                embedding_logger.error(f"[{operation_id}] Error: {str(e)}")
                embedding_logger.error(f"[{operation_id}] Error type: {type(e).__name__}")
                
                raise
        
        return wrapper
    return decorator


def log_llm_response(operation_type: str = "llm_generation"):
    """Decorator for logging LLM response generation with prompt context."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
            
            # Extract context from arguments
            prompt = kwargs.get('prompt', '')
            model = kwargs.get('model', 'unknown')
            use_openai = kwargs.get('use_openai', False)
            
            llm_response_logger.info(f"[{operation_id}] Starting {operation_type}")
            llm_response_logger.info(f"[{operation_id}] Model: {model}")
            llm_response_logger.info(f"[{operation_id}] Using OpenAI: {use_openai}")
            llm_response_logger.info(f"[{operation_id}] Prompt length: {len(prompt)} characters")
            llm_response_logger.info(f"[{operation_id}] Start time: {start_time.isoformat()}")
            
            # Log prompt preview
            prompt_preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
            llm_response_logger.debug(f"[{operation_id}] Prompt preview: {prompt_preview}")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log success details
                if isinstance(result, str):
                    llm_response_logger.info(f"[{operation_id}] Response generated successfully")
                    llm_response_logger.info(f"[{operation_id}] Response length: {len(result)} characters")
                    llm_response_logger.info(f"[{operation_id}] Duration: {duration:.3f}s")
                    llm_response_logger.info(f"[{operation_id}] End time: {end_time.isoformat()}")
                    
                    # Log response preview
                    response_preview = result[:200] + "..." if len(result) > 200 else result
                    llm_response_logger.debug(f"[{operation_id}] Response preview: {response_preview}")
                else:
                    llm_response_logger.info(f"[{operation_id}] Operation completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                llm_response_logger.error(f"[{operation_id}] Operation failed after {duration:.3f}s")
                llm_response_logger.error(f"[{operation_id}] Error: {str(e)}")
                llm_response_logger.error(f"[{operation_id}] Error type: {type(e).__name__}")
                
                raise
        
        return wrapper
    return decorator


def log_augmentation_operation(operation_type: str = "context_augmentation"):
    """Decorator for logging context augmentation operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
            
            # Extract context from arguments
            contexts = kwargs.get('contexts', [])
            query = kwargs.get('query', '')
            
            augmentation_logger.info(f"[{operation_id}] Starting {operation_type}")
            augmentation_logger.info(f"[{operation_id}] Query length: {len(query)} characters")
            augmentation_logger.info(f"[{operation_id}] Number of contexts: {len(contexts)}")
            augmentation_logger.info(f"[{operation_id}] Start time: {start_time.isoformat()}")
            
            # Log context details
            for i, context in enumerate(contexts[:5]):  # Log first 5 contexts
                if isinstance(context, dict):
                    context_preview = str(context)[:150] + "..." if len(str(context)) > 150 else str(context)
                    augmentation_logger.debug(f"[{operation_id}] Context {i+1}: {context_preview}")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log success details
                if isinstance(result, str):
                    augmentation_logger.info(f"[{operation_id}] Augmentation completed successfully")
                    augmentation_logger.info(f"[{operation_id}] Augmented context length: {len(result)} characters")
                    augmentation_logger.info(f"[{operation_id}] Duration: {duration:.3f}s")
                    augmentation_logger.info(f"[{operation_id}] End time: {end_time.isoformat()}")
                    
                    # Log augmented context preview
                    result_preview = result[:200] + "..." if len(result) > 200 else result
                    augmentation_logger.debug(f"[{operation_id}] Augmented context preview: {result_preview}")
                else:
                    augmentation_logger.info(f"[{operation_id}] Operation completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                augmentation_logger.error(f"[{operation_id}] Operation failed after {duration:.3f}s")
                augmentation_logger.error(f"[{operation_id}] Error: {str(e)}")
                augmentation_logger.error(f"[{operation_id}] Error type: {type(e).__name__}")
                
                raise
        
        return wrapper
    return decorator


def log_similarity_computation(operation_type: str = "similarity_computation"):
    """Decorator for logging similarity computation operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
            
            # Extract context from arguments (skip 'self' for methods)
            if args and hasattr(args[0], '__class__') and hasattr(args[0], '__dict__'):
                # This is likely a method call, skip 'self'
                emb1 = args[1] if len(args) > 1 else None
                emb2 = args[2] if len(args) > 2 else None
            else:
                emb1 = args[0] if len(args) > 0 else None
                emb2 = args[1] if len(args) > 1 else None
            
            similarity_logger.info(f"[{operation_id}] Starting {operation_type}")
            similarity_logger.info(f"[{operation_id}] Start time: {start_time.isoformat()}")
            
            if emb1 is not None and hasattr(emb1, 'shape'):
                similarity_logger.info(f"[{operation_id}] Embedding 1 dimension: {emb1.shape[0]}")
            if emb2 is not None and hasattr(emb2, 'shape'):
                similarity_logger.info(f"[{operation_id}] Embedding 2 dimension: {emb2.shape[0]}")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log success details
                if isinstance(result, (int, float)):
                    similarity_logger.info(f"[{operation_id}] Similarity computed successfully")
                    similarity_logger.info(f"[{operation_id}] Similarity score: {result:.4f}")
                    similarity_logger.info(f"[{operation_id}] Duration: {duration:.3f}s")
                    similarity_logger.info(f"[{operation_id}] End time: {end_time.isoformat()}")
                else:
                    similarity_logger.info(f"[{operation_id}] Operation completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                similarity_logger.error(f"[{operation_id}] Operation failed after {duration:.3f}s")
                similarity_logger.error(f"[{operation_id}] Error: {str(e)}")
                similarity_logger.error(f"[{operation_id}] Error type: {type(e).__name__}")
                
                raise
        
        return wrapper
    return decorator


def log_retrieval_operation(operation_type: str = "context_retrieval"):
    """Decorator for logging context retrieval operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
            
            # Extract context from arguments (skip 'self' for methods)
            if args and hasattr(args[0], '__class__') and hasattr(args[0], '__dict__'):
                # This is likely a method call, skip 'self'
                query_embedding = args[1] if len(args) > 1 else kwargs.get('query_embedding')
            else:
                query_embedding = args[0] if args else kwargs.get('query_embedding')
            top_k = kwargs.get('top_k', 5)
            
            retrieval_logger.info(f"[{operation_id}] Starting {operation_type}")
            retrieval_logger.info(f"[{operation_id}] Top-K: {top_k}")
            retrieval_logger.info(f"[{operation_id}] Start time: {start_time.isoformat()}")
            
            if query_embedding is not None and hasattr(query_embedding, 'shape'):
                retrieval_logger.info(f"[{operation_id}] Query embedding dimension: {query_embedding.shape[0]}")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log success details
                if isinstance(result, list):
                    retrieval_logger.info(f"[{operation_id}] Retrieval completed successfully")
                    retrieval_logger.info(f"[{operation_id}] Retrieved {len(result)} contexts")
                    retrieval_logger.info(f"[{operation_id}] Duration: {duration:.3f}s")
                    retrieval_logger.info(f"[{operation_id}] End time: {end_time.isoformat()}")
                    
                    # Log similarity scores if available
                    for i, context in enumerate(result[:3]):  # Log first 3 results
                        if isinstance(context, dict) and 'similarity' in context:
                            similarity_logger.debug(f"[{operation_id}] Context {i+1} similarity: {context['similarity']:.4f}")
                else:
                    retrieval_logger.info(f"[{operation_id}] Operation completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                retrieval_logger.error(f"[{operation_id}] Operation failed after {duration:.3f}s")
                retrieval_logger.error(f"[{operation_id}] Error: {str(e)}")
                retrieval_logger.error(f"[{operation_id}] Error type: {type(e).__name__}")
                
                raise
        
        return wrapper
    return decorator


def log_openai_api_call(operation_type: str = "openai_api_call"):
    """Decorator for logging OpenAI API calls with detailed request/response logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
            
            # Extract API call details
            url = kwargs.get('url', 'unknown')
            model = kwargs.get('model', 'unknown')
            prompt = kwargs.get('prompt', '')
            
            llm_response_logger.info(f"[{operation_id}] Starting OpenAI API call")
            llm_response_logger.info(f"[{operation_id}] URL: {url}")
            llm_response_logger.info(f"[{operation_id}] Model: {model}")
            llm_response_logger.info(f"[{operation_id}] Prompt length: {len(prompt)} characters")
            llm_response_logger.info(f"[{operation_id}] Start time: {start_time.isoformat()}")
            
            # Log prompt context for augmentation tracking
            prompt_preview = prompt[:400] + "..." if len(prompt) > 400 else prompt
            augmentation_logger.info(f"[{operation_id}] Data sent to OpenAI with prompt context:")
            augmentation_logger.info(f"[{operation_id}] Prompt preview: {prompt_preview}")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log success details
                llm_response_logger.info(f"[{operation_id}] OpenAI API call successful")
                llm_response_logger.info(f"[{operation_id}] Duration: {duration:.3f}s")
                llm_response_logger.info(f"[{operation_id}] End time: {end_time.isoformat()}")
                
                # Log response details if available
                if hasattr(result, 'status_code'):
                    llm_response_logger.info(f"[{operation_id}] Response status: {result.status_code}")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                llm_response_logger.error(f"[{operation_id}] OpenAI API call failed after {duration:.3f}s")
                llm_response_logger.error(f"[{operation_id}] Error: {str(e)}")
                llm_response_logger.error(f"[{operation_id}] Error type: {type(e).__name__}")
                
                raise
        
        return wrapper
    return decorator


class RAGSessionLogger:
    """Context manager for logging complete RAG sessions."""
    
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.session_id = f"{session_name}_{int(time.time() * 1000)}"
        self.start_time = None
        self.operations = []
    
    def __enter__(self):
        self.start_time = datetime.now()
        
        # Log session start across all loggers
        for logger in [embedding_logger, llm_response_logger, augmentation_logger, similarity_logger, retrieval_logger]:
            logger.info(f"[{self.session_id}] Starting RAG session: {self.session_name}")
            logger.info(f"[{self.session_id}] Session start time: {self.start_time.isoformat()}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Log session end across all loggers
        for logger in [embedding_logger, llm_response_logger, augmentation_logger, similarity_logger, retrieval_logger]:
            if exc_type is None:
                logger.info(f"[{self.session_id}] RAG session completed successfully")
            else:
                logger.error(f"[{self.session_id}] RAG session failed with error: {exc_val}")
            
            logger.info(f"[{self.session_id}] Total session duration: {duration:.3f}s")
            logger.info(f"[{self.session_id}] Session end time: {end_time.isoformat()}")
    
    def log_operation(self, operation_type: str, details: Dict[str, Any]):
        """Log an operation within the session."""
        self.operations.append({
            'type': operation_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })
        
        # Log to appropriate logger
        if operation_type.startswith('embedding'):
            embedding_logger.info(f"[{self.session_id}] {operation_type}: {details}")
        elif operation_type.startswith('llm'):
            llm_response_logger.info(f"[{self.session_id}] {operation_type}: {details}")
        elif operation_type.startswith('augmentation'):
            augmentation_logger.info(f"[{self.session_id}] {operation_type}: {details}")
        elif operation_type.startswith('similarity'):
            similarity_logger.info(f"[{self.session_id}] {operation_type}: {details}")
        elif operation_type.startswith('retrieval'):
            retrieval_logger.info(f"[{self.session_id}] {operation_type}: {details}")


def export_rag_logs_summary(session_id: str = None) -> str:
    """Export a summary of RAG logs for analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"rag_logs_summary_{timestamp}.json"
    summary_path = Path("../../logs") / "rag" / summary_filename  # Go up to project root
    
    # This would collect logs from all RAG loggers and create a summary
    # Implementation would depend on specific requirements
    
    return str(summary_path)
