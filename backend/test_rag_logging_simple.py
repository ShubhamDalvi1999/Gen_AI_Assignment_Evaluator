#!/usr/bin/env python3
"""
Simple test to verify RAG logging is working correctly.
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.core.rag_logging import start_fresh_rag_session, log_embedding_operation
from app.core.logging import embedding_logger

def test_rag_logging():
    """Test RAG logging functionality."""
    print("Testing RAG logging...")
    
    # Start fresh session
    start_fresh_rag_session()
    
    # Test direct logging
    embedding_logger.info("Direct test log message")
    print("Direct log message sent")
    
    # Test decorator
    @log_embedding_operation("test_operation")
    def test_function(text: str):
        return f"Processed: {text}"
    
    result = test_function("test input")
    print(f"Decorator test result: {result}")
    
    # Check if log files exist and have content
    log_files = [
        "logs/rag/embeddings/embedding_generation.log",
        "logs/rag/llm_responses/llm_generation.log",
        "logs/rag/augmentation/context_augmentation.log",
        "logs/rag/similarity/similarity_computation.log",
        "logs/rag/retrieval/context_retrieval.log"
    ]
    
    print("\nChecking log files:")
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if content:
                    print(f"✅ {log_file}: {len(content)} characters")
                    print(f"   Preview: {content[:100]}...")
                else:
                    print(f"❌ {log_file}: Empty")
        else:
            print(f"❌ {log_file}: Not found")

if __name__ == "__main__":
    test_rag_logging()
