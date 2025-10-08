#!/usr/bin/env python3
"""
Test script for the RAG logging system.

This script tests the comprehensive RAG logging functionality to ensure
all logging decorators and session logging work correctly.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.core.logging import setup_logging
from app.core.rag_logging import (
    log_embedding_operation,
    log_llm_response,
    log_augmentation_operation,
    log_similarity_computation,
    log_retrieval_operation,
    RAGSessionLogger
)


def test_embedding_logging():
    """Test embedding operation logging."""
    print("Testing embedding logging...")
    
    @log_embedding_operation("test_embedding")
    def test_embedding_function(text: str) -> str:
        """Test function for embedding logging."""
        return f"Generated embedding for: {text[:50]}..."
    
    result = test_embedding_function("This is a test text for embedding generation with sufficient length to test the logging system")
    print(f"Embedding result: {result}")


def test_llm_response_logging():
    """Test LLM response logging."""
    print("Testing LLM response logging...")
    
    @log_llm_response("test_llm_response")
    def test_llm_function(prompt: str) -> str:
        """Test function for LLM response logging."""
        return f"Generated response for prompt: {prompt[:50]}..."
    
    result = test_llm_function("This is a test prompt for LLM response generation with sufficient length to test the logging system")
    print(f"LLM response result: {result}")


def test_similarity_logging():
    """Test similarity computation logging."""
    print("Testing similarity computation logging...")
    
    @log_similarity_computation("test_similarity")
    def test_similarity_function(score: float) -> float:
        """Test function for similarity computation logging."""
        return score * 0.95
    
    result = test_similarity_function(0.85)
    print(f"Similarity result: {result}")


def test_retrieval_logging():
    """Test context retrieval logging."""
    print("Testing context retrieval logging...")
    
    @log_retrieval_operation("test_retrieval")
    def test_retrieval_function(query: str, top_k: int = 5) -> list:
        """Test function for context retrieval logging."""
        return [
            {"context": f"Context {i} for query: {query[:30]}...", "similarity": 0.9 - i*0.1}
            for i in range(top_k)
        ]
    
    result = test_retrieval_function("sample query for testing", top_k=3)
    print(f"Retrieval result: {len(result)} contexts retrieved")


def test_augmentation_logging():
    """Test context augmentation logging."""
    print("Testing context augmentation logging...")
    
    @log_augmentation_operation("test_augmentation")
    def test_augmentation_function(contexts: list, query: str) -> str:
        """Test function for context augmentation logging."""
        context_text = "\n".join([ctx["context"] for ctx in contexts])
        return f"Augmented context for query '{query[:30]}...':\n{context_text[:200]}..."
    
    contexts = [
        {"context": "Sample context 1 for augmentation testing"},
        {"context": "Sample context 2 for augmentation testing"},
        {"context": "Sample context 3 for augmentation testing"}
    ]
    result = test_augmentation_function(contexts, "sample query for augmentation")
    print(f"Augmentation result: {result[:100]}...")


async def test_rag_session_logging():
    """Test RAG session logging."""
    print("Testing RAG session logging...")
    
    with RAGSessionLogger("test_rag_session") as rag_session:
        # Log session start
        rag_session.log_operation("session_start", {
            "timestamp": "2024-01-01T00:00:00",
            "description": "Starting test RAG session"
        })
        
        # Simulate embedding generation
        rag_session.log_operation("embedding_generation", {
            "text_length": 100,
            "model": "test_model"
        })
        
        # Simulate context retrieval
        rag_session.log_operation("context_retrieval", {
            "query": "test query",
            "top_k": 5
        })
        
        # Simulate similarity computation
        rag_session.log_operation("similarity_computation", {
            "contexts_count": 5,
            "similarity_threshold": 0.8
        })
        
        # Simulate context augmentation
        rag_session.log_operation("context_augmentation", {
            "input_contexts": 5,
            "query": "test query"
        })
        
        # Simulate LLM response generation
        rag_session.log_operation("llm_response_generation", {
            "prompt_length": 500,
            "model": "test_model"
        })
        
        # Log session completion
        rag_session.log_operation("session_completion", {
            "total_operations": 5,
            "status": "success"
        })
        
        print("RAG session completed successfully")


def test_log_directory_structure():
    """Test that the log directory structure is created correctly."""
    print("Testing log directory structure...")
    
    logs_dir = Path("logs")
    rag_dirs = [
        "rag/embeddings",
        "rag/llm_responses", 
        "rag/augmentation",
        "rag/similarity",
        "rag/retrieval"
    ]
    
    for dir_path in rag_dirs:
        full_path = logs_dir / dir_path
        if full_path.exists():
            print(f"✅ Directory exists: {full_path}")
        else:
            print(f"❌ Directory missing: {full_path}")
    
    # Check for log files
    for dir_path in rag_dirs:
        full_path = logs_dir / dir_path
        if full_path.exists():
            log_files = list(full_path.glob("*.log"))
            if log_files:
                print(f"📄 Log files found in {full_path}: {len(log_files)} files")
                for log_file in log_files:
                    print(f"   - {log_file.name}")
            else:
                print(f"⚠️  No log files found in {full_path}")


async def main():
    """Main test function."""
    print("🚀 Starting RAG Logging System Tests")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    print("✅ Logging system initialized")
    
    # Test individual decorators
    print("\n📝 Testing Individual Logging Decorators")
    print("-" * 40)
    test_embedding_logging()
    test_llm_response_logging()
    test_similarity_logging()
    test_retrieval_logging()
    test_augmentation_logging()
    
    # Test RAG session logging
    print("\n🎯 Testing RAG Session Logging")
    print("-" * 40)
    await test_rag_session_logging()
    
    # Test directory structure
    print("\n📁 Testing Log Directory Structure")
    print("-" * 40)
    test_log_directory_structure()
    
    print("\n✅ All RAG logging tests completed!")
    print("📊 Check the logs/rag/ directory for detailed logs")
    print("🔍 Each RAG operation type has its own log file with timestamps")


if __name__ == "__main__":
    asyncio.run(main())
