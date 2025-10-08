#!/usr/bin/env python3
"""
Test script to verify the fresh RAG logging system.
This script tests that logs are cleared and recreated for each evaluation session.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.core.rag_logging import start_fresh_rag_session, log_embedding_operation, log_llm_response
from app.core.logging import clear_rag_logs
import numpy as np

def test_embedding_generation():
    """Test embedding generation with logging."""
    @log_embedding_operation("test_embedding")
    def generate_test_embedding(text: str):
        """Generate a test embedding."""
        # Simulate embedding generation
        return np.random.rand(384)  # Simulate 384-dimensional embedding
    
    # Generate test embedding
    embedding = generate_test_embedding("This is a test text for embedding generation")
    print(f"Generated embedding with shape: {embedding.shape}")
    return embedding

def test_llm_response():
    """Test LLM response generation with logging."""
    @log_llm_response("test_llm_response")
    def generate_test_response(prompt: str):
        """Generate a test LLM response."""
        # Simulate LLM response
        return f"Test response for prompt: {prompt[:50]}..."
    
    # Generate test response
    response = generate_test_response("This is a test prompt for LLM response generation")
    print(f"Generated response: {response}")
    return response

def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Fresh RAG Logging System")
    print("=" * 60)
    
    # Test 1: Start fresh session
    print("\n1. Starting fresh RAG session...")
    start_fresh_rag_session()
    
    # Test 2: Generate embeddings with logging
    print("\n2. Testing embedding generation with logging...")
    embedding = test_embedding_generation()
    
    # Test 3: Generate LLM response with logging
    print("\n3. Testing LLM response generation with logging...")
    response = test_llm_response()
    
    # Test 4: Check log files
    print("\n4. Checking log files...")
    log_files = [
        "logs/rag/embeddings/embedding_generation.log",
        "logs/rag/llm_responses/llm_generation.log"
    ]
    
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            print(f"✓ Log file exists: {log_file}")
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Content length: {len(content)} characters")
                print(f"  First line: {content.split(chr(10))[0] if content else 'Empty'}")
        else:
            print(f"✗ Log file missing: {log_file}")
    
    # Test 5: Start another fresh session (should clear logs)
    print("\n5. Starting another fresh session (should clear previous logs)...")
    start_fresh_rag_session()
    
    # Test 6: Generate more logs
    print("\n6. Generating more logs after clearing...")
    test_embedding_generation()
    test_llm_response()
    
    # Test 7: Check log files again
    print("\n7. Checking log files after clearing...")
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            print(f"✓ Log file exists: {log_file}")
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Content length: {len(content)} characters")
                print(f"  First line: {content.split(chr(10))[0] if content else 'Empty'}")
        else:
            print(f"✗ Log file missing: {log_file}")
    
    print("\n" + "=" * 60)
    print("Fresh RAG Logging System Test Completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
