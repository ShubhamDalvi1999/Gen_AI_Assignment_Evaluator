#!/usr/bin/env python3
"""
Celery worker startup script for Windows.
This script starts a Celery worker process for the AI Assignment Evaluator.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import Celery app
from celery_config import celery_app

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Celery Worker for AI Assignment Evaluator")
    print("=" * 60)
    print(f"Backend directory: {backend_dir}")
    print(f"Python path: {sys.path[:3]}...")
    print("=" * 60)
    
    # Start the worker
    celery_app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=2',  # Use 2 workers for Windows
        '--queues=default,embeddings,llm,processing,evaluation',
        '--hostname=worker@%h'
    ])
