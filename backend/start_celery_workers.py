#!/usr/bin/env python3
"""
Startup script for Celery workers.
This script starts multiple Celery workers for different task types.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def start_celery_worker(queue_name: str, concurrency: int = 2, log_level: str = "INFO"):
    """Start a Celery worker for a specific queue."""
    cmd = [
        "celery", "-A", "celery_config", "worker",
        "--loglevel", log_level,
        "--queues", queue_name,
        "--concurrency", str(concurrency),
        "--hostname", f"{queue_name}@%h"
    ]
    
    print(f"Starting Celery worker for queue '{queue_name}' with concurrency {concurrency}")
    print(f"Command: {' '.join(cmd)}")
    
    return subprocess.Popen(cmd, cwd=os.path.dirname(__file__))

def start_flower():
    """Start Flower monitoring interface."""
    cmd = [
        "celery", "-A", "celery_config", "flower",
        "--port", "5555",
        "--broker", "redis://localhost:6379/0"
    ]
    
    print("Starting Flower monitoring interface on http://localhost:5555")
    print(f"Command: {' '.join(cmd)}")
    
    return subprocess.Popen(cmd, cwd=os.path.dirname(__file__))

def main():
    """Main function to start all Celery workers."""
    print("=" * 60)
    print("Starting AI Assignment Evaluator Celery Workers")
    print("=" * 60)
    
    # Check if Redis is running
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✓ Redis connection successful")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print("Please start Redis server before running Celery workers")
        return
    
    # Start workers for different queues
    workers = []
    
    try:
        # Start embedding worker (high concurrency for parallel embedding generation)
        workers.append(start_celery_worker("embeddings", concurrency=4))
        time.sleep(1)
        
        # Start LLM worker (moderate concurrency for API rate limiting)
        workers.append(start_celery_worker("llm", concurrency=2))
        time.sleep(1)
        
        # Start processing worker (moderate concurrency for file operations)
        workers.append(start_celery_worker("processing", concurrency=3))
        time.sleep(1)
        
        # Start evaluation worker (low concurrency for orchestration)
        workers.append(start_celery_worker("evaluation", concurrency=1))
        time.sleep(1)
        
        # Start default worker (for any unassigned tasks)
        workers.append(start_celery_worker("default", concurrency=2))
        time.sleep(1)
        
        # Start Flower monitoring (optional)
        if os.getenv('ENABLE_FLOWER', 'true').lower() == 'true':
            workers.append(start_flower())
        
        print("\n" + "=" * 60)
        print("All Celery workers started successfully!")
        print("=" * 60)
        print("Workers running:")
        print("- Embeddings worker (4 concurrent tasks)")
        print("- LLM worker (2 concurrent tasks)")
        print("- Processing worker (3 concurrent tasks)")
        print("- Evaluation worker (1 concurrent task)")
        print("- Default worker (2 concurrent tasks)")
        if os.getenv('ENABLE_FLOWER', 'true').lower() == 'true':
            print("- Flower monitoring: http://localhost:5555")
        print("\nPress Ctrl+C to stop all workers")
        print("=" * 60)
        
        # Wait for all workers
        try:
            for worker in workers:
                worker.wait()
        except KeyboardInterrupt:
            print("\nShutting down workers...")
            for worker in workers:
                worker.terminate()
            print("All workers stopped.")
            
    except Exception as e:
        print(f"Error starting workers: {e}")
        # Clean up any started workers
        for worker in workers:
            try:
                worker.terminate()
            except:
                pass

if __name__ == "__main__":
    main()
