#!/usr/bin/env python3
"""
Test script to verify Celery setup is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_celery_imports():
    """Test if Celery can be imported and configured."""
    try:
        print("Testing Celery imports...")
        
        # Test basic Celery import
        from celery import Celery
        print("✅ Celery imported successfully")
        
        # Test Redis import
        import redis
        print("✅ Redis imported successfully")
        
        # Test our Celery config
        from celery_config import celery_app
        print("✅ Celery app configured successfully")
        
        # Test Redis connection
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            print("✅ Redis connection successful")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
        
        # Test Celery broker connection
        try:
            # This will test if Celery can connect to Redis
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            if stats:
                print("✅ Celery broker connection successful")
            else:
                print("⚠️  Celery broker connected but no workers running")
        except Exception as e:
            print(f"❌ Celery broker connection failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install Celery: pip install celery redis")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_task_imports():
    """Test if our Celery tasks can be imported."""
    try:
        print("\nTesting task imports...")
        
        # Test task imports
        from app.tasks.evaluation_tasks import evaluate_code_parallel_task, evaluate_text_parallel_task
        print("✅ Evaluation tasks imported successfully")
        
        from app.tasks.embedding_tasks import generate_embedding_task
        print("✅ Embedding tasks imported successfully")
        
        from app.tasks.llm_tasks import generate_feedback_batch_task
        print("✅ LLM tasks imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Task import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Task error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Celery Setup Test")
    print("=" * 60)
    
    # Test imports
    celery_ok = test_celery_imports()
    tasks_ok = test_task_imports()
    
    print("\n" + "=" * 60)
    if celery_ok and tasks_ok:
        print("✅ All tests passed! Celery is ready to use.")
        print("\nTo start a Celery worker, run:")
        print("  python start_celery_worker.py")
        print("  or")
        print("  start_celery_worker.bat")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not celery_ok:
            print("\nTo install Celery:")
            print("  pip install celery redis")
    print("=" * 60)
