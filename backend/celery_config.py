"""
Celery configuration for AI Assignment Evaluator.
This module configures Celery for distributed task processing.
"""

from celery import Celery
from kombu import Queue
import os

# Redis configuration (recommended for production)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'ai_assignment_evaluator',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        'app.tasks.embedding_tasks',
        'app.tasks.llm_tasks', 
        'app.tasks.processing_tasks',
        'app.tasks.evaluation_tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'app.tasks.embedding_tasks.*': {'queue': 'embeddings'},
        'app.tasks.llm_tasks.*': {'queue': 'llm'},
        'app.tasks.processing_tasks.*': {'queue': 'processing'},
        'app.tasks.evaluation_tasks.*': {'queue': 'evaluation'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('embeddings', routing_key='embeddings'),
        Queue('llm', routing_key='llm'),
        Queue('processing', routing_key='processing'),
        Queue('evaluation', routing_key='evaluation'),
    ),
    
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Task retry settings
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Optional: Flower monitoring
if os.getenv('ENABLE_FLOWER', 'false').lower() == 'true':
    celery_app.conf.update(
        flower_port=5555,
        flower_basic_auth=os.getenv('FLOWER_AUTH', 'admin:admin')
    )
