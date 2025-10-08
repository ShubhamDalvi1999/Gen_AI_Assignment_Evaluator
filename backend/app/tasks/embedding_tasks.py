"""
Celery tasks for embedding generation.
These tasks handle parallel embedding generation for code and text.
"""

from celery import current_task
from ..services.embedding_service import get_embedding, EmbeddingModel
from ..core.logging import service_logger as logger
import numpy as np
from typing import Dict, List, Tuple
import json

# Import Celery app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from celery_config import celery_app

@celery_app.task(bind=True, name='generate_embedding')
def generate_embedding_task(self, text: str, model: str, task_id: str = None) -> Dict:
    """
    Generate embedding for a single text/code snippet.
    
    Args:
        text: The text/code to generate embedding for
        model: The embedding model to use ('ollama' or 'openai')
        task_id: Optional task ID for tracking
    
    Returns:
        Dict containing the embedding and metadata
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating embedding...', 'progress': 0}
        )
        
        logger.info(f"Starting embedding generation for task {self.request.id}")
        
        # Generate embedding
        embedding = get_embedding(text, EmbeddingModel(model))
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        result = {
            'task_id': self.request.id,
            'embedding': embedding_list,
            'text_length': len(text),
            'model': model,
            'dimension': len(embedding),
            'status': 'success'
        }
        
        logger.info(f"Embedding generation completed for task {self.request.id}")
        return result
        
    except Exception as e:
        logger.error(f"Embedding generation failed for task {self.request.id}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='generate_embeddings_batch')
def generate_embeddings_batch_task(self, texts: List[str], model: str, task_ids: List[str] = None) -> Dict:
    """
    Generate embeddings for multiple texts in parallel.
    
    Args:
        texts: List of texts to generate embeddings for
        model: The embedding model to use
        task_ids: Optional list of task IDs for tracking
    
    Returns:
        Dict containing all embeddings and metadata
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting batch embedding generation...', 'progress': 0}
        )
        
        logger.info(f"Starting batch embedding generation for {len(texts)} texts")
        
        # Create subtasks for parallel processing
        subtasks = []
        for i, text in enumerate(texts):
            task_id = task_ids[i] if task_ids and i < len(task_ids) else f"batch_{i}"
            subtask = generate_embedding_task.delay(text, model, task_id)
            subtasks.append(subtask)
        
        # Wait for all subtasks to complete
        results = []
        for i, subtask in enumerate(subtasks):
            try:
                result = subtask.get(timeout=300)  # 5 minute timeout per subtask
                results.append(result)
                
                # Update progress
                progress = int((i + 1) / len(subtasks) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={'status': f'Completed {i + 1}/{len(subtasks)} embeddings', 'progress': progress}
                )
                
            except Exception as e:
                logger.error(f"Subtask {i} failed: {e}")
                results.append({
                    'task_id': f"batch_{i}",
                    'error': str(e),
                    'status': 'failed'
                })
        
        final_result = {
            'task_id': self.request.id,
            'total_texts': len(texts),
            'successful': len([r for r in results if r.get('status') == 'success']),
            'failed': len([r for r in results if r.get('status') == 'failed']),
            'results': results,
            'status': 'completed'
        }
        
        logger.info(f"Batch embedding generation completed: {final_result['successful']}/{final_result['total_texts']} successful")
        return final_result
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='generate_ideal_embeddings')
def generate_ideal_embeddings_task(self, functions: Dict[str, str], model: str) -> Dict:
    """
    Generate embeddings for all ideal functions.
    
    Args:
        functions: Dictionary of function names to code
        model: The embedding model to use
    
    Returns:
        Dict containing all ideal embeddings
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating ideal embeddings...', 'progress': 0}
        )
        
        logger.info(f"Generating embeddings for {len(functions)} ideal functions")
        
        # Create subtasks for each function
        subtasks = []
        for func_name, code in functions.items():
            subtask = generate_embedding_task.delay(code, model, func_name)
            subtasks.append((func_name, subtask))
        
        # Collect results
        embeddings = {}
        for i, (func_name, subtask) in enumerate(subtasks):
            try:
                result = subtask.get(timeout=300)
                if result['status'] == 'success':
                    embeddings[func_name] = {
                        'embedding': result['embedding'],
                        'dimension': result['dimension']
                    }
                
                # Update progress
                progress = int((i + 1) / len(subtasks) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={'status': f'Completed {i + 1}/{len(subtasks)} ideal functions', 'progress': progress}
                )
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for ideal function {func_name}: {e}")
        
        final_result = {
            'task_id': self.request.id,
            'total_functions': len(functions),
            'successful_embeddings': len(embeddings),
            'embeddings': embeddings,
            'status': 'completed'
        }
        
        logger.info(f"Ideal embeddings generation completed: {len(embeddings)}/{len(functions)} successful")
        return final_result
        
    except Exception as e:
        logger.error(f"Ideal embeddings generation failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise
