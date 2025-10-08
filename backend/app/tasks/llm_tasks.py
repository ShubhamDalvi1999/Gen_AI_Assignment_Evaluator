"""
Celery tasks for LLM operations.
These tasks handle parallel LLM calls for feedback generation.
"""

from celery import current_task
from ..services.feedback_service import generate_code_feedback, generate_text_feedback, generate_summary_feedback
from ..core.logging import service_logger as logger
from typing import Dict, List, Any
import json

# Import Celery app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from celery_config import celery_app

@celery_app.task(bind=True, name='generate_code_feedback')
def generate_code_feedback_task(
    self, 
    student_code: str, 
    ideal_code: str, 
    similarity: float, 
    structure_analysis: Dict[str, Any], 
    similar_contexts: List[Dict[str, Any]],
    use_openai: bool = False,
    function_name: str = None
) -> Dict:
    """
    Generate feedback for a single code function.
    
    Args:
        student_code: The student's code
        ideal_code: The ideal code
        similarity: Similarity score
        structure_analysis: Code structure analysis
        similar_contexts: Similar code contexts
        use_openai: Whether to use OpenAI or Ollama
        function_name: Name of the function being evaluated
    
    Returns:
        Dict containing the feedback and metadata
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Generating feedback for {function_name or "function"}...', 'progress': 0}
        )
        
        logger.info(f"Starting code feedback generation for function: {function_name}")
        
        # Generate feedback
        feedback = generate_code_feedback(
            student_code, 
            ideal_code, 
            similarity, 
            structure_analysis, 
            similar_contexts,
            use_openai=use_openai
        )
        
        result = {
            'task_id': self.request.id,
            'function_name': function_name,
            'feedback': feedback,
            'similarity': similarity,
            'use_openai': use_openai,
            'feedback_length': len(feedback),
            'status': 'success'
        }
        
        logger.info(f"Code feedback generation completed for function: {function_name}")
        return result
        
    except Exception as e:
        logger.error(f"Code feedback generation failed for function {function_name}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='generate_text_feedback')
def generate_text_feedback_task(
    self,
    student_answer: str,
    reference_answer: str,
    similarity: float,
    use_openai: bool = False,
    question_id: str = None
) -> Dict:
    """
    Generate feedback for a text Q&A pair.
    
    Args:
        student_answer: The student's answer
        reference_answer: The reference answer
        similarity: Similarity score
        use_openai: Whether to use OpenAI or Ollama
        question_id: ID of the question being evaluated
    
    Returns:
        Dict containing the feedback and metadata
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Generating feedback for question {question_id or "Q&A"}...', 'progress': 0}
        )
        
        logger.info(f"Starting text feedback generation for question: {question_id}")
        
        # Generate feedback
        feedback = generate_text_feedback(
            student_answer,
            reference_answer,
            similarity,
            use_openai=use_openai
        )
        
        result = {
            'task_id': self.request.id,
            'question_id': question_id,
            'feedback': feedback,
            'similarity': similarity,
            'use_openai': use_openai,
            'feedback_length': len(feedback),
            'status': 'success'
        }
        
        logger.info(f"Text feedback generation completed for question: {question_id}")
        return result
        
    except Exception as e:
        logger.error(f"Text feedback generation failed for question {question_id}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='generate_feedback_batch')
def generate_feedback_batch_task(
    self,
    feedback_requests: List[Dict[str, Any]],
    feedback_type: str = 'code'  # 'code' or 'text'
) -> Dict:
    """
    Generate feedback for multiple items in parallel.
    
    Args:
        feedback_requests: List of feedback request dictionaries
        feedback_type: Type of feedback ('code' or 'text')
    
    Returns:
        Dict containing all feedback results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Starting batch {feedback_type} feedback generation...', 'progress': 0}
        )
        
        logger.info(f"Starting batch {feedback_type} feedback generation for {len(feedback_requests)} items")
        
        # Create subtasks based on feedback type
        subtasks = []
        for i, request in enumerate(feedback_requests):
            if feedback_type == 'code':
                subtask = generate_code_feedback_task.delay(
                    request['student_code'],
                    request['ideal_code'],
                    request['similarity'],
                    request['structure_analysis'],
                    request['similar_contexts'],
                    request.get('use_openai', False),
                    request.get('function_name', f'function_{i}')
                )
            else:  # text feedback
                subtask = generate_text_feedback_task.delay(
                    request['student_answer'],
                    request['reference_answer'],
                    request['similarity'],
                    request.get('use_openai', False),
                    request.get('question_id', f'question_{i}')
                )
            subtasks.append(subtask)
        
        # Wait for all subtasks to complete
        results = []
        for i, subtask in enumerate(subtasks):
            try:
                result = subtask.get(timeout=600)  # 10 minute timeout per subtask
                results.append(result)
                
                # Update progress
                progress = int((i + 1) / len(subtasks) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={'status': f'Completed {i + 1}/{len(subtasks)} feedback items', 'progress': progress}
                )
                
            except Exception as e:
                logger.error(f"Feedback subtask {i} failed: {e}")
                results.append({
                    'task_id': f"batch_{i}",
                    'error': str(e),
                    'status': 'failed'
                })
        
        final_result = {
            'task_id': self.request.id,
            'feedback_type': feedback_type,
            'total_requests': len(feedback_requests),
            'successful': len([r for r in results if r.get('status') == 'success']),
            'failed': len([r for r in results if r.get('status') == 'failed']),
            'results': results,
            'status': 'completed'
        }
        
        logger.info(f"Batch {feedback_type} feedback generation completed: {final_result['successful']}/{final_result['total_requests']} successful")
        return final_result
        
    except Exception as e:
        logger.error(f"Batch {feedback_type} feedback generation failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='generate_summary_feedback')
def generate_summary_feedback_task(
    self,
    evaluation_summary: str,
    use_openai: bool = False
) -> Dict:
    """
    Generate overall summary feedback for an evaluation.
    
    Args:
        evaluation_summary: Summary of the evaluation results
        use_openai: Whether to use OpenAI or Ollama
    
    Returns:
        Dict containing the summary feedback
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating summary feedback...', 'progress': 0}
        )
        
        logger.info("Starting summary feedback generation")
        
        # Generate summary feedback
        summary_feedback = generate_summary_feedback(
            evaluation_summary,
            use_openai=use_openai
        )
        
        result = {
            'task_id': self.request.id,
            'summary_feedback': summary_feedback,
            'use_openai': use_openai,
            'feedback_length': len(summary_feedback),
            'status': 'success'
        }
        
        logger.info("Summary feedback generation completed")
        return result
        
    except Exception as e:
        logger.error(f"Summary feedback generation failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

