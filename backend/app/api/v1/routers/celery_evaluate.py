"""
Celery-based evaluation endpoints for parallel processing.
These endpoints use Celery tasks for distributed evaluation.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
import tempfile
import os
import uuid
from datetime import datetime

from ....services.embedding_service import EmbeddingModel
from ....core.logging import service_logger as logger

try:
    from ....tasks.evaluation_tasks import evaluate_code_parallel_task, evaluate_text_parallel_task
except ImportError as e:
    logger.error(f"Failed to import Celery tasks: {e}")
    # Create dummy tasks for now
    evaluate_code_parallel_task = None
    evaluate_text_parallel_task = None

router = APIRouter(prefix="/api/v1/celery", tags=["celery-evaluation"])

# Store task results temporarily (in production, use Redis or database)
task_results = {}

@router.post("/evaluate/code")
async def evaluate_code_celery(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA),
    use_openai_feedback: bool = Form(False)
) -> Dict[str, Any]:
    """
    Evaluate code using Celery for parallel processing.
    
    This endpoint:
    1. Saves uploaded files temporarily
    2. Starts a Celery task for parallel evaluation
    3. Returns a task ID for tracking progress
    """
    try:
        # Check if Celery tasks are available
        if evaluate_code_parallel_task is None:
            raise HTTPException(
                status_code=503, 
                detail="Celery tasks are not available. Please ensure Redis is running and Celery workers are started."
            )
        
        # Generate unique evaluation ID
        evaluation_id = str(uuid.uuid4())
        logger.info(f"Starting Celery code evaluation {evaluation_id}")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as student_temp:
            student_content = await submission.read()
            student_temp.write(student_content)
            student_zip_path = student_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as ideal_temp:
            ideal_content = await ideal.read()
            ideal_temp.write(ideal_content)
            ideal_zip_path = ideal_temp.name
        
        # Start Celery task
        task = evaluate_code_parallel_task.delay(
            student_zip_path=student_zip_path,
            ideal_zip_path=ideal_zip_path,
            model=model.value,
            use_openai_feedback=use_openai_feedback
        )
        
        # Store task information
        task_results[evaluation_id] = {
            'task_id': task.id,
            'status': 'PENDING',
            'started_at': datetime.now().isoformat(),
            'student_zip_path': student_zip_path,
            'ideal_zip_path': ideal_zip_path,
            'model': model.value,
            'use_openai_feedback': use_openai_feedback
        }
        
        logger.info(f"Celery task started: {task.id} for evaluation {evaluation_id}")
        
        return {
            "evaluation_id": evaluation_id,
            "task_id": task.id,
            "status": "PENDING",
            "message": "Evaluation started. Use the task_id to check progress.",
            "check_status_url": f"/api/v1/celery/status/{evaluation_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Celery code evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")

@router.post("/evaluate/text")
async def evaluate_text_celery(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA)
) -> Dict[str, Any]:
    """
    Evaluate text Q&A using Celery for parallel processing.
    
    This endpoint:
    1. Saves uploaded DOCX files temporarily
    2. Starts a Celery task for parallel evaluation
    3. Returns a task ID for tracking progress
    """
    try:
        # Check if Celery tasks are available
        if evaluate_text_parallel_task is None:
            raise HTTPException(
                status_code=503, 
                detail="Celery tasks are not available. Please ensure Redis is running and Celery workers are started."
            )
        
        # Generate unique evaluation ID
        evaluation_id = str(uuid.uuid4())
        logger.info(f"Starting Celery text evaluation {evaluation_id}")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as student_temp:
            student_content = await submission.read()
            student_temp.write(student_content)
            student_docx_path = student_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as ideal_temp:
            ideal_content = await ideal.read()
            ideal_temp.write(ideal_content)
            ideal_docx_path = ideal_temp.name
        
        # Start Celery task
        task = evaluate_text_parallel_task.delay(
            student_docx_path=student_docx_path,
            ideal_docx_path=ideal_docx_path,
            model=model.value
        )
        
        # Store task information
        task_results[evaluation_id] = {
            'task_id': task.id,
            'status': 'PENDING',
            'started_at': datetime.now().isoformat(),
            'student_docx_path': student_docx_path,
            'ideal_docx_path': ideal_docx_path,
            'model': model.value
        }
        
        logger.info(f"Celery task started: {task.id} for evaluation {evaluation_id}")
        
        return {
            "evaluation_id": evaluation_id,
            "task_id": task.id,
            "status": "PENDING",
            "message": "Evaluation started. Use the task_id to check progress.",
            "check_status_url": f"/api/v1/celery/status/{evaluation_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Celery text evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")

@router.get("/status/{evaluation_id}")
async def get_evaluation_status(evaluation_id: str) -> Dict[str, Any]:
    """
    Get the status of a Celery evaluation task.
    
    Args:
        evaluation_id: The evaluation ID returned from the evaluation endpoint
    
    Returns:
        Dict containing task status and results (if completed)
    """
    try:
        if evaluation_id not in task_results:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        task_info = task_results[evaluation_id]
        task_id = task_info['task_id']
        
        # Import Celery app to check task status
        from ...tasks.evaluation_tasks import evaluate_code_parallel_task, evaluate_text_parallel_task
        
        # Get task result
        task = evaluate_code_parallel_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'evaluation_id': evaluation_id,
                'task_id': task_id,
                'status': 'PENDING',
                'message': 'Task is waiting to be processed...'
            }
        elif task.state == 'PROGRESS':
            response = {
                'evaluation_id': evaluation_id,
                'task_id': task_id,
                'status': 'PROGRESS',
                'message': task.info.get('status', 'Processing...'),
                'progress': task.info.get('progress', 0)
            }
        elif task.state == 'SUCCESS':
            # Task completed successfully
            result = task.result
            response = {
                'evaluation_id': evaluation_id,
                'task_id': task_id,
                'status': 'SUCCESS',
                'message': 'Evaluation completed successfully',
                'result': result
            }
            
            # Clean up temporary files
            try:
                if 'student_zip_path' in task_info:
                    os.unlink(task_info['student_zip_path'])
                if 'ideal_zip_path' in task_info:
                    os.unlink(task_info['ideal_zip_path'])
                if 'student_docx_path' in task_info:
                    os.unlink(task_info['student_docx_path'])
                if 'ideal_docx_path' in task_info:
                    os.unlink(task_info['ideal_docx_path'])
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
            
            # Remove from task_results (optional, for memory management)
            del task_results[evaluation_id]
            
        else:  # FAILURE
            response = {
                'evaluation_id': evaluation_id,
                'task_id': task_id,
                'status': 'FAILURE',
                'message': 'Task failed',
                'error': str(task.info)
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get evaluation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/tasks")
async def list_active_tasks() -> Dict[str, Any]:
    """
    List all active evaluation tasks.
    
    Returns:
        Dict containing information about all active tasks
    """
    try:
        active_tasks = []
        for evaluation_id, task_info in task_results.items():
            active_tasks.append({
                'evaluation_id': evaluation_id,
                'task_id': task_info['task_id'],
                'started_at': task_info['started_at'],
                'model': task_info.get('model', 'unknown')
            })
        
        return {
            'total_tasks': len(active_tasks),
            'active_tasks': active_tasks
        }
        
    except Exception as e:
        logger.error(f"Failed to list active tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.delete("/tasks/{evaluation_id}")
async def cancel_evaluation(evaluation_id: str) -> Dict[str, Any]:
    """
    Cancel a running evaluation task.
    
    Args:
        evaluation_id: The evaluation ID to cancel
    
    Returns:
        Dict containing cancellation status
    """
    try:
        if evaluation_id not in task_results:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        task_info = task_results[evaluation_id]
        task_id = task_info['task_id']
        
        # Import Celery app to revoke task
        from ...tasks.evaluation_tasks import evaluate_code_parallel_task
        
        # Revoke the task
        evaluate_code_parallel_task.control.revoke(task_id, terminate=True)
        
        # Clean up temporary files
        try:
            if 'student_zip_path' in task_info:
                os.unlink(task_info['student_zip_path'])
            if 'ideal_zip_path' in task_info:
                os.unlink(task_info['ideal_zip_path'])
            if 'student_docx_path' in task_info:
                os.unlink(task_info['student_docx_path'])
            if 'ideal_docx_path' in task_info:
                os.unlink(task_info['ideal_docx_path'])
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")
        
        # Remove from task_results
        del task_results[evaluation_id]
        
        return {
            'evaluation_id': evaluation_id,
            'task_id': task_id,
            'status': 'CANCELLED',
            'message': 'Evaluation cancelled successfully'
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel evaluation: {str(e)}")

