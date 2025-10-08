"""
Celery tasks for file processing and analysis.
These tasks handle parallel file extraction, parsing, and analysis.
"""

from celery import current_task
from ..utils.code_analyzer import extract_functions_from_zip, analyze_code_structure
from ..utils.docx_processor import DocxProcessor
from ..core.logging import service_logger as logger
from typing import Dict, List, Any
import os
import tempfile

# Import Celery app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from celery_config import celery_app

@celery_app.task(bind=True, name='extract_functions_from_zip')
def extract_functions_from_zip_task(self, zip_path: str, zip_type: str = 'student') -> Dict:
    """
    Extract functions from a ZIP file containing Python code.
    
    Args:
        zip_path: Path to the ZIP file
        zip_type: Type of ZIP ('student' or 'ideal')
    
    Returns:
        Dict containing extracted functions and metadata
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Extracting functions from {zip_type} ZIP...', 'progress': 0}
        )
        
        logger.info(f"Starting function extraction from {zip_type} ZIP: {zip_path}")
        
        # Extract functions
        functions = extract_functions_from_zip(zip_path)
        
        result = {
            'task_id': self.request.id,
            'zip_path': zip_path,
            'zip_type': zip_type,
            'functions': functions,
            'function_count': len(functions),
            'status': 'success'
        }
        
        logger.info(f"Function extraction completed: {len(functions)} functions from {zip_type} ZIP")
        return result
        
    except Exception as e:
        logger.error(f"Function extraction failed for {zip_type} ZIP {zip_path}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='analyze_code_structure')
def analyze_code_structure_task(self, student_code: str, ideal_code: str, function_name: str = None) -> Dict:
    """
    Analyze code structure for a single function.
    
    Args:
        student_code: The student's code
        ideal_code: The ideal code
        function_name: Name of the function being analyzed
    
    Returns:
        Dict containing structure analysis results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Analyzing code structure for {function_name or "function"}...', 'progress': 0}
        )
        
        logger.info(f"Starting code structure analysis for function: {function_name}")
        
        # Analyze code structure
        structure_analysis = analyze_code_structure(student_code, ideal_code)
        
        result = {
            'task_id': self.request.id,
            'function_name': function_name,
            'structure_analysis': structure_analysis,
            'status': 'success'
        }
        
        logger.info(f"Code structure analysis completed for function: {function_name}")
        return result
        
    except Exception as e:
        logger.error(f"Code structure analysis failed for function {function_name}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='process_docx_document')
def process_docx_document_task(self, docx_path: str, doc_type: str = 'student') -> Dict:
    """
    Process a DOCX document to extract Q&A pairs.
    
    Args:
        docx_path: Path to the DOCX file
        doc_type: Type of document ('student' or 'ideal')
    
    Returns:
        Dict containing extracted Q&A pairs and metadata
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Processing {doc_type} DOCX document...', 'progress': 0}
        )
        
        logger.info(f"Starting DOCX processing for {doc_type} document: {docx_path}")
        
        # Process DOCX document
        processor = DocxProcessor()
        qa_pairs = processor.extract_qa_pairs(docx_path)
        
        result = {
            'task_id': self.request.id,
            'docx_path': docx_path,
            'doc_type': doc_type,
            'qa_pairs': qa_pairs,
            'qa_count': len(qa_pairs),
            'status': 'success'
        }
        
        logger.info(f"DOCX processing completed: {len(qa_pairs)} Q&A pairs from {doc_type} document")
        return result
        
    except Exception as e:
        logger.error(f"DOCX processing failed for {doc_type} document {docx_path}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='process_files_parallel')
def process_files_parallel_task(
    self,
    student_zip_path: str = None,
    ideal_zip_path: str = None,
    student_docx_path: str = None,
    ideal_docx_path: str = None
) -> Dict:
    """
    Process multiple files in parallel (ZIP and/or DOCX files).
    
    Args:
        student_zip_path: Path to student ZIP file
        ideal_zip_path: Path to ideal ZIP file
        student_docx_path: Path to student DOCX file
        ideal_docx_path: Path to ideal DOCX file
    
    Returns:
        Dict containing all processing results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting parallel file processing...', 'progress': 0}
        )
        
        logger.info("Starting parallel file processing")
        
        # Create subtasks for each file
        subtasks = []
        
        if student_zip_path:
            subtask = extract_functions_from_zip_task.delay(student_zip_path, 'student')
            subtasks.append(('student_zip', subtask))
        
        if ideal_zip_path:
            subtask = extract_functions_from_zip_task.delay(ideal_zip_path, 'ideal')
            subtasks.append(('ideal_zip', subtask))
        
        if student_docx_path:
            subtask = process_docx_document_task.delay(student_docx_path, 'student')
            subtasks.append(('student_docx', subtask))
        
        if ideal_docx_path:
            subtask = process_docx_document_task.delay(ideal_docx_path, 'ideal')
            subtasks.append(('ideal_docx', subtask))
        
        # Wait for all subtasks to complete
        results = {}
        for i, (file_type, subtask) in enumerate(subtasks):
            try:
                result = subtask.get(timeout=300)  # 5 minute timeout per subtask
                results[file_type] = result
                
                # Update progress
                progress = int((i + 1) / len(subtasks) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={'status': f'Completed {i + 1}/{len(subtasks)} files', 'progress': progress}
                )
                
            except Exception as e:
                logger.error(f"File processing subtask {file_type} failed: {e}")
                results[file_type] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        final_result = {
            'task_id': self.request.id,
            'total_files': len(subtasks),
            'successful': len([r for r in results.values() if r.get('status') == 'success']),
            'failed': len([r for r in results.values() if r.get('status') == 'failed']),
            'results': results,
            'status': 'completed'
        }
        
        logger.info(f"Parallel file processing completed: {final_result['successful']}/{final_result['total_files']} successful")
        return final_result
        
    except Exception as e:
        logger.error(f"Parallel file processing failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

