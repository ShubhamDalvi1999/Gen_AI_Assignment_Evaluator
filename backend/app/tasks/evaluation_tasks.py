"""
Celery tasks for complete evaluation workflows.
These tasks orchestrate the entire evaluation process using other task modules.
"""

from celery import current_task, group, chain
from ..services.code_evaluation_service import CodeEvaluationService
from ..services.text_evaluation_service import TextEvaluationService
from ..services.embedding_service import EmbeddingModel
from ..core.logging import service_logger as logger
from typing import Dict, List, Any
import os

# Import other task modules
from .embedding_tasks import generate_ideal_embeddings_task, generate_embedding_task
from .llm_tasks import generate_feedback_batch_task, generate_summary_feedback_task
from .processing_tasks import process_files_parallel_task

# Import Celery app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from celery_config import celery_app

@celery_app.task(bind=True, name='evaluate_code_parallel')
def evaluate_code_parallel_task(
    self,
    student_zip_path: str,
    ideal_zip_path: str,
    model: str = 'ollama',
    use_openai_feedback: bool = False
) -> Dict:
    """
    Complete code evaluation using parallel processing.
    
    Args:
        student_zip_path: Path to student ZIP file
        ideal_zip_path: Path to ideal ZIP file
        model: Embedding model to use
        use_openai_feedback: Whether to use OpenAI for feedback
    
    Returns:
        Dict containing complete evaluation results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting parallel code evaluation...', 'progress': 0}
        )
        
        logger.info("Starting parallel code evaluation")
        
        # Step 1: Process files in parallel
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Processing ZIP files...', 'progress': 10}
        )
        
        file_processing_result = process_files_parallel_task.delay(
            student_zip_path=student_zip_path,
            ideal_zip_path=ideal_zip_path
        ).get(timeout=300)
        
        if file_processing_result['status'] != 'completed':
            raise Exception("File processing failed")
        
        student_functions = file_processing_result['results']['student_zip']['functions']
        ideal_functions = file_processing_result['results']['ideal_zip']['functions']
        
        # Step 2: Generate ideal embeddings in parallel
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating ideal embeddings...', 'progress': 30}
        )
        
        ideal_embeddings_result = generate_ideal_embeddings_task.delay(
            ideal_functions, model
        ).get(timeout=600)
        
        if ideal_embeddings_result['status'] != 'completed':
            raise Exception("Ideal embeddings generation failed")
        
        # Step 3: Generate student embeddings in parallel (but wait for completion)
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating student embeddings...', 'progress': 50}
        )
        
        # Create tasks for student embeddings - run in parallel
        student_embedding_tasks = []
        for func_name, code in student_functions.items():
            if func_name in ideal_functions:
                task = generate_embedding_task.delay(code, model, func_name)
                student_embedding_tasks.append((func_name, task))
        
        # Wait for ALL student embeddings to complete before proceeding
        student_embeddings = {}
        for func_name, task in student_embedding_tasks:
            try:
                result = task.get(timeout=300)
                if result['status'] == 'success':
                    student_embeddings[func_name] = result['embedding']
                else:
                    logger.error(f"Failed to generate embedding for {func_name}")
            except Exception as e:
                logger.error(f"Failed to process function {func_name}: {e}")
        
        # Step 4: Calculate similarities (sequential after embeddings are ready)
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Calculating similarities...', 'progress': 70}
        )
        
        similarities = {}
        for func_name in student_embeddings:
            if func_name in ideal_embeddings_result['embeddings']:
                try:
                    from ..services.embedding_service import compute_similarity
                    import numpy as np
                    
                    student_embedding = np.array(student_embeddings[func_name])
                    ideal_embedding = np.array(ideal_embeddings_result['embeddings'][func_name]['embedding'])
                    similarity = compute_similarity(student_embedding, ideal_embedding)
                    
                    similarities[func_name] = {
                        'similarity': similarity,
                        'student_code': student_functions[func_name],
                        'ideal_code': ideal_functions[func_name]
                    }
                except Exception as e:
                    logger.error(f"Failed to calculate similarity for {func_name}: {e}")
        
        # Step 5: Generate feedback in parallel (only after similarities are ready)
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating feedback...', 'progress': 80}
        )
        
        # Prepare feedback requests
        feedback_requests = []
        for func_name, data in similarities.items():
            # Simple structure analysis (can be enhanced)
            structure_analysis = {
                'student_lines': len(data['student_code'].split('\n')),
                'ideal_lines': len(data['ideal_code'].split('\n')),
                'similarity': data['similarity']
            }
            
            feedback_requests.append({
                'student_code': data['student_code'],
                'ideal_code': data['ideal_code'],
                'similarity': data['similarity'],
                'structure_analysis': structure_analysis,
                'similar_contexts': [],  # Can be enhanced with RAG
                'use_openai': use_openai_feedback,
                'function_name': func_name
            })
        
        # Generate feedback in parallel
        feedback_result = generate_feedback_batch_task.delay(
            feedback_requests, 'code'
        ).get(timeout=900)  # 15 minute timeout for feedback
        
        # Step 5: Compile final results
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Compiling results...', 'progress': 95}
        )
        
        # Calculate overall score
        total_functions = len(ideal_functions)
        matched_functions = len(similarities)
        average_similarity = sum(s['similarity'] for s in similarities.values()) / matched_functions if matched_functions > 0 else 0
        
        # Compile function results
        function_results = []
        for i, result in enumerate(feedback_result['results']):
            if result['status'] == 'success':
                func_name = result['function_name']
                function_results.append({
                    'function_name': func_name,
                    'similarity': similarities[func_name]['similarity'],
                    'feedback': result['feedback'],
                    'status': 'evaluated'
                })
        
        final_result = {
            'task_id': self.request.id,
            'status': 'success',
            'total_functions': total_functions,
            'matched_functions': matched_functions,
            'average_similarity': average_similarity,
            'function_results': function_results,
            'model_used': model,
            'use_openai_feedback': use_openai_feedback,
            'processing_time': 'calculated_in_service'
        }
        
        self.update_state(
            state='SUCCESS',
            meta={'status': 'Evaluation completed successfully', 'progress': 100}
        )
        
        logger.info("Parallel code evaluation completed successfully")
        return final_result
        
    except Exception as e:
        logger.error(f"Parallel code evaluation failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

@celery_app.task(bind=True, name='evaluate_text_parallel')
def evaluate_text_parallel_task(
    self,
    student_docx_path: str,
    ideal_docx_path: str,
    model: str = 'ollama'
) -> Dict:
    """
    Complete text evaluation using parallel processing.
    
    Args:
        student_docx_path: Path to student DOCX file
        ideal_docx_path: Path to ideal DOCX file
        model: Embedding model to use
    
    Returns:
        Dict containing complete evaluation results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting parallel text evaluation...', 'progress': 0}
        )
        
        logger.info("Starting parallel text evaluation")
        
        # Step 1: Process DOCX files in parallel
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Processing DOCX files...', 'progress': 20}
        )
        
        file_processing_result = process_files_parallel_task.delay(
            student_docx_path=student_docx_path,
            ideal_docx_path=ideal_docx_path
        ).get(timeout=300)
        
        if file_processing_result['status'] != 'completed':
            raise Exception("File processing failed")
        
        student_qa_pairs = file_processing_result['results']['student_docx']['qa_pairs']
        ideal_qa_pairs = file_processing_result['results']['ideal_docx']['qa_pairs']
        
        # Step 2: Generate embeddings for all questions in parallel
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating question embeddings...', 'progress': 40}
        )
        
        # Create embedding tasks for ideal questions
        ideal_question_tasks = {}
        for qa_pair in ideal_qa_pairs:
            question = qa_pair['question']
            task = generate_embedding_task.delay(question, model, f"ideal_q_{qa_pair['id']}")
            ideal_question_tasks[qa_pair['id']] = (question, task)
        
        # Create embedding tasks for student questions
        student_question_tasks = {}
        for qa_pair in student_qa_pairs:
            question = qa_pair['question']
            task = generate_embedding_task.delay(question, model, f"student_q_{qa_pair['id']}")
            student_question_tasks[qa_pair['id']] = (question, task)
        
        # Wait for all question embeddings
        ideal_question_embeddings = {}
        for qa_id, (question, task) in ideal_question_tasks.items():
            try:
                result = task.get(timeout=300)
                if result['status'] == 'success':
                    ideal_question_embeddings[qa_id] = np.array(result['embedding'])
            except Exception as e:
                logger.error(f"Failed to generate embedding for ideal question {qa_id}: {e}")
        
        student_question_embeddings = {}
        for qa_id, (question, task) in student_question_tasks.items():
            try:
                result = task.get(timeout=300)
                if result['status'] == 'success':
                    student_question_embeddings[qa_id] = np.array(result['embedding'])
            except Exception as e:
                logger.error(f"Failed to generate embedding for student question {qa_id}: {e}")
        
        # Step 3: Match questions (sequential after embeddings are ready)
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Matching questions...', 'progress': 60}
        )
        
        # Match student questions to ideal questions based on embeddings
        question_matches = {}
        for student_id, student_embedding in student_question_embeddings.items():
            best_match_id = None
            best_similarity = 0
            
            for ideal_id, ideal_embedding in ideal_question_embeddings.items():
                from ..services.embedding_service import compute_similarity
                similarity = compute_similarity(student_embedding, ideal_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = ideal_id
            
            if best_match_id and best_similarity > 0.7:  # Threshold for matching
                question_matches[student_id] = {
                    'ideal_id': best_match_id,
                    'similarity': best_similarity
                }
        
        # Step 4: Generate answer embeddings for matched pairs
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Matching questions and generating answer embeddings...', 'progress': 60}
        )
        
        # Match questions and prepare answer comparison tasks
        answer_comparison_tasks = []
        for student_qa in student_qa_pairs:
            student_q_embedding = student_question_embeddings.get(student_qa['id'])
            if student_q_embedding is None:
                continue
            
            # Find best matching ideal question
            best_match_id = None
            best_similarity = 0
            for ideal_qa in ideal_qa_pairs:
                ideal_q_embedding = ideal_question_embeddings.get(ideal_qa['id'])
                if ideal_q_embedding is None:
                    continue
                
                from ..services.embedding_service import compute_similarity
                similarity = compute_similarity(student_q_embedding, ideal_q_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = ideal_qa['id']
            
            if best_match_id and best_similarity > 0.7:  # Threshold for matching
                # Generate embeddings for answers
                student_answer = student_qa['answer']
                ideal_answer = next(qa['answer'] for qa in ideal_qa_pairs if qa['id'] == best_match_id)
                
                answer_comparison_tasks.append({
                    'student_answer': student_answer,
                    'ideal_answer': ideal_answer,
                    'question_similarity': best_similarity,
                    'student_qa_id': student_qa['id'],
                    'ideal_qa_id': best_match_id
                })
        
        # Step 4: Generate answer embeddings and compare
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Comparing answers...', 'progress': 80}
        )
        
        answer_similarities = []
        for comparison in answer_comparison_tasks:
            try:
                # Generate embeddings for answers
                student_answer_task = generate_embedding_task.delay(
                    comparison['student_answer'], model, f"student_a_{comparison['student_qa_id']}"
                )
                ideal_answer_task = generate_embedding_task.delay(
                    comparison['ideal_answer'], model, f"ideal_a_{comparison['ideal_qa_id']}"
                )
                
                student_answer_result = student_answer_task.get(timeout=300)
                ideal_answer_result = ideal_answer_task.get(timeout=300)
                
                if student_answer_result['status'] == 'success' and ideal_answer_result['status'] == 'success':
                    from ..services.embedding_service import compute_similarity
                    import numpy as np
                    
                    student_a_embedding = np.array(student_answer_result['embedding'])
                    ideal_a_embedding = np.array(ideal_answer_result['embedding'])
                    answer_similarity = compute_similarity(student_a_embedding, ideal_a_embedding)
                    
                    answer_similarities.append({
                        'student_qa_id': comparison['student_qa_id'],
                        'ideal_qa_id': comparison['ideal_qa_id'],
                        'question_similarity': comparison['question_similarity'],
                        'answer_similarity': answer_similarity,
                        'student_answer': comparison['student_answer'],
                        'ideal_answer': comparison['ideal_answer']
                    })
            except Exception as e:
                logger.error(f"Failed to compare answers: {e}")
        
        # Step 5: Generate feedback in parallel
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating feedback...', 'progress': 90}
        )
        
        # Prepare feedback requests
        feedback_requests = []
        for comparison in answer_similarities:
            feedback_requests.append({
                'student_answer': comparison['student_answer'],
                'reference_answer': comparison['ideal_answer'],
                'similarity': comparison['answer_similarity'],
                'use_openai': False,  # Can be made configurable
                'question_id': f"q_{comparison['student_qa_id']}"
            })
        
        # Generate feedback in parallel
        feedback_result = generate_feedback_batch_task.delay(
            feedback_requests, 'text'
        ).get(timeout=900)
        
        # Step 6: Compile final results
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Compiling results...', 'progress': 95}
        )
        
        # Calculate overall score
        total_questions = len(ideal_qa_pairs)
        matched_questions = len(answer_similarities)
        average_similarity = sum(s['answer_similarity'] for s in answer_similarities) / matched_questions if matched_questions > 0 else 0
        
        # Compile question results
        question_results = []
        for i, result in enumerate(feedback_result['results']):
            if result['status'] == 'success':
                question_results.append({
                    'question_id': result['question_id'],
                    'similarity': answer_similarities[i]['answer_similarity'],
                    'feedback': result['feedback'],
                    'status': 'evaluated'
                })
        
        final_result = {
            'task_id': self.request.id,
            'status': 'success',
            'total_questions': total_questions,
            'matched_questions': matched_questions,
            'average_similarity': average_similarity,
            'question_results': question_results,
            'model_used': model,
            'processing_time': 'calculated_in_service'
        }
        
        self.update_state(
            state='SUCCESS',
            meta={'status': 'Text evaluation completed successfully', 'progress': 100}
        )
        
        logger.info("Parallel text evaluation completed successfully")
        return final_result
        
    except Exception as e:
        logger.error(f"Parallel text evaluation failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'failed'}
        )
        raise

