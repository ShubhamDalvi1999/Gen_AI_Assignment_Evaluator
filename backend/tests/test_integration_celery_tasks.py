"""
Integration tests for Celery task execution and distributed processing.
Tests the complete Celery pipeline including task distribution, parallel execution, and result aggregation.
"""

import pytest
import asyncio
import tempfile
import os
import sys
import zipfile
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import numpy as np

# Add the backend directory to Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.tasks.evaluation_tasks import (
    evaluate_code_parallel_task,
    evaluate_text_parallel_task,
    generate_embedding_task,
    generate_ideal_embeddings_task,
    process_files_parallel_task,
    generate_feedback_batch_task
)
from app.schemas.evaluate import EmbeddingModel


class TestCeleryTaskIntegration:
    """Integration tests for Celery task execution and distributed processing."""
    
    @pytest.fixture
    def sample_code_files(self):
        """Create sample code files for Celery testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Student code
        student_code = '''
def calculate_mean(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    if not numbers:
        return 0
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]
'''
        
        # Ideal code
        ideal_code = '''
def calculate_mean(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    if not numbers:
        return 0
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]

def calculate_variance(numbers):
    if not numbers:
        return 0
    mean = calculate_mean(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)
'''
        
        # Create ZIP files
        student_zip_path = os.path.join(temp_dir, "student_code.zip")
        ideal_zip_path = os.path.join(temp_dir, "ideal_code.zip")
        
        with zipfile.ZipFile(student_zip_path, 'w') as zipf:
            zipf.writestr("statistics.py", student_code)
        
        with zipfile.ZipFile(ideal_zip_path, 'w') as zipf:
            zipf.writestr("statistics.py", ideal_code)
        
        yield {
            "student_path": student_zip_path,
            "ideal_path": ideal_zip_path,
            "temp_dir": temp_dir
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_parallel_embedding_generation_task(self):
        """Test parallel embedding generation task execution."""
        # Mock embedding generation
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        with patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = mock_embedding
            
            # Mock MongoDB storage
            with patch('app.tasks.embedding_tasks.EmbeddingRepository.store_embedding') as mock_store:
                mock_store.return_value = "test_embedding_id"
                
                # Execute embedding generation task
                result = await generate_embedding_task(
                    text="Test text for parallel processing",
                    model=EmbeddingModel.OPENAI,
                    metadata={"task_id": "test_123"}
                )
                
                # Validate task execution
                assert result is not None
                assert result["embedding_id"] == "test_embedding_id"
                assert result["text"] == "Test text for parallel processing"
                assert result["model"] == EmbeddingModel.OPENAI
                
                # Validate embedding was generated and stored
                mock_get_embedding.assert_called_once()
                mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ideal_embeddings_batch_task(self):
        """Test batch ideal embeddings generation task."""
        # Mock function extraction
        ideal_functions = {
            "calculate_mean": "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
            "calculate_median": "def calculate_median(numbers): return sorted(numbers)[len(numbers)//2]"
        }
        
        with patch('app.tasks.processing_tasks.extract_functions_from_zip') as mock_extract:
            mock_extract.return_value = ideal_functions
            
            # Mock embedding generation
            mock_embedding = np.random.rand(1536).astype(np.float32)
            
            with patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding:
                mock_get_embedding.return_value = mock_embedding
                
                # Mock MongoDB storage
                with patch('app.tasks.embedding_tasks.EmbeddingRepository.store_embedding') as mock_store:
                    mock_store.return_value = "test_embedding_id"
                    
                    # Execute ideal embeddings batch task
                    result = await generate_ideal_embeddings_task(
                        ideal_zip_path="test_ideal.zip",
                        model=EmbeddingModel.OPENAI
                    )
                    
                    # Validate batch execution
                    assert result is not None
                    assert "embeddings" in result
                    assert len(result["embeddings"]) == 2
                    
                    # Validate each function was processed
                    for func_name in ideal_functions.keys():
                        assert func_name in result["embeddings"]
                        assert result["embeddings"][func_name]["embedding_id"] == "test_embedding_id"
    
    @pytest.mark.asyncio
    async def test_parallel_file_processing_task(self):
        """Test parallel file processing task execution."""
        # Mock file processing
        student_functions = {
            "calculate_mean": "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
            "calculate_median": "def calculate_median(numbers): return sorted(numbers)[len(numbers)//2]"
        }
        
        ideal_functions = {
            "calculate_mean": "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
            "calculate_median": "def calculate_median(numbers): return sorted(numbers)[len(numbers)//2]",
            "calculate_variance": "def calculate_variance(numbers): return sum((x - mean) ** 2 for x in numbers) / len(numbers)"
        }
        
        with patch('app.tasks.processing_tasks.extract_functions_from_zip') as mock_extract:
            mock_extract.side_effect = [ideal_functions, student_functions]
            
            # Execute parallel file processing task
            result = await process_files_parallel_task(
                student_zip_path="test_student.zip",
                ideal_zip_path="test_ideal.zip"
            )
            
            # Validate parallel processing results
            assert result is not None
            assert "student_functions" in result
            assert "ideal_functions" in result
            
            # Validate function extraction
            assert len(result["student_functions"]) == 2
            assert len(result["ideal_functions"]) == 3
            assert "calculate_mean" in result["student_functions"]
            assert "calculate_mean" in result["ideal_functions"]
    
    @pytest.mark.asyncio
    async def test_feedback_batch_generation_task(self):
        """Test batch feedback generation task execution."""
        # Mock feedback generation
        mock_feedback = {
            "feedback": "Good implementation!",
            "score": 8.0,
            "suggestions": ["Add validation"],
            "strengths": ["Correct logic"],
            "improvements": ["Error handling"]
        }
        
        with patch('app.tasks.llm_tasks.FeedbackService.generate_code_feedback') as mock_feedback_service:
            mock_feedback_service.return_value = mock_feedback
            
            # Prepare feedback requests
            feedback_requests = [
                {
                    "student_code": "def add(a, b): return a + b",
                    "ideal_code": "def add(a, b): return a + b",
                    "similarity": 0.95,
                    "function_name": "add"
                },
                {
                    "student_code": "def multiply(x, y): return x * y",
                    "ideal_code": "def multiply(x, y): return x * y",
                    "similarity": 0.98,
                    "function_name": "multiply"
                }
            ]
            
            # Execute batch feedback generation task
            result = await generate_feedback_batch_task(
                feedback_requests=feedback_requests,
                model=EmbeddingModel.OPENAI
            )
            
            # Validate batch feedback generation
            assert result is not None
            assert "feedback_results" in result
            assert len(result["feedback_results"]) == 2
            
            # Validate each feedback result
            for feedback_result in result["feedback_results"]:
                assert "function_name" in feedback_result
                assert "feedback" in feedback_result
                assert "score" in feedback_result
                assert feedback_result["score"] == 8.0
    
    @pytest.mark.asyncio
    async def test_complete_code_evaluation_parallel_workflow(self, sample_code_files):
        """Test complete parallel code evaluation workflow."""
        # Mock all external services
        mock_embedding = np.random.rand(1536).astype(np.float32)
        mock_feedback = {
            "feedback": "Good implementation!",
            "score": 8.0,
            "suggestions": ["Add validation"],
            "strengths": ["Correct logic"],
            "improvements": ["Error handling"]
        }
        
        with patch('app.tasks.processing_tasks.extract_functions_from_zip') as mock_extract, \
             patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding, \
             patch('app.tasks.embedding_tasks.EmbeddingRepository.store_embedding') as mock_store, \
             patch('app.tasks.llm_tasks.FeedbackService.generate_code_feedback') as mock_feedback_service:
            
            # Mock function extraction
            student_functions = {
                "calculate_mean": "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
                "calculate_median": "def calculate_median(numbers): return sorted(numbers)[len(numbers)//2]"
            }
            
            ideal_functions = {
                "calculate_mean": "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
                "calculate_median": "def calculate_median(numbers): return sorted(numbers)[len(numbers)//2]",
                "calculate_variance": "def calculate_variance(numbers): return sum((x - mean) ** 2 for x in numbers) / len(numbers)"
            }
            
            mock_extract.side_effect = [ideal_functions, student_functions]
            mock_get_embedding.return_value = mock_embedding
            mock_store.return_value = "test_embedding_id"
            mock_feedback_service.return_value = mock_feedback
            
            # Execute complete parallel evaluation workflow
            result = await evaluate_code_parallel_task(
                student_zip_path=sample_code_files["student_path"],
                ideal_zip_path=sample_code_files["ideal_path"],
                model=EmbeddingModel.OPENAI
            )
            
            # Validate complete workflow results
            assert result is not None
            assert "status" in result
            assert result["status"] == "success"
            assert "overall_score" in result
            assert "evaluations" in result
            assert len(result["evaluations"]) > 0
            
            # Validate parallel execution
            assert mock_extract.call_count == 2  # Called for both files
            assert mock_get_embedding.call_count > 0  # Called for each function
            assert mock_store.call_count > 0  # Called for each embedding
            assert mock_feedback_service.call_count > 0  # Called for each evaluation
    
    @pytest.mark.asyncio
    async def test_celery_task_error_handling(self):
        """Test Celery task error handling and recovery."""
        # Test embedding generation failure
        with patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding:
            mock_get_embedding.side_effect = Exception("Embedding service unavailable")
            
            with pytest.raises(Exception, match="Embedding service unavailable"):
                await generate_embedding_task(
                    text="Test text",
                    model=EmbeddingModel.OPENAI
                )
        
        # Test MongoDB storage failure
        with patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding, \
             patch('app.tasks.embedding_tasks.EmbeddingRepository.store_embedding') as mock_store:
            
            mock_get_embedding.return_value = np.random.rand(1536).astype(np.float32)
            mock_store.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await generate_embedding_task(
                    text="Test text",
                    model=EmbeddingModel.OPENAI
                )
    
    @pytest.mark.asyncio
    async def test_celery_task_retry_mechanism(self):
        """Test Celery task retry mechanism for transient failures."""
        # Mock transient failure followed by success
        call_count = 0
        
        def mock_embedding_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two times
                raise Exception("Temporary service unavailable")
            return np.random.rand(1536).astype(np.float32)
        
        with patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding, \
             patch('app.tasks.embedding_tasks.EmbeddingRepository.store_embedding') as mock_store:
            
            mock_get_embedding.side_effect = mock_embedding_side_effect
            mock_store.return_value = "test_embedding_id"
            
            # Execute task with retry mechanism
            result = await generate_embedding_task(
                text="Test text with retry",
                model=EmbeddingModel.OPENAI
            )
            
            # Validate retry mechanism worked
            assert result is not None
            assert result["embedding_id"] == "test_embedding_id"
            assert call_count == 3  # Failed twice, succeeded on third try
    
    @pytest.mark.asyncio
    async def test_celery_task_parallel_execution_performance(self):
        """Test Celery task parallel execution performance."""
        import time
        
        # Mock services for performance testing
        mock_embedding = np.random.rand(1536).astype(np.float32)
        mock_feedback = {
            "feedback": "Good implementation!",
            "score": 8.0,
            "suggestions": ["Add validation"],
            "strengths": ["Correct logic"],
            "improvements": ["Error handling"]
        }
        
        with patch('app.tasks.processing_tasks.extract_functions_from_zip') as mock_extract, \
             patch('app.tasks.embedding_tasks.get_embedding') as mock_get_embedding, \
             patch('app.tasks.embedding_tasks.EmbeddingRepository.store_embedding') as mock_store, \
             patch('app.tasks.llm_tasks.FeedbackService.generate_code_feedback') as mock_feedback_service:
            
            # Mock function extraction
            student_functions = {
                f"function_{i}": f"def function_{i}(): return {i}"
                for i in range(10)  # 10 functions for performance testing
            }
            
            ideal_functions = {
                f"function_{i}": f"def function_{i}(): return {i}"
                for i in range(10)
            }
            
            mock_extract.side_effect = [ideal_functions, student_functions]
            mock_get_embedding.return_value = mock_embedding
            mock_store.return_value = "test_embedding_id"
            mock_feedback_service.return_value = mock_feedback
            
            # Measure parallel execution time
            start_time = time.time()
            
            result = await evaluate_code_parallel_task(
                student_zip_path="test_student.zip",
                ideal_zip_path="test_ideal.zip",
                model=EmbeddingModel.OPENAI
            )
            
            execution_time = time.time() - start_time
            
            # Validate performance
            assert result["status"] == "success"
            assert execution_time < 60.0  # Should complete within 60 seconds
            
            # Validate parallel execution
            assert mock_get_embedding.call_count >= 10  # Called for each function
            assert mock_store.call_count >= 10  # Called for each embedding
            assert mock_feedback_service.call_count >= 10  # Called for each evaluation
    
    @pytest.mark.asyncio
    async def test_celery_task_result_aggregation(self):
        """Test Celery task result aggregation and final output formatting."""
        # Mock individual task results
        embedding_results = [
            {"embedding_id": f"embedding_{i}", "function_name": f"function_{i}"}
            for i in range(5)
        ]
        
        feedback_results = [
            {
                "function_name": f"function_{i}",
                "feedback": f"Good implementation of function_{i}!",
                "score": 8.0 + i * 0.1
            }
            for i in range(5)
        ]
        
        # Mock task execution results
        with patch('app.tasks.evaluation_tasks.process_files_parallel_task') as mock_process, \
             patch('app.tasks.evaluation_tasks.generate_ideal_embeddings_task') as mock_ideal, \
             patch('app.tasks.evaluation_tasks.generate_embedding_task') as mock_student, \
             patch('app.tasks.evaluation_tasks.generate_feedback_batch_task') as mock_feedback:
            
            mock_process.return_value = {
                "student_functions": {f"function_{i}": f"code_{i}" for i in range(5)},
                "ideal_functions": {f"function_{i}": f"code_{i}" for i in range(5)}
            }
            
            mock_ideal.return_value = {"embeddings": {f"function_{i}": {"embedding_id": f"ideal_{i}"} for i in range(5)}}
            
            mock_student.side_effect = [
                {"embedding_id": f"student_{i}", "function_name": f"function_{i}"}
                for i in range(5)
            ]
            
            mock_feedback.return_value = {"feedback_results": feedback_results}
            
            # Execute complete workflow
            result = await evaluate_code_parallel_task(
                student_zip_path="test_student.zip",
                ideal_zip_path="test_ideal.zip",
                model=EmbeddingModel.OPENAI
            )
            
            # Validate result aggregation
            assert result["status"] == "success"
            assert "overall_score" in result
            assert "evaluations" in result
            assert len(result["evaluations"]) == 5
            
            # Validate individual evaluation results
            for i, evaluation in enumerate(result["evaluations"]):
                assert evaluation["function_name"] == f"function_{i}"
                assert evaluation["score"] == 8.0 + i * 0.1
                assert "feedback" in evaluation
                assert "similarity" in evaluation
            
            # Validate overall score calculation
            expected_overall_score = sum(8.0 + i * 0.1 for i in range(5)) / 5
            assert abs(result["overall_score"] - expected_overall_score) < 0.01
