"""
End-to-end tests for complete evaluation workflows.
Tests the entire pipeline from file upload to final evaluation results.
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

from app.services.code_evaluation_service import CodeEvaluationService
from app.services.text_evaluation_service import TextEvaluationService
from app.schemas.evaluate import EmbeddingModel


class TestEndToEndEvaluationWorkflow:
    """End-to-end tests for complete evaluation workflows."""
    
    @pytest.fixture
    def code_evaluation_service(self):
        """Create code evaluation service instance."""
        return CodeEvaluationService()
    
    @pytest.fixture
    def text_evaluation_service(self):
        """Create text evaluation service instance."""
        return TextEvaluationService()
    
    @pytest.fixture
    def sample_code_files(self):
        """Create sample code files for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Student code
        student_code = '''
def calculate_mean(numbers):
    """Calculate the mean of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    """Calculate the median of a list of numbers."""
    if not numbers:
        return 0
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]

def calculate_variance(numbers):
    """Calculate the variance of a list of numbers."""
    if not numbers:
        return 0
    mean = calculate_mean(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)
'''
        
        # Ideal code
        ideal_code = '''
def calculate_mean(numbers):
    """Calculate the mean of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    """Calculate the median of a list of numbers."""
    if not numbers:
        return 0
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]

def calculate_variance(numbers):
    """Calculate the variance of a list of numbers."""
    if not numbers:
        return 0
    mean = calculate_mean(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)

def calculate_std_deviation(numbers):
    """Calculate the standard deviation of a list of numbers."""
    return calculate_variance(numbers) ** 0.5

def calculate_range(numbers):
    """Calculate the range of a list of numbers."""
    if not numbers:
        return 0
    return max(numbers) - min(numbers)
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
    
    @pytest.fixture
    def sample_text_files(self):
        """Create sample text files for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Student text content
        student_text = '''
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that focuses on algorithms.

Q: What is the difference between supervised and unsupervised learning?
A: Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.

Q: What is overfitting?
A: Overfitting occurs when a model performs well on training data but poorly on new data.
'''
        
        # Ideal text content
        ideal_text = '''
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn patterns.

Q: What is the difference between supervised and unsupervised learning?
A: Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds hidden patterns in data without labeled examples. Supervised learning is used for classification and regression, while unsupervised learning is used for clustering and dimensionality reduction.

Q: What is overfitting?
A: Overfitting occurs when a machine learning model learns the training data too well, including noise and irrelevant details, resulting in poor performance on new, unseen data. It can be prevented through techniques like regularization, cross-validation, and early stopping.
'''
        
        # Create DOCX files (simplified for testing)
        student_docx_path = os.path.join(temp_dir, "student_text.docx")
        ideal_docx_path = os.path.join(temp_dir, "ideal_text.docx")
        
        # For testing purposes, we'll create simple text files
        # In real implementation, these would be proper DOCX files
        with open(student_docx_path, 'w', encoding='utf-8') as f:
            f.write(student_text)
        
        with open(ideal_docx_path, 'w', encoding='utf-8') as f:
            f.write(ideal_text)
        
        yield {
            "student_path": student_docx_path,
            "ideal_path": ideal_docx_path,
            "temp_dir": temp_dir
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_code_evaluation_workflow(self, code_evaluation_service, sample_code_files):
        """Test complete code evaluation workflow from file upload to results."""
        # Mock embedding generation
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        with patch('app.services.embedding_service.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = mock_embedding
            
            # Mock feedback generation
            mock_feedback = {
                "feedback": "Good implementation! Consider adding input validation.",
                "score": 8.5,
                "suggestions": ["Add input validation", "Handle edge cases"],
                "strengths": ["Correct logic", "Clean code"],
                "improvements": ["Add validation", "Error handling"]
            }
            
            with patch('app.services.feedback_service.FeedbackService.generate_code_feedback') as mock_feedback_service:
                mock_feedback_service.return_value = mock_feedback
                
                # Mock MongoDB operations
                with patch('app.repositories.embedding_repository.EmbeddingRepository.store_embedding') as mock_store:
                    mock_store.return_value = "test_embedding_id"
                    
                    # Execute complete evaluation workflow
                    result = await code_evaluation_service.evaluate_code(
                        student_path=sample_code_files["student_path"],
                        ideal_path=sample_code_files["ideal_path"],
                        model=EmbeddingModel.OPENAI
                    )
                    
                    # Validate complete workflow results
                    assert result.status == "success"
                    assert result.overall_score > 0
                    assert len(result.evaluations) > 0
                    
                    # Validate individual function evaluations
                    for evaluation in result.evaluations:
                        assert "function_name" in evaluation
                        assert "similarity" in evaluation
                        assert "feedback" in evaluation
                        assert "score" in evaluation
                        assert 0 <= evaluation["similarity"] <= 1
                        assert 0 <= evaluation["score"] <= 10
                        
                        # Validate feedback structure
                        feedback = evaluation["feedback"]
                        assert "feedback" in feedback
                        assert "suggestions" in feedback
                        assert "strengths" in feedback
                        assert "improvements" in feedback
                    
                    # Validate summary feedback
                    assert "summary_feedback" in result
                    assert isinstance(result["summary_feedback"], dict)
                    assert "overall_feedback" in result["summary_feedback"]
                    assert "total_functions" in result["summary_feedback"]
                    assert "average_score" in result["summary_feedback"]
    
    @pytest.mark.asyncio
    async def test_complete_text_evaluation_workflow(self, text_evaluation_service, sample_text_files):
        """Test complete text evaluation workflow from file upload to results."""
        # Mock embedding generation
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        with patch('app.services.embedding_service.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = mock_embedding
            
            # Mock feedback generation
            mock_feedback = {
                "feedback": "Good understanding! Consider expanding on the concepts.",
                "score": 7.8,
                "suggestions": ["Add more detail", "Include examples"],
                "strengths": ["Correct concepts", "Clear explanation"],
                "improvements": ["More depth", "Better examples"]
            }
            
            with patch('app.services.feedback_service.FeedbackService.generate_text_feedback') as mock_feedback_service:
                mock_feedback_service.return_value = mock_feedback
                
                # Mock MongoDB operations
                with patch('app.repositories.text_rag_repository.TextRAGRepository.store_qa_pair') as mock_store:
                    mock_store.return_value = "test_qa_id"
                    
                    # Execute complete evaluation workflow
                    result = await text_evaluation_service.evaluate_text(
                        submission_path=sample_text_files["student_path"],
                        ideal_path=sample_text_files["ideal_path"],
                        model=EmbeddingModel.OPENAI
                    )
                    
                    # Validate complete workflow results
                    assert result.status == "success"
                    assert result.overall_score > 0
                    assert len(result.evaluations) > 0
                    
                    # Validate individual Q&A evaluations
                    for evaluation in result.evaluations:
                        assert "question" in evaluation
                        assert "answer_similarity" in evaluation
                        assert "feedback" in evaluation
                        assert "score" in evaluation
                        assert 0 <= evaluation["answer_similarity"] <= 1
                        assert 0 <= evaluation["score"] <= 10
                        
                        # Validate feedback structure
                        feedback = evaluation["feedback"]
                        assert "feedback" in feedback
                        assert "suggestions" in feedback
                        assert "strengths" in feedback
                        assert "improvements" in feedback
                    
                    # Validate summary feedback
                    assert "summary_feedback" in result
                    assert isinstance(result["summary_feedback"], dict)
                    assert "overall_feedback" in result["summary_feedback"]
                    assert "total_questions" in result["summary_feedback"]
                    assert "average_score" in result["summary_feedback"]
    
    @pytest.mark.asyncio
    async def test_evaluation_workflow_with_error_handling(self, code_evaluation_service, sample_code_files):
        """Test evaluation workflow error handling and recovery."""
        # Test with invalid file paths
        with pytest.raises(Exception):
            await code_evaluation_service.evaluate_code(
                student_path="invalid_path.zip",
                ideal_path="invalid_path.zip",
                model=EmbeddingModel.OPENAI
            )
        
        # Test with embedding service failure
        with patch('app.services.embedding_service.get_embedding') as mock_get_embedding:
            mock_get_embedding.side_effect = Exception("Embedding service unavailable")
            
            result = await code_evaluation_service.evaluate_code(
                student_path=sample_code_files["student_path"],
                ideal_path=sample_code_files["ideal_path"],
                model=EmbeddingModel.OPENAI
            )
            
            # Should handle error gracefully
            assert result.status == "error"
            assert "error" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_evaluation_workflow_performance(self, code_evaluation_service, sample_code_files):
        """Test evaluation workflow performance with timing measurements."""
        import time
        
        # Mock all external services for performance testing
        mock_embedding = np.random.rand(1536).astype(np.float32)
        mock_feedback = {
            "feedback": "Good implementation!",
            "score": 8.0,
            "suggestions": ["Add validation"],
            "strengths": ["Correct logic"],
            "improvements": ["Error handling"]
        }
        
        with patch('app.services.embedding_service.get_embedding') as mock_get_embedding, \
             patch('app.services.feedback_service.FeedbackService.generate_code_feedback') as mock_feedback_service, \
             patch('app.repositories.embedding_repository.EmbeddingRepository.store_embedding') as mock_store:
            
            mock_get_embedding.return_value = mock_embedding
            mock_feedback_service.return_value = mock_feedback
            mock_store.return_value = "test_id"
            
            # Measure evaluation time
            start_time = time.time()
            
            result = await code_evaluation_service.evaluate_code(
                student_path=sample_code_files["student_path"],
                ideal_path=sample_code_files["ideal_path"],
                model=EmbeddingModel.OPENAI
            )
            
            evaluation_time = time.time() - start_time
            
            # Validate performance
            assert result.status == "success"
            assert evaluation_time < 30.0  # Should complete within 30 seconds
            
            # Validate timing information in result
            assert "evaluation_time" in result
            assert result["evaluation_time"] > 0
    
    @pytest.mark.asyncio
    async def test_evaluation_workflow_data_persistence(self, code_evaluation_service, sample_code_files):
        """Test that evaluation data is properly persisted to MongoDB."""
        # Mock embedding generation
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        with patch('app.services.embedding_service.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = mock_embedding
            
            # Mock feedback generation
            mock_feedback = {
                "feedback": "Good implementation!",
                "score": 8.0,
                "suggestions": ["Add validation"],
                "strengths": ["Correct logic"],
                "improvements": ["Error handling"]
            }
            
            with patch('app.services.feedback_service.FeedbackService.generate_code_feedback') as mock_feedback_service:
                mock_feedback_service.return_value = mock_feedback
                
                # Mock MongoDB storage with verification
                stored_embeddings = []
                
                def mock_store_embedding(text, embedding, model, metadata=None):
                    stored_embeddings.append({
                        "text": text,
                        "embedding": embedding,
                        "model": model,
                        "metadata": metadata
                    })
                    return f"embedding_{len(stored_embeddings)}"
                
                with patch('app.repositories.embedding_repository.EmbeddingRepository.store_embedding', side_effect=mock_store_embedding):
                    # Execute evaluation
                    result = await code_evaluation_service.evaluate_code(
                        student_path=sample_code_files["student_path"],
                        ideal_path=sample_code_files["ideal_path"],
                        model=EmbeddingModel.OPENAI
                    )
                    
                    # Validate data persistence
                    assert result.status == "success"
                    assert len(stored_embeddings) > 0
                    
                    # Validate stored embedding data
                    for stored_data in stored_embeddings:
                        assert "text" in stored_data
                        assert "embedding" in stored_data
                        assert "model" in stored_data
                        assert stored_data["model"] == EmbeddingModel.OPENAI
                        assert isinstance(stored_data["embedding"], np.ndarray)
                        assert stored_data["embedding"].shape == (1536,)
    
    @pytest.mark.asyncio
    async def test_evaluation_workflow_with_different_models(self, code_evaluation_service, sample_code_files):
        """Test evaluation workflow with different embedding models."""
        models_to_test = [EmbeddingModel.OPENAI, EmbeddingModel.OLLAMA]
        
        for model in models_to_test:
            # Mock embedding generation for different models
            embedding_size = 1536 if model == EmbeddingModel.OPENAI else 4096
            mock_embedding = np.random.rand(embedding_size).astype(np.float32)
            
            with patch('app.services.embedding_service.get_embedding') as mock_get_embedding:
                mock_get_embedding.return_value = mock_embedding
                
                # Mock feedback generation
                mock_feedback = {
                    "feedback": f"Good implementation with {model.value}!",
                    "score": 8.0,
                    "suggestions": ["Add validation"],
                    "strengths": ["Correct logic"],
                    "improvements": ["Error handling"]
                }
                
                with patch('app.services.feedback_service.FeedbackService.generate_code_feedback') as mock_feedback_service:
                    mock_feedback_service.return_value = mock_feedback
                    
                    with patch('app.repositories.embedding_repository.EmbeddingRepository.store_embedding') as mock_store:
                        mock_store.return_value = f"test_id_{model.value}"
                        
                        # Execute evaluation
                        result = await code_evaluation_service.evaluate_code(
                            student_path=sample_code_files["student_path"],
                            ideal_path=sample_code_files["ideal_path"],
                            model=model
                        )
                        
                        # Validate results for each model
                        assert result.status == "success"
                        assert result.overall_score > 0
                        assert len(result.evaluations) > 0
                        
                        # Validate that correct model was used
                        call_args = mock_store.call_args
                        assert call_args[1]["model"] == model
    
    @pytest.mark.asyncio
    async def test_evaluation_workflow_result_export(self, code_evaluation_service, sample_code_files):
        """Test evaluation workflow result export functionality."""
        # Mock all services
        mock_embedding = np.random.rand(1536).astype(np.float32)
        mock_feedback = {
            "feedback": "Good implementation!",
            "score": 8.0,
            "suggestions": ["Add validation"],
            "strengths": ["Correct logic"],
            "improvements": ["Error handling"]
        }
        
        with patch('app.services.embedding_service.get_embedding') as mock_get_embedding, \
             patch('app.services.feedback_service.FeedbackService.generate_code_feedback') as mock_feedback_service, \
             patch('app.repositories.embedding_repository.EmbeddingRepository.store_embedding') as mock_store:
            
            mock_get_embedding.return_value = mock_embedding
            mock_feedback_service.return_value = mock_feedback
            mock_store.return_value = "test_id"
            
            # Execute evaluation
            result = await code_evaluation_service.evaluate_code(
                student_path=sample_code_files["student_path"],
                ideal_path=sample_code_files["ideal_path"],
                model=EmbeddingModel.OPENAI
            )
            
            # Test JSON export
            json_export = result.to_json()
            assert isinstance(json_export, str)
            
            # Validate JSON can be parsed
            parsed_json = json.loads(json_export)
            assert "status" in parsed_json
            assert "overall_score" in parsed_json
            assert "evaluations" in parsed_json
            
            # Test CSV export (if implemented)
            if hasattr(result, 'to_csv'):
                csv_export = result.to_csv()
                assert isinstance(csv_export, str)
                assert "function_name" in csv_export
                assert "score" in csv_export
            
            # Test TXT export (if implemented)
            if hasattr(result, 'to_txt'):
                txt_export = result.to_txt()
                assert isinstance(txt_export, str)
                assert len(txt_export) > 0
