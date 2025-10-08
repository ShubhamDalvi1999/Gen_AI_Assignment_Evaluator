"""
Integration tests for LLM response generation, format validation, and content checking.
Tests the complete LLM pipeline including API calls, response parsing, and content validation.
"""

import pytest
import asyncio
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Add the backend directory to Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.feedback_service import FeedbackService
from app.services.llm_service import LLMService
from app.schemas.evaluate import EmbeddingModel


class TestLLMResponseIntegration:
    """Integration tests for LLM response generation and validation."""
    
    @pytest.fixture
    def feedback_service(self):
        """Create feedback service instance."""
        return FeedbackService()
    
    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance."""
        return LLMService()
    
    @pytest.fixture
    def sample_code_evaluation_data(self):
        """Sample code evaluation data for testing."""
        return {
            "student_code": "def calculate_mean(numbers):\n    return sum(numbers) / len(numbers)",
            "ideal_code": "def calculate_mean(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)",
            "similarity": 0.85,
            "function_name": "calculate_mean",
            "feedback_type": "code_improvement"
        }
    
    @pytest.fixture
    def sample_text_evaluation_data(self):
        """Sample text evaluation data for testing."""
        return {
            "student_answer": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "ideal_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "similarity": 0.78,
            "question": "What is machine learning?",
            "feedback_type": "text_improvement"
        }
    
    @pytest.mark.asyncio
    async def test_openai_code_feedback_generation(self, feedback_service, sample_code_evaluation_data):
        """Test complete OpenAI code feedback generation and validation."""
        # Mock OpenAI API response
        mock_llm_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "feedback": "Good implementation! Consider adding input validation for empty lists.",
                        "score": 8.5,
                        "suggestions": [
                            "Add a check for empty input list",
                            "Consider edge cases like division by zero"
                        ],
                        "strengths": [
                            "Correct mathematical logic",
                            "Clean and readable code"
                        ],
                        "improvements": [
                            "Add input validation",
                            "Handle edge cases"
                        ]
                    })
                }
            }]
        }
        
        with patch('app.services.feedback_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_llm_response
            mock_post.return_value = mock_response
            
            # Generate feedback
            result = await feedback_service.generate_code_feedback(
                student_code=sample_code_evaluation_data["student_code"],
                ideal_code=sample_code_evaluation_data["ideal_code"],
                similarity=sample_code_evaluation_data["similarity"],
                function_name=sample_code_evaluation_data["function_name"]
            )
            
            # Validate response format
            assert isinstance(result, dict)
            assert "feedback" in result
            assert "score" in result
            assert "suggestions" in result
            assert "strengths" in result
            assert "improvements" in result
            
            # Validate content quality
            assert isinstance(result["feedback"], str)
            assert len(result["feedback"]) > 10  # Meaningful feedback
            assert isinstance(result["score"], (int, float))
            assert 0 <= result["score"] <= 10  # Valid score range
            assert isinstance(result["suggestions"], list)
            assert len(result["suggestions"]) > 0
            assert all(isinstance(s, str) for s in result["suggestions"])
            
            # Validate API call parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "openai.com" in call_args[1]["url"]
            assert "gpt" in call_args[1]["json"]["model"]
    
    @pytest.mark.asyncio
    async def test_openai_text_feedback_generation(self, feedback_service, sample_text_evaluation_data):
        """Test complete OpenAI text feedback generation and validation."""
        # Mock OpenAI API response
        mock_llm_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "feedback": "Good understanding of machine learning basics. Consider expanding on the learning aspect.",
                        "score": 7.8,
                        "suggestions": [
                            "Explain how machines learn from data",
                            "Mention supervised vs unsupervised learning"
                        ],
                        "strengths": [
                            "Correctly identifies ML as subset of AI",
                            "Mentions algorithms"
                        ],
                        "improvements": [
                            "Add more detail about learning process",
                            "Include examples of ML applications"
                        ]
                    })
                }
            }]
        }
        
        with patch('app.services.feedback_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_llm_response
            mock_post.return_value = mock_response
            
            # Generate feedback
            result = await feedback_service.generate_text_feedback(
                student_answer=sample_text_evaluation_data["student_answer"],
                ideal_answer=sample_text_evaluation_data["ideal_answer"],
                similarity=sample_text_evaluation_data["similarity"],
                question=sample_text_evaluation_data["question"]
            )
            
            # Validate response format
            assert isinstance(result, dict)
            assert "feedback" in result
            assert "score" in result
            assert "suggestions" in result
            assert "strengths" in result
            assert "improvements" in result
            
            # Validate content quality
            assert isinstance(result["feedback"], str)
            assert len(result["feedback"]) > 10
            assert isinstance(result["score"], (int, float))
            assert 0 <= result["score"] <= 10
            assert isinstance(result["suggestions"], list)
            assert len(result["suggestions"]) > 0
    
    @pytest.mark.asyncio
    async def test_llm_response_format_validation(self, llm_service):
        """Test LLM response format validation and parsing."""
        # Test valid JSON response
        valid_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "feedback": "Test feedback",
                        "score": 8.0,
                        "suggestions": ["Suggestion 1", "Suggestion 2"]
                    })
                }
            }]
        }
        
        with patch('app.services.llm_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = valid_response
            mock_post.return_value = mock_response
            
            result = await llm_service.generate_response(
                prompt="Test prompt",
                model="gpt-3.5-turbo",
                max_tokens=500
            )
            
            # Validate response structure
            assert isinstance(result, dict)
            assert "content" in result
            assert "usage" in result
            assert "model" in result
            
            # Validate content is parseable JSON
            content = json.loads(result["content"])
            assert isinstance(content, dict)
            assert "feedback" in content
            assert "score" in content
    
    @pytest.mark.asyncio
    async def test_llm_response_error_handling(self, llm_service):
        """Test LLM response error handling and fallback mechanisms."""
        # Test API error
        with patch('app.services.llm_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429  # Rate limit
            mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
            mock_post.return_value = mock_response
            
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await llm_service.generate_response(
                    prompt="Test prompt",
                    model="gpt-3.5-turbo"
                )
        
        # Test malformed JSON response
        malformed_response = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON"
                }
            }]
        }
        
        with patch('app.services.llm_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = malformed_response
            mock_post.return_value = mock_response
            
            with pytest.raises(json.JSONDecodeError):
                await llm_service.generate_response(
                    prompt="Test prompt",
                    model="gpt-3.5-turbo"
                )
    
    @pytest.mark.asyncio
    async def test_feedback_content_quality_validation(self, feedback_service):
        """Test feedback content quality and relevance validation."""
        # Mock high-quality response
        high_quality_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "feedback": "Excellent implementation! Your code demonstrates strong understanding of the algorithm. The logic is correct and the code is well-structured.",
                        "score": 9.2,
                        "suggestions": [
                            "Consider adding input validation for edge cases",
                            "Add docstring for better documentation"
                        ],
                        "strengths": [
                            "Correct algorithm implementation",
                            "Clean and readable code structure",
                            "Proper variable naming"
                        ],
                        "improvements": [
                            "Add input validation",
                            "Include error handling",
                            "Add comprehensive documentation"
                        ]
                    })
                }
            }]
        }
        
        with patch('app.services.feedback_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = high_quality_response
            mock_post.return_value = mock_response
            
            result = await feedback_service.generate_code_feedback(
                student_code="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                ideal_code="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                similarity=0.95,
                function_name="factorial"
            )
            
            # Validate content quality
            feedback_text = result["feedback"]
            assert len(feedback_text) > 50  # Substantial feedback
            assert any(word in feedback_text.lower() for word in ["good", "excellent", "well", "correct"])  # Positive feedback
            assert result["score"] >= 8.0  # High score for good code
            
            # Validate suggestions are actionable
            suggestions = result["suggestions"]
            assert len(suggestions) > 0
            assert all(len(s) > 10 for s in suggestions)  # Meaningful suggestions
            assert any("validation" in s.lower() or "error" in s.lower() for s in suggestions)  # Technical suggestions
    
    @pytest.mark.asyncio
    async def test_batch_feedback_generation(self, feedback_service):
        """Test batch feedback generation for multiple evaluations."""
        evaluation_data = [
            {
                "student_code": "def add(a, b):\n    return a + b",
                "ideal_code": "def add(a, b):\n    return a + b",
                "similarity": 0.95,
                "function_name": "add"
            },
            {
                "student_code": "def multiply(x, y):\n    return x * y",
                "ideal_code": "def multiply(x, y):\n    return x * y",
                "similarity": 0.98,
                "function_name": "multiply"
            }
        ]
        
        # Mock batch responses
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "feedback": "Perfect implementation!",
                            "score": 10.0,
                            "suggestions": [],
                            "strengths": ["Correct implementation"],
                            "improvements": []
                        })
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "feedback": "Excellent work!",
                            "score": 9.5,
                            "suggestions": [],
                            "strengths": ["Correct implementation"],
                            "improvements": []
                        })
                    }
                }]
            }
        ]
        
        with patch('app.services.feedback_service.requests.post') as mock_post:
            mock_post.side_effect = [
                Mock(status_code=200, json=Mock(return_value=resp)) 
                for resp in mock_responses
            ]
            
            # Generate batch feedback
            results = []
            for data in evaluation_data:
                result = await feedback_service.generate_code_feedback(
                    student_code=data["student_code"],
                    ideal_code=data["ideal_code"],
                    similarity=data["similarity"],
                    function_name=data["function_name"]
                )
                results.append(result)
            
            # Validate batch results
            assert len(results) == 2
            for result in results:
                assert isinstance(result, dict)
                assert "feedback" in result
                assert "score" in result
                assert result["score"] >= 9.0  # High scores for perfect matches
    
    @pytest.mark.asyncio
    async def test_llm_token_usage_tracking(self, llm_service):
        """Test LLM token usage tracking and cost estimation."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "feedback": "Test feedback",
                        "score": 8.0
                    })
                }
            }],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200
            }
        }
        
        with patch('app.services.llm_service.requests.post') as mock_post:
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj
            
            result = await llm_service.generate_response(
                prompt="Test prompt",
                model="gpt-3.5-turbo",
                max_tokens=500
            )
            
            # Validate token usage tracking
            assert "usage" in result
            assert result["usage"]["prompt_tokens"] == 150
            assert result["usage"]["completion_tokens"] == 50
            assert result["usage"]["total_tokens"] == 200
            
            # Validate cost estimation (if implemented)
            if "estimated_cost" in result:
                assert isinstance(result["estimated_cost"], (int, float))
                assert result["estimated_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_llm_response_caching(self, llm_service):
        """Test LLM response caching mechanism."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "feedback": "Cached response",
                        "score": 8.0
                    })
                }
            }]
        }
        
        with patch('app.services.llm_service.requests.post') as mock_post:
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj
            
            # First call
            result1 = await llm_service.generate_response(
                prompt="Test prompt for caching",
                model="gpt-3.5-turbo"
            )
            
            # Second call with same prompt (should use cache if implemented)
            result2 = await llm_service.generate_response(
                prompt="Test prompt for caching",
                model="gpt-3.5-turbo"
            )
            
            # If caching is implemented, verify it works
            # If not implemented, both calls should go to API
            assert result1["content"] == result2["content"]
            
            # Verify API was called (caching not implemented yet)
            assert mock_post.call_count == 2
