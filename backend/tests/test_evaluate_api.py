import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from app.main import app

client = TestClient(app)


class TestEvaluateAPI:
    """Test class for evaluation API endpoints."""
    
    def test_code_evaluation_invalid_file_extension(self):
        """Test code evaluation with invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/evaluate/code",
                        files={
                            "submission": ("test.txt", f, "text/plain"),
                            "ideal": ("test.txt", f, "text/plain")
                        },
                        data={"model": "ollama"}
                    )
                
                assert response.status_code == 400
                assert "must be a ZIP file" in response.json()["detail"]
            finally:
                os.unlink(temp_file.name)
    
    def test_code_evaluation_empty_file(self):
        """Test code evaluation with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            # Create empty file
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/evaluate/code",
                        files={
                            "submission": ("test.zip", f, "application/zip"),
                            "ideal": ("test.zip", f, "application/zip")
                        },
                        data={"model": "ollama"}
                    )
                
                # Should fail due to empty file or invalid ZIP
                assert response.status_code in [400, 500]
            finally:
                os.unlink(temp_file.name)
    
    def test_text_evaluation_invalid_file_extension(self):
        """Test text evaluation with invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/evaluate/text",
                        files={
                            "submission": ("test.txt", f, "text/plain"),
                            "ideal": ("test.txt", f, "text/plain")
                        },
                        data={"model": "ollama"}
                    )
                
                assert response.status_code == 400
                assert "must be a DOCX file" in response.json()["detail"]
            finally:
                os.unlink(temp_file.name)
    
    def test_token_estimation_empty_file(self):
        """Test token estimation with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            # Create empty file
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/estimate/tokens",
                        files={
                            "submission": ("test.zip", f, "application/zip"),
                            "ideal": ("test.zip", f, "application/zip")
                        },
                        data={"model": "ollama"}
                    )
                
                assert response.status_code == 400
                assert "empty" in response.json()["detail"]
            finally:
                os.unlink(temp_file.name)
    
    @patch('app.services.code_evaluation_service.CodeEvaluationService.evaluate_code')
    def test_code_evaluation_success_mock(self, mock_evaluate):
        """Test successful code evaluation with mocked service."""
        from app.schemas.evaluate import CodeEvaluationResult
        
        # Mock the service response
        mock_evaluate.return_value = CodeEvaluationResult(
            status="success",
            functions_evaluated=2,
            average_similarity=0.85,
            function_results=[
                {"name": "test_func", "similarity": 0.85, "status": "Correct"}
            ],
            extra_functions=[],
            missing_functions=[]
        )
        
        # Create a small ZIP file with valid content
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            # Write ZIP signature
            temp_file.write(b'PK\x03\x04')
            temp_file.write(b'\x00' * 100)  # Add some content
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/evaluate/code",
                        files={
                            "submission": ("test.zip", f, "application/zip"),
                            "ideal": ("test.zip", f, "application/zip")
                        },
                        data={"model": "ollama"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "functions_evaluated" in data
            finally:
                os.unlink(temp_file.name)
    
    @patch('app.services.text_evaluation_service.TextEvaluationService.evaluate_text')
    def test_text_evaluation_success_mock(self, mock_evaluate):
        """Test successful text evaluation with mocked service."""
        from app.schemas.evaluate import TextEvaluationResult
        
        # Mock the service response
        mock_evaluate.return_value = TextEvaluationResult(
            status="success",
            session_id="test_session_123",
            matched_questions=3,
            average_similarity=0.78,
            processed_questions=[],
            model_used="ollama",
            overall_score=78.0,
            evaluations=[],
            summary="Good performance overall",
            stats={"total_questions": 3, "high_count": 2, "medium_count": 1, "low_count": 0, "poor_count": 0, "missing_count": 0}
        )
        
        # Create a temporary DOCX-like file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            # Write minimal content
            temp_file.write(b'test docx content')
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/evaluate/text",
                        files={
                            "submission": ("test.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                            "ideal": ("test.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                        },
                        data={"model": "ollama"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "session_id" in data
                assert "overall_score" in data
            finally:
                os.unlink(temp_file.name)
    
    @patch('app.services.token_estimation_service.TokenEstimationService.estimate_tokens')
    def test_token_estimation_success_mock(self, mock_estimate):
        """Test successful token estimation with mocked service."""
        from app.schemas.evaluate import TokenEstimateResult
        
        # Mock the service response
        mock_estimate.return_value = TokenEstimateResult(
            status="success",
            message="Token estimation completed",
            estimated_tokens=1500,
            cost_estimate=0.0015,
            warnings=[]
        )
        
        # Create a temporary file with content
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_file.write(b'test content for tokens')
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/estimate/tokens",
                        files={
                            "submission": ("test.zip", f, "application/zip"),
                            "ideal": ("test.zip", f, "application/zip")
                        },
                        data={"model": "ollama"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "estimated_tokens" in data
                assert data["estimated_tokens"] == 1500
            finally:
                os.unlink(temp_file.name)
    
    def test_invalid_model_parameter(self):
        """Test API endpoints with invalid model parameter."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_file.write(b'PK\x03\x04' + b'\x00' * 100)
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = client.post(
                        "/api/v1/evaluate/code",
                        files={
                            "submission": ("test.zip", f, "application/zip"),
                            "ideal": ("test.zip", f, "application/zip")
                        },
                        data={"model": "invalid_model"}
                    )
                
                # Should fail with validation error
                assert response.status_code == 422
            finally:
                os.unlink(temp_file.name)
