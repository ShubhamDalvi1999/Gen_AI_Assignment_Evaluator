import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import numpy as np

from app.services.embedding_service import get_embedding, compute_similarity, EmbeddingModel
from app.services.text_evaluation_service import TextEvaluationService
from app.services.token_estimation_service import TokenEstimationService
from app.schemas.evaluate import EmbeddingModel as SchemaEmbeddingModel


class TestEmbeddingService:
    """Test class for embedding service."""
    
    @patch('app.services.embedding_service.requests.post')
    def test_get_embedding_ollama_success(self, mock_post):
        """Test successful embedding generation with Ollama."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        mock_post.return_value = mock_response
        
        embedding = get_embedding("test text", EmbeddingModel.OLLAMA)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 5
        assert embedding[0] == 0.1
    
    @patch('app.services.embedding_service.requests.post')
    def test_get_embedding_ollama_failure(self, mock_post):
        """Test embedding generation failure with Ollama."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError):
            get_embedding("test text", EmbeddingModel.OLLAMA)
    
    def test_compute_similarity(self):
        """Test cosine similarity computation."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])
        
        similarity = compute_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 1e-6  # Should be exactly 1.0
        
        # Test orthogonal vectors
        emb3 = np.array([0.0, 1.0, 0.0])
        similarity = compute_similarity(emb1, emb3)
        assert abs(similarity - 0.0) < 1e-6  # Should be exactly 0.0


class TestTextEvaluationService:
    """Test class for text evaluation service."""
    
    def setUp(self):
        self.service = TextEvaluationService()
    
    @patch('app.services.text_evaluation_service.TextEvaluationService._map_qa_pairs')
    @patch('app.repositories.text_rag_repository.TextRAGRepository.process_qa_document')
    @pytest.mark.asyncio
    async def test_evaluate_text_success(self, mock_process_qa, mock_map_qa):
        """Test successful text evaluation."""
        self.setUp()
        
        # Mock Q&A processing
        mock_process_qa.side_effect = [
            # Ideal Q&A pairs
            {
                "ideal_1": {
                    "question": "What is AI?",
                    "answer": "Artificial Intelligence is...",
                    "embedding": np.array([0.1, 0.2, 0.3]),
                    "question_embedding": np.array([0.4, 0.5, 0.6])
                }
            },
            # Student Q&A pairs
            {
                "student_1_1": {
                    "question": "What is AI?",
                    "answer": "AI is artificial intelligence...",
                    "embedding": np.array([0.1, 0.2, 0.3]),
                    "question_embedding": np.array([0.4, 0.5, 0.6])
                }
            }
        ]
        
        # Mock Q&A mapping
        mock_map_qa.return_value = [
            {
                "student_qa_id": "student_1_1",
                "ideal_qa_id": "ideal_1",
                "question_similarity": 0.95,
                "answer_similarity": 0.85,
                "similarity": 0.85,
                "quality": "high",
                "student_question": "What is AI?",
                "student_answer": "AI is artificial intelligence..."
            }
        ]
        
        # Create temporary files
        sub_file_path = tempfile.mktemp(suffix='.docx')
        ideal_file_path = tempfile.mktemp(suffix='.docx')
        
        try:
            with open(sub_file_path, 'wb') as sub_file:
                sub_file.write(b'test content')
            with open(ideal_file_path, 'wb') as ideal_file:
                ideal_file.write(b'test content')
            
            result = await self.service.evaluate_text(
                submission_path=sub_file_path,
                ideal_path=ideal_file_path,
                model=SchemaEmbeddingModel.OLLAMA
            )
            
            assert result.status == "success"
            assert result.overall_score > 0
            assert len(result.evaluations) > 0
            
        finally:
            if os.path.exists(sub_file_path):
                os.unlink(sub_file_path)
            if os.path.exists(ideal_file_path):
                os.unlink(ideal_file_path)
    
    @pytest.mark.asyncio
    async def test_evaluate_text_no_qa_pairs(self):
        """Test text evaluation with no Q&A pairs found."""
        self.setUp()
        
        with patch('app.repositories.text_rag_repository.TextRAGRepository.process_qa_document') as mock_process:
            mock_process.return_value = {}  # No Q&A pairs found
            
            sub_file_path = tempfile.mktemp(suffix='.docx')
            ideal_file_path = tempfile.mktemp(suffix='.docx')
            
            try:
                with open(sub_file_path, 'wb') as sub_file:
                    sub_file.write(b'test content')
                with open(ideal_file_path, 'wb') as ideal_file:
                    ideal_file.write(b'test content')
                
                result = await self.service.evaluate_text(
                    submission_path=sub_file_path,
                    ideal_path=ideal_file_path,
                    model=SchemaEmbeddingModel.OLLAMA
                )
                
                assert result.status == "error"
                assert "No Q&A pairs found" in result.message
                
            finally:
                if os.path.exists(sub_file_path):
                    os.unlink(sub_file_path)
                if os.path.exists(ideal_file_path):
                    os.unlink(ideal_file_path)


class TestTokenEstimationService:
    """Test class for token estimation service."""
    
    def setUp(self):
        self.service = TokenEstimationService()
    
    @patch('app.utils.tokenizer_utils.count_tokens')
    @patch('app.utils.docx_processor.DocxProcessor.extract_text_from_docx')
    @pytest.mark.asyncio
    async def test_estimate_text_tokens(self, mock_extract_text, mock_count_tokens):
        """Test token estimation for text files."""
        self.setUp()
        
        # Mock text extraction
        mock_extract_text.side_effect = [
            "This is student text content.",
            "This is ideal text content."
        ]
        
        # Mock token counting
        mock_count_tokens.side_effect = [100, 150]  # Student: 100, Ideal: 150
        
        sub_file_path = tempfile.mktemp(suffix='.docx')
        ideal_file_path = tempfile.mktemp(suffix='.docx')
        
        try:
            with open(sub_file_path, 'wb') as sub_file:
                sub_file.write(b'test content')
            with open(ideal_file_path, 'wb') as ideal_file:
                ideal_file.write(b'test content')
            
            result = await self.service.estimate_tokens(
                submission_path=sub_file_path,
                ideal_path=ideal_file_path,
                model=SchemaEmbeddingModel.OLLAMA
            )
            
            assert result.status == "success"
            assert result.estimated_tokens > 250  # Should include overhead
            assert "text" in result.message
            
        finally:
            if os.path.exists(sub_file_path):
                os.unlink(sub_file_path)
            if os.path.exists(ideal_file_path):
                os.unlink(ideal_file_path)
    
    @patch('app.utils.tokenizer_utils.count_tokens')
    @patch('app.utils.code_analyzer.extract_functions_from_zip')
    @pytest.mark.asyncio
    async def test_estimate_code_tokens(self, mock_extract_functions, mock_count_tokens):
        """Test token estimation for code files."""
        self.setUp()
        
        # Mock function extraction
        mock_extract_functions.side_effect = [
            {"func1": "def func1(): pass", "func2": "def func2(): return 1"},  # Student
            {"func1": "def func1(): pass"}  # Ideal
        ]
        
        # Mock token counting
        mock_count_tokens.side_effect = [20, 25, 20]  # func1: 20, func2: 25, ideal func1: 20
        
        # Create proper ZIP files for testing
        import zipfile
        
        sub_file_path = tempfile.mktemp(suffix='.zip')
        ideal_file_path = tempfile.mktemp(suffix='.zip')
        
        try:
            # Create test ZIP files
            with zipfile.ZipFile(sub_file_path, 'w') as zipf:
                zipf.writestr('test.py', 'def func1(): pass\ndef func2(): return 1')
            with zipfile.ZipFile(ideal_file_path, 'w') as zipf:
                zipf.writestr('test.py', 'def func1(): pass')
            
            result = await self.service.estimate_tokens(
                submission_path=sub_file_path,
                ideal_path=ideal_file_path,
                model=SchemaEmbeddingModel.OLLAMA
            )
            
            assert result.status == "success"
            assert result.estimated_tokens > 65  # Should include overhead
            assert "code" in result.message
            
        finally:
            if os.path.exists(sub_file_path):
                os.unlink(sub_file_path)
            if os.path.exists(ideal_file_path):
                os.unlink(ideal_file_path)
    
    @pytest.mark.asyncio
    async def test_estimate_tokens_empty_files(self):
        """Test token estimation with empty files."""
        self.setUp()
        
        with patch('app.utils.docx_processor.DocxProcessor.extract_text_from_docx') as mock_extract:
            mock_extract.return_value = ""  # Empty text
            
            sub_file_path = tempfile.mktemp(suffix='.docx')
            ideal_file_path = tempfile.mktemp(suffix='.docx')
            
            try:
                with open(sub_file_path, 'wb') as sub_file:
                    sub_file.write(b'test content')
                with open(ideal_file_path, 'wb') as ideal_file:
                    ideal_file.write(b'test content')
                
                result = await self.service.estimate_tokens(
                    submission_path=sub_file_path,
                    ideal_path=ideal_file_path,
                    model=SchemaEmbeddingModel.OLLAMA
                )
                
                # Should still succeed but with warnings
                assert result.status == "success"
                assert len(result.warnings) > 0
                
            finally:
                if os.path.exists(sub_file_path):
                    os.unlink(sub_file_path)
                if os.path.exists(ideal_file_path):
                    os.unlink(ideal_file_path)
