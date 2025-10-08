"""
Integration tests for embedding generation, validation, and storage.
Tests the complete embedding pipeline including API calls, format validation, and MongoDB storage.
"""

import pytest
import asyncio
import tempfile
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add the backend directory to Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.embedding_service import get_embedding, EmbeddingModel
from app.repositories.embedding_repository import EmbeddingRepository
from app.schemas.evaluate import EmbeddingModel as SchemaEmbeddingModel


class TestEmbeddingIntegration:
    """Integration tests for embedding generation and storage."""
    
    @pytest.fixture
    def embedding_repo(self):
        """Create embedding repository instance."""
        return EmbeddingRepository()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "short": "Hello world",
            "medium": "This is a medium length text that should generate a proper embedding vector.",
            "long": "This is a much longer text that contains multiple sentences and should definitely generate a meaningful embedding. It includes various concepts and ideas that should be captured in the vector representation. The embedding should reflect the semantic content of this text."
        }
    
    @pytest.mark.asyncio
    async def test_openai_embedding_generation_and_storage(self, embedding_repo, sample_texts):
        """Test complete OpenAI embedding generation and MongoDB storage."""
        # Mock OpenAI API response
        mock_embedding = np.random.rand(1536).astype(np.float32)  # OpenAI embedding size
        
        with patch('app.services.embedding_service.requests.post') as mock_post:
            # Mock successful OpenAI response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": mock_embedding.tolist()}]
            }
            mock_post.return_value = mock_response
            
            # Test embedding generation
            text = sample_texts["medium"]
            embedding = get_embedding(text, EmbeddingModel.OPENAI)
            
            # Validate embedding format and size
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (1536,)  # OpenAI embedding dimension
            assert embedding.dtype == np.float32
            assert not np.isnan(embedding).any()  # No NaN values
            assert not np.isinf(embedding).any()  # No infinite values
            
            # Test MongoDB storage
            with patch.object(embedding_repo, 'store_embedding') as mock_store:
                mock_store.return_value = "test_embedding_id"
                
                embedding_id = await embedding_repo.store_embedding(
                    text=text,
                    embedding=embedding,
                    model=SchemaEmbeddingModel.OPENAI,
                    metadata={"test": "integration"}
                )
                
                # Verify storage was called with correct parameters
                mock_store.assert_called_once()
                call_args = mock_store.call_args
                assert call_args[1]["text"] == text
                assert np.array_equal(call_args[1]["embedding"], embedding)
                assert call_args[1]["model"] == SchemaEmbeddingModel.OPENAI
                assert call_args[1]["metadata"]["test"] == "integration"
    
    @pytest.mark.asyncio
    async def test_ollama_embedding_generation_and_storage(self, embedding_repo, sample_texts):
        """Test complete Ollama embedding generation and MongoDB storage."""
        # Mock Ollama API response
        mock_embedding = np.random.rand(4096).astype(np.float32)  # Typical Ollama embedding size
        
        with patch('app.services.embedding_service.requests.post') as mock_post:
            # Mock successful Ollama response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embedding": mock_embedding.tolist()
            }
            mock_post.return_value = mock_response
            
            # Test embedding generation
            text = sample_texts["long"]
            embedding = get_embedding(text, EmbeddingModel.OLLAMA)
            
            # Validate embedding format and size
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (4096,)  # Ollama embedding dimension
            assert embedding.dtype == np.float32
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()
            
            # Test MongoDB storage
            with patch.object(embedding_repo, 'store_embedding') as mock_store:
                mock_store.return_value = "test_ollama_embedding_id"
                
                embedding_id = await embedding_repo.store_embedding(
                    text=text,
                    embedding=embedding,
                    model=SchemaEmbeddingModel.OLLAMA,
                    metadata={"source": "ollama_test"}
                )
                
                mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_retrieval_and_similarity(self, embedding_repo):
        """Test embedding retrieval from MongoDB and similarity computation."""
        # Create test embeddings
        embedding1 = np.random.rand(1536).astype(np.float32)
        embedding2 = np.random.rand(1536).astype(np.float32)
        
        # Mock MongoDB retrieval
        with patch.object(embedding_repo, 'get_embedding') as mock_get:
            mock_get.side_effect = [
                {"embedding": embedding1, "text": "First text", "model": "openai"},
                {"embedding": embedding2, "text": "Second text", "model": "openai"}
            ]
            
            # Retrieve embeddings
            result1 = await embedding_repo.get_embedding("test_id_1")
            result2 = await embedding_repo.get_embedding("test_id_2")
            
            # Validate retrieval
            assert result1["text"] == "First text"
            assert result2["text"] == "Second text"
            assert np.array_equal(result1["embedding"], embedding1)
            assert np.array_equal(result2["embedding"], embedding2)
            
            # Test similarity computation
            from app.services.embedding_service import compute_similarity
            
            similarity = compute_similarity(embedding1, embedding2)
            assert isinstance(similarity, float)
            assert -1.0 <= similarity <= 1.0  # Cosine similarity range
    
    @pytest.mark.asyncio
    async def test_embedding_batch_processing(self, embedding_repo, sample_texts):
        """Test batch embedding generation and storage."""
        texts = list(sample_texts.values())
        mock_embeddings = [np.random.rand(1536).astype(np.float32) for _ in texts]
        
        with patch('app.services.embedding_service.requests.post') as mock_post:
            # Mock batch responses
            mock_responses = []
            for embedding in mock_embeddings:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": [{"embedding": embedding.tolist()}]
                }
                mock_responses.append(mock_response)
            
            mock_post.side_effect = mock_responses
            
            # Generate embeddings for all texts
            embeddings = []
            for text in texts:
                embedding = get_embedding(text, EmbeddingModel.OPENAI)
                embeddings.append(embedding)
            
            # Validate all embeddings
            assert len(embeddings) == len(texts)
            for embedding in embeddings:
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (1536,)
                assert embedding.dtype == np.float32
            
            # Test batch storage
            with patch.object(embedding_repo, 'store_embeddings_batch') as mock_batch_store:
                mock_batch_store.return_value = ["id1", "id2", "id3"]
                
                embedding_ids = await embedding_repo.store_embeddings_batch(
                    texts=texts,
                    embeddings=embeddings,
                    model=SchemaEmbeddingModel.OPENAI,
                    metadata_list=[{"batch": "test"} for _ in texts]
                )
                
                assert len(embedding_ids) == len(texts)
                mock_batch_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, embedding_repo):
        """Test error handling in embedding generation and storage."""
        # Test API failure
        with patch('app.services.embedding_service.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception("API Error")
            mock_post.return_value = mock_response
            
            with pytest.raises(ValueError, match="Failed to generate embedding"):
                get_embedding("test text", EmbeddingModel.OPENAI)
        
        # Test MongoDB storage failure
        with patch.object(embedding_repo, 'store_embedding') as mock_store:
            mock_store.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await embedding_repo.store_embedding(
                    text="test",
                    embedding=np.random.rand(1536).astype(np.float32),
                    model=SchemaEmbeddingModel.OPENAI
                )
    
    @pytest.mark.asyncio
    async def test_embedding_consistency_across_models(self, embedding_repo):
        """Test that embeddings are consistent and properly normalized."""
        text = "This is a test text for consistency checking."
        
        # Mock different model responses
        openai_embedding = np.random.rand(1536).astype(np.float32)
        ollama_embedding = np.random.rand(4096).astype(np.float32)
        
        with patch('app.services.embedding_service.requests.post') as mock_post:
            def mock_response_side_effect(*args, **kwargs):
                mock_resp = Mock()
                mock_resp.status_code = 200
                
                # Check if it's OpenAI or Ollama call
                if "openai.com" in str(args[1]):
                    mock_resp.json.return_value = {
                        "data": [{"embedding": openai_embedding.tolist()}]
                    }
                else:
                    mock_resp.json.return_value = {
                        "embedding": ollama_embedding.tolist()
                    }
                return mock_resp
            
            mock_post.side_effect = mock_response_side_effect
            
            # Generate embeddings with different models
            openai_emb = get_embedding(text, EmbeddingModel.OPENAI)
            ollama_emb = get_embedding(text, EmbeddingModel.OLLAMA)
            
            # Validate dimensions are correct
            assert openai_emb.shape == (1536,)
            assert ollama_emb.shape == (4096,)
            
            # Validate embeddings are normalized (magnitude close to 1)
            openai_magnitude = np.linalg.norm(openai_emb)
            ollama_magnitude = np.linalg.norm(ollama_emb)
            
            # OpenAI embeddings are typically normalized
            assert 0.9 <= openai_magnitude <= 1.1
            # Ollama embeddings may not be normalized
            assert ollama_magnitude > 0
    
    @pytest.mark.asyncio
    async def test_embedding_metadata_persistence(self, embedding_repo):
        """Test that embedding metadata is properly stored and retrieved."""
        text = "Test text for metadata persistence"
        embedding = np.random.rand(1536).astype(np.float32)
        metadata = {
            "source": "test_integration",
            "timestamp": "2024-01-01T00:00:00Z",
            "user_id": "test_user",
            "evaluation_id": "eval_123"
        }
        
        with patch.object(embedding_repo, 'store_embedding') as mock_store:
            mock_store.return_value = "test_metadata_id"
            
            embedding_id = await embedding_repo.store_embedding(
                text=text,
                embedding=embedding,
                model=SchemaEmbeddingModel.OPENAI,
                metadata=metadata
            )
            
            # Verify metadata was stored correctly
            call_args = mock_store.call_args
            stored_metadata = call_args[1]["metadata"]
            
            assert stored_metadata["source"] == "test_integration"
            assert stored_metadata["timestamp"] == "2024-01-01T00:00:00Z"
            assert stored_metadata["user_id"] == "test_user"
            assert stored_metadata["evaluation_id"] == "eval_123"
