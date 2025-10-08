"""
Integration tests for MongoDB operations, data persistence, and retrieval.
Tests the complete MongoDB pipeline including connection, CRUD operations, and data validation.
"""

import pytest
import asyncio
import tempfile
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from datetime import datetime

# Add the backend directory to Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.repositories.db import Database
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.text_rag_repository import TextRAGRepository
from app.schemas.evaluate import EmbeddingModel


class TestMongoDBIntegration:
    """Integration tests for MongoDB operations and data persistence."""
    
    @pytest.fixture
    async def db_connection(self):
        """Create database connection for testing."""
        db = Database()
        await db.connect()
        yield db
        await db.disconnect()
    
    @pytest.fixture
    def embedding_repo(self, db_connection):
        """Create embedding repository instance."""
        return EmbeddingRepository()
    
    @pytest.fixture
    def text_rag_repo(self, db_connection):
        """Create text RAG repository instance."""
        return TextRAGRepository()
    
    @pytest.fixture
    def sample_embedding_data(self):
        """Sample embedding data for testing."""
        return {
            "text": "This is a test text for embedding storage",
            "embedding": np.random.rand(1536).astype(np.float32),
            "model": EmbeddingModel.OPENAI,
            "metadata": {
                "source": "test_integration",
                "timestamp": datetime.now().isoformat(),
                "user_id": "test_user_123"
            }
        }
    
    @pytest.fixture
    def sample_qa_data(self):
        """Sample Q&A data for testing."""
        return {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence.",
            "question_embedding": np.random.rand(1536).astype(np.float32),
            "answer_embedding": np.random.rand(1536).astype(np.float32),
            "metadata": {
                "document_id": "test_doc_123",
                "page_number": 1,
                "section": "introduction"
            }
        }
    
    @pytest.mark.asyncio
    async def test_embedding_storage_and_retrieval(self, embedding_repo, sample_embedding_data):
        """Test complete embedding storage and retrieval from MongoDB."""
        # Store embedding
        embedding_id = await embedding_repo.store_embedding(
            text=sample_embedding_data["text"],
            embedding=sample_embedding_data["embedding"],
            model=sample_embedding_data["model"],
            metadata=sample_embedding_data["metadata"]
        )
        
        # Validate storage
        assert embedding_id is not None
        assert isinstance(embedding_id, str)
        assert len(embedding_id) > 0
        
        # Retrieve embedding
        retrieved_data = await embedding_repo.get_embedding(embedding_id)
        
        # Validate retrieval
        assert retrieved_data is not None
        assert retrieved_data["text"] == sample_embedding_data["text"]
        assert np.array_equal(retrieved_data["embedding"], sample_embedding_data["embedding"])
        assert retrieved_data["model"] == sample_embedding_data["model"]
        assert retrieved_data["metadata"]["source"] == "test_integration"
        assert retrieved_data["metadata"]["user_id"] == "test_user_123"
    
    @pytest.mark.asyncio
    async def test_embedding_batch_operations(self, embedding_repo):
        """Test batch embedding storage and retrieval operations."""
        # Prepare batch data
        batch_data = []
        for i in range(5):
            batch_data.append({
                "text": f"Test text {i}",
                "embedding": np.random.rand(1536).astype(np.float32),
                "model": EmbeddingModel.OPENAI,
                "metadata": {"batch_id": f"batch_{i}"}
            })
        
        # Store batch embeddings
        embedding_ids = await embedding_repo.store_embeddings_batch(
            texts=[data["text"] for data in batch_data],
            embeddings=[data["embedding"] for data in batch_data],
            model=EmbeddingModel.OPENAI,
            metadata_list=[data["metadata"] for data in batch_data]
        )
        
        # Validate batch storage
        assert len(embedding_ids) == 5
        assert all(isinstance(id, str) and len(id) > 0 for id in embedding_ids)
        
        # Retrieve batch embeddings
        retrieved_data = await embedding_repo.get_embeddings_batch(embedding_ids)
        
        # Validate batch retrieval
        assert len(retrieved_data) == 5
        for i, data in enumerate(retrieved_data):
            assert data["text"] == f"Test text {i}"
            assert data["metadata"]["batch_id"] == f"batch_{i}"
            assert np.array_equal(data["embedding"], batch_data[i]["embedding"])
    
    @pytest.mark.asyncio
    async def test_embedding_search_and_filtering(self, embedding_repo):
        """Test embedding search and filtering capabilities."""
        # Store test embeddings with different metadata
        test_embeddings = [
            {
                "text": "Python programming tutorial",
                "embedding": np.random.rand(1536).astype(np.float32),
                "metadata": {"category": "programming", "language": "python"}
            },
            {
                "text": "Machine learning algorithms",
                "embedding": np.random.rand(1536).astype(np.float32),
                "metadata": {"category": "ai", "language": "python"}
            },
            {
                "text": "Web development with JavaScript",
                "embedding": np.random.rand(1536).astype(np.float32),
                "metadata": {"category": "programming", "language": "javascript"}
            }
        ]
        
        # Store embeddings
        embedding_ids = []
        for data in test_embeddings:
            embedding_id = await embedding_repo.store_embedding(
                text=data["text"],
                embedding=data["embedding"],
                model=EmbeddingModel.OPENAI,
                metadata=data["metadata"]
            )
            embedding_ids.append(embedding_id)
        
        # Test filtering by category
        programming_embeddings = await embedding_repo.get_embeddings_by_filter(
            filter_dict={"metadata.category": "programming"}
        )
        
        assert len(programming_embeddings) == 2
        assert all(emb["metadata"]["category"] == "programming" for emb in programming_embeddings)
        
        # Test filtering by language
        python_embeddings = await embedding_repo.get_embeddings_by_filter(
            filter_dict={"metadata.language": "python"}
        )
        
        assert len(python_embeddings) == 2
        assert all(emb["metadata"]["language"] == "python" for emb in python_embeddings)
    
    @pytest.mark.asyncio
    async def test_qa_data_storage_and_retrieval(self, text_rag_repo, sample_qa_data):
        """Test Q&A data storage and retrieval from MongoDB."""
        # Store Q&A data
        qa_id = await text_rag_repo.store_qa_pair(
            question=sample_qa_data["question"],
            answer=sample_qa_data["answer"],
            question_embedding=sample_qa_data["question_embedding"],
            answer_embedding=sample_qa_data["answer_embedding"],
            metadata=sample_qa_data["metadata"]
        )
        
        # Validate storage
        assert qa_id is not None
        assert isinstance(qa_id, str)
        
        # Retrieve Q&A data
        retrieved_qa = await text_rag_repo.get_qa_pair(qa_id)
        
        # Validate retrieval
        assert retrieved_qa is not None
        assert retrieved_qa["question"] == sample_qa_data["question"]
        assert retrieved_qa["answer"] == sample_qa_data["answer"]
        assert np.array_equal(retrieved_qa["question_embedding"], sample_qa_data["question_embedding"])
        assert np.array_equal(retrieved_qa["answer_embedding"], sample_qa_data["answer_embedding"])
        assert retrieved_qa["metadata"]["document_id"] == "test_doc_123"
    
    @pytest.mark.asyncio
    async def test_qa_similarity_search(self, text_rag_repo):
        """Test Q&A similarity search functionality."""
        # Store multiple Q&A pairs
        qa_pairs = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of AI.",
                "question_embedding": np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 307).astype(np.float32),  # 1536 dim
                "answer_embedding": np.array([0.2, 0.3, 0.4, 0.5, 0.6] * 307).astype(np.float32),
                "metadata": {"topic": "ai"}
            },
            {
                "question": "What is deep learning?",
                "answer": "Deep learning uses neural networks.",
                "question_embedding": np.array([0.3, 0.4, 0.5, 0.6, 0.7] * 307).astype(np.float32),
                "answer_embedding": np.array([0.4, 0.5, 0.6, 0.7, 0.8] * 307).astype(np.float32),
                "metadata": {"topic": "ai"}
            },
            {
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "question_embedding": np.array([0.9, 0.8, 0.7, 0.6, 0.5] * 307).astype(np.float32),
                "answer_embedding": np.array([0.8, 0.7, 0.6, 0.5, 0.4] * 307).astype(np.float32),
                "metadata": {"topic": "programming"}
            }
        ]
        
        # Store Q&A pairs
        qa_ids = []
        for qa in qa_pairs:
            qa_id = await text_rag_repo.store_qa_pair(
                question=qa["question"],
                answer=qa["answer"],
                question_embedding=qa["question_embedding"],
                answer_embedding=qa["answer_embedding"],
                metadata=qa["metadata"]
            )
            qa_ids.append(qa_id)
        
        # Test similarity search
        query_embedding = np.array([0.2, 0.3, 0.4, 0.5, 0.6] * 307).astype(np.float32)  # Similar to first Q&A
        similar_qa = await text_rag_repo.find_similar_qa_pairs(
            query_embedding=query_embedding,
            limit=2,
            threshold=0.5
        )
        
        # Validate similarity search results
        assert len(similar_qa) <= 2
        assert all("similarity" in qa for qa in similar_qa)
        assert all(qa["similarity"] >= 0.5 for qa in similar_qa)
        
        # Results should be ordered by similarity (highest first)
        if len(similar_qa) > 1:
            assert similar_qa[0]["similarity"] >= similar_qa[1]["similarity"]
    
    @pytest.mark.asyncio
    async def test_database_connection_handling(self, db_connection):
        """Test database connection handling and error recovery."""
        # Test connection status
        assert db_connection.is_connected()
        
        # Test reconnection
        await db_connection.disconnect()
        assert not db_connection.is_connected()
        
        await db_connection.connect()
        assert db_connection.is_connected()
    
    @pytest.mark.asyncio
    async def test_data_consistency_and_integrity(self, embedding_repo):
        """Test data consistency and integrity in MongoDB operations."""
        # Store embedding with specific data
        original_text = "Test text for consistency"
        original_embedding = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 307).astype(np.float32)  # 1536 dim
        original_metadata = {"test": "consistency", "value": 123}
        
        embedding_id = await embedding_repo.store_embedding(
            text=original_text,
            embedding=original_embedding,
            model=EmbeddingModel.OPENAI,
            metadata=original_metadata
        )
        
        # Retrieve and validate data integrity
        retrieved_data = await embedding_repo.get_embedding(embedding_id)
        
        # Validate text integrity
        assert retrieved_data["text"] == original_text
        
        # Validate embedding integrity
        assert np.array_equal(retrieved_data["embedding"], original_embedding)
        assert retrieved_data["embedding"].dtype == np.float32
        assert retrieved_data["embedding"].shape == (1536,)
        
        # Validate metadata integrity
        assert retrieved_data["metadata"]["test"] == "consistency"
        assert retrieved_data["metadata"]["value"] == 123
        
        # Test update operation
        updated_metadata = {"test": "consistency", "value": 456, "updated": True}
        await embedding_repo.update_embedding_metadata(embedding_id, updated_metadata)
        
        # Validate update
        updated_data = await embedding_repo.get_embedding(embedding_id)
        assert updated_data["metadata"]["value"] == 456
        assert updated_data["metadata"]["updated"] is True
        assert updated_data["text"] == original_text  # Text should remain unchanged
    
    @pytest.mark.asyncio
    async def test_database_performance_and_scalability(self, embedding_repo):
        """Test database performance with larger datasets."""
        # Create larger batch of embeddings
        batch_size = 100
        batch_texts = [f"Performance test text {i}" for i in range(batch_size)]
        batch_embeddings = [np.random.rand(1536).astype(np.float32) for _ in range(batch_size)]
        batch_metadata = [{"batch": "performance", "index": i} for i in range(batch_size)]
        
        # Measure batch storage time
        import time
        start_time = time.time()
        
        embedding_ids = await embedding_repo.store_embeddings_batch(
            texts=batch_texts,
            embeddings=batch_embeddings,
            model=EmbeddingModel.OPENAI,
            metadata_list=batch_metadata
        )
        
        storage_time = time.time() - start_time
        
        # Validate storage performance
        assert len(embedding_ids) == batch_size
        assert storage_time < 10.0  # Should complete within 10 seconds
        
        # Measure batch retrieval time
        start_time = time.time()
        
        retrieved_data = await embedding_repo.get_embeddings_batch(embedding_ids)
        
        retrieval_time = time.time() - start_time
        
        # Validate retrieval performance
        assert len(retrieved_data) == batch_size
        assert retrieval_time < 5.0  # Should complete within 5 seconds
        
        # Validate data integrity in batch operations
        for i, data in enumerate(retrieved_data):
            assert data["text"] == f"Performance test text {i}"
            assert data["metadata"]["index"] == i
            assert np.array_equal(data["embedding"], batch_embeddings[i])
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, embedding_repo):
        """Test database error handling and recovery."""
        # Test invalid embedding ID
        with pytest.raises(Exception):
            await embedding_repo.get_embedding("invalid_id")
        
        # Test invalid data types
        with pytest.raises(Exception):
            await embedding_repo.store_embedding(
                text="test",
                embedding="invalid_embedding",  # Should be numpy array
                model=EmbeddingModel.OPENAI
            )
        
        # Test empty text
        with pytest.raises(Exception):
            await embedding_repo.store_embedding(
                text="",  # Empty text should be rejected
                embedding=np.random.rand(1536).astype(np.float32),
                model=EmbeddingModel.OPENAI
            )
    
    @pytest.mark.asyncio
    async def test_database_cleanup_and_maintenance(self, embedding_repo):
        """Test database cleanup and maintenance operations."""
        # Store test embeddings
        test_ids = []
        for i in range(5):
            embedding_id = await embedding_repo.store_embedding(
                text=f"Cleanup test {i}",
                embedding=np.random.rand(1536).astype(np.float32),
                model=EmbeddingModel.OPENAI,
                metadata={"cleanup": "test"}
            )
            test_ids.append(embedding_id)
        
        # Test cleanup by metadata filter
        deleted_count = await embedding_repo.delete_embeddings_by_filter(
            filter_dict={"metadata.cleanup": "test"}
        )
        
        assert deleted_count == 5
        
        # Verify deletion
        for embedding_id in test_ids:
            with pytest.raises(Exception):
                await embedding_repo.get_embedding(embedding_id)
        
        # Test database statistics
        stats = await embedding_repo.get_database_stats()
        assert "total_embeddings" in stats
        assert "total_size_mb" in stats
        assert isinstance(stats["total_embeddings"], int)
        assert isinstance(stats["total_size_mb"], (int, float))
