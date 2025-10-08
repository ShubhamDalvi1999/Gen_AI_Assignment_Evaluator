"""
Pytest configuration and shared fixtures for integration and end-to-end tests.
"""

import pytest
import asyncio
import tempfile
import os
import sys
import numpy as np
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

# Add the backend directory to Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.repositories.db import Database
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.text_rag_repository import TextRAGRepository
from app.schemas.evaluate import EmbeddingModel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_database():
    """Mock database connection for testing."""
    db = Mock(spec=Database)
    db.is_connected.return_value = True
    db.connect = AsyncMock()
    db.disconnect = AsyncMock()
    yield db


@pytest.fixture
def mock_embedding_repository():
    """Mock embedding repository for testing."""
    repo = Mock(spec=EmbeddingRepository)
    repo.store_embedding = AsyncMock(return_value="test_embedding_id")
    repo.get_embedding = AsyncMock(return_value={
        "text": "test text",
        "embedding": np.random.rand(1536).astype(np.float32),
        "model": EmbeddingModel.OPENAI,
        "metadata": {"test": "data"}
    })
    repo.store_embeddings_batch = AsyncMock(return_value=["id1", "id2", "id3"])
    repo.get_embeddings_batch = AsyncMock(return_value=[
        {"text": "test1", "embedding": np.random.rand(1536).astype(np.float32)},
        {"text": "test2", "embedding": np.random.rand(1536).astype(np.float32)},
        {"text": "test3", "embedding": np.random.rand(1536).astype(np.float32)}
    ])
    repo.get_embeddings_by_filter = AsyncMock(return_value=[])
    repo.delete_embeddings_by_filter = AsyncMock(return_value=0)
    repo.get_database_stats = AsyncMock(return_value={
        "total_embeddings": 100,
        "total_size_mb": 50.5
    })
    yield repo


@pytest.fixture
def mock_text_rag_repository():
    """Mock text RAG repository for testing."""
    repo = Mock(spec=TextRAGRepository)
    repo.store_qa_pair = AsyncMock(return_value="test_qa_id")
    repo.get_qa_pair = AsyncMock(return_value={
        "question": "test question",
        "answer": "test answer",
        "question_embedding": np.random.rand(1536).astype(np.float32),
        "answer_embedding": np.random.rand(1536).astype(np.float32),
        "metadata": {"test": "data"}
    })
    repo.find_similar_qa_pairs = AsyncMock(return_value=[
        {
            "question": "similar question",
            "answer": "similar answer",
            "similarity": 0.85,
            "metadata": {"test": "data"}
        }
    ])
    repo.process_qa_document = AsyncMock(return_value={
        "qa_1": {
            "question": "What is AI?",
            "answer": "Artificial Intelligence",
            "embedding": np.random.rand(1536).astype(np.float32),
            "question_embedding": np.random.rand(1536).astype(np.float32)
        }
    })
    yield repo


@pytest.fixture
def sample_embedding_data():
    """Sample embedding data for testing."""
    return {
        "text": "This is a test text for embedding operations",
        "embedding": np.random.rand(1536).astype(np.float32),
        "model": EmbeddingModel.OPENAI,
        "metadata": {
            "source": "test_fixture",
            "timestamp": "2024-01-01T00:00:00Z",
            "user_id": "test_user"
        }
    }


@pytest.fixture
def sample_qa_data():
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


@pytest.fixture
def sample_code_functions():
    """Sample code functions for testing."""
    return {
        "calculate_mean": "def calculate_mean(numbers):\n    return sum(numbers) / len(numbers)",
        "calculate_median": "def calculate_median(numbers):\n    sorted_numbers = sorted(numbers)\n    n = len(sorted_numbers)\n    if n % 2 == 0:\n        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2\n    else:\n        return sorted_numbers[n//2]",
        "calculate_variance": "def calculate_variance(numbers):\n    mean = calculate_mean(numbers)\n    return sum((x - mean) ** 2 for x in numbers) / len(numbers)"
    }


@pytest.fixture
def sample_qa_pairs():
    """Sample Q&A pairs for testing."""
    return {
        "qa_1": {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "embedding": np.random.rand(1536).astype(np.float32),
            "question_embedding": np.random.rand(1536).astype(np.float32)
        },
        "qa_2": {
            "question": "What is the difference between supervised and unsupervised learning?",
            "answer": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
            "embedding": np.random.rand(1536).astype(np.float32),
            "question_embedding": np.random.rand(1536).astype(np.float32)
        },
        "qa_3": {
            "question": "What is overfitting?",
            "answer": "Overfitting occurs when a model performs well on training data but poorly on new data.",
            "embedding": np.random.rand(1536).astype(np.float32),
            "question_embedding": np.random.rand(1536).astype(np.float32)
        }
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""
    return {
        "choices": [{
            "message": {
                "content": '{"feedback": "Good implementation!", "score": 8.5, "suggestions": ["Add validation"], "strengths": ["Correct logic"], "improvements": ["Error handling"]}'
            }
        }],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200
        }
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response for testing."""
    return {
        "embedding": np.random.rand(4096).astype(np.float32).tolist()
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_zip_file(temp_directory):
    """Create a sample ZIP file for testing."""
    import zipfile
    
    zip_path = os.path.join(temp_directory, "test_code.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.writestr("test.py", '''
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]
''')
    
    yield zip_path


@pytest.fixture
def sample_docx_file(temp_directory):
    """Create a sample DOCX file for testing."""
    docx_path = os.path.join(temp_directory, "test_text.docx")
    
    # For testing purposes, create a simple text file
    # In real implementation, this would be a proper DOCX file
    with open(docx_path, 'w', encoding='utf-8') as f:
        f.write('''
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence.

Q: What is deep learning?
A: Deep learning uses neural networks with multiple layers.
''')
    
    yield docx_path


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for API testing."""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": np.random.rand(1536).astype(np.float32).tolist()}]
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    with patch('app.services.embedding_service.get_embedding') as mock_get_embedding:
        mock_get_embedding.return_value = np.random.rand(1536).astype(np.float32)
        yield mock_get_embedding


@pytest.fixture
def mock_feedback_service():
    """Mock feedback service for testing."""
    mock_feedback = {
        "feedback": "Good implementation!",
        "score": 8.0,
        "suggestions": ["Add validation"],
        "strengths": ["Correct logic"],
        "improvements": ["Error handling"]
    }
    
    with patch('app.services.feedback_service.FeedbackService.generate_code_feedback') as mock_code_feedback, \
         patch('app.services.feedback_service.FeedbackService.generate_text_feedback') as mock_text_feedback:
        
        mock_code_feedback.return_value = mock_feedback
        mock_text_feedback.return_value = mock_feedback
        
        yield {
            "code_feedback": mock_code_feedback,
            "text_feedback": mock_text_feedback
        }


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    mock_response = {
        "content": '{"feedback": "Good implementation!", "score": 8.0}',
        "usage": {"total_tokens": 200},
        "model": "gpt-3.5-turbo"
    }
    
    with patch('app.services.llm_service.LLMService.generate_response') as mock_generate:
        mock_generate.return_value = mock_response
        yield mock_generate


@pytest.fixture
def mock_celery_tasks():
    """Mock Celery tasks for testing."""
    with patch('app.tasks.evaluation_tasks.evaluate_code_parallel_task') as mock_code_task, \
         patch('app.tasks.evaluation_tasks.evaluate_text_parallel_task') as mock_text_task, \
         patch('app.tasks.embedding_tasks.generate_embedding_task') as mock_embedding_task:
        
        mock_code_task.return_value = {
            "status": "success",
            "overall_score": 8.5,
            "evaluations": [
                {
                    "function_name": "test_function",
                    "similarity": 0.85,
                    "score": 8.5,
                    "feedback": {"feedback": "Good implementation!"}
                }
            ]
        }
        
        mock_text_task.return_value = {
            "status": "success",
            "overall_score": 7.8,
            "evaluations": [
                {
                    "question": "test question",
                    "answer_similarity": 0.78,
                    "score": 7.8,
                    "feedback": {"feedback": "Good understanding!"}
                }
            ]
        }
        
        mock_embedding_task.return_value = {
            "embedding_id": "test_embedding_id",
            "text": "test text",
            "model": EmbeddingModel.OPENAI
        }
        
        yield {
            "code_task": mock_code_task,
            "text_task": mock_text_task,
            "embedding_task": mock_embedding_task
        }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "celery: mark test as requiring Celery"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker to end-to-end tests
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        
        # Add celery marker to Celery tests
        if "celery" in item.nodeid:
            item.add_marker(pytest.mark.celery)
        
        # Add slow marker to performance tests
        if "performance" in item.nodeid or "workflow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
