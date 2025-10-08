# AI Assignment Evaluator - Test Suite

This directory contains comprehensive integration and end-to-end tests for the AI Assignment Evaluator project.

## Test Structure

### Integration Tests

#### `test_integration_embeddings.py`
- **Purpose**: Tests the complete embedding generation and storage pipeline
- **Coverage**:
  - OpenAI and Ollama embedding generation
  - Embedding format validation (size, type, normalization)
  - MongoDB storage and retrieval operations
  - Batch embedding processing
  - Error handling and recovery
  - Data consistency and integrity

#### `test_integration_llm_responses.py`
- **Purpose**: Tests LLM response generation and validation
- **Coverage**:
  - OpenAI API integration
  - Response format validation
  - Content quality assessment
  - Error handling and fallback mechanisms
  - Token usage tracking
  - Batch feedback generation

#### `test_integration_mongodb.py`
- **Purpose**: Tests MongoDB operations and data persistence
- **Coverage**:
  - Database connection handling
  - CRUD operations for embeddings and Q&A data
  - Batch operations and performance
  - Data consistency and integrity
  - Error handling and recovery
  - Database cleanup and maintenance

#### `test_integration_celery_tasks.py`
- **Purpose**: Tests Celery distributed task execution
- **Coverage**:
  - Parallel task execution
  - Task error handling and retry mechanisms
  - Result aggregation and formatting
  - Performance testing
  - Distributed processing workflows

### End-to-End Tests

#### `test_e2e_evaluation_workflow.py`
- **Purpose**: Tests complete evaluation workflows from file upload to results
- **Coverage**:
  - Complete code evaluation pipeline
  - Complete text evaluation pipeline
  - Error handling throughout the workflow
  - Performance measurements
  - Data persistence validation
  - Result export functionality

### Configuration

#### `conftest.py`
- **Purpose**: Shared fixtures and pytest configuration
- **Features**:
  - Database connection fixtures
  - Mock service fixtures
  - Sample data fixtures
  - Test markers and configuration

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

2. Set up test environment variables (optional for mocked tests):
```bash
export MONGODB_URI="mongodb://localhost:27017/test_db"
export OPENAI_API_KEY="your-test-api-key"
export REDIS_URL="redis://localhost:6379/0"
```

### Running Tests

#### Option 1: Using the Test Runner Script (Recommended)

```bash
# Navigate to backend directory
cd backend

# Run all tests
python run_tests.py all

# Run integration tests only
python run_tests.py integration

# Run end-to-end tests only
python run_tests.py e2e

# Run Celery tests only
python run_tests.py celery

# Run quick tests (excluding slow tests)
python run_tests.py quick

# Run with coverage
python run_tests.py all --coverage

# Run specific test file
python run_tests.py all --file test_integration_embeddings.py
```

#### Option 2: Using pytest directly

```bash
# Navigate to backend directory
cd backend

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Running Specific Test Categories

```bash
# Run only integration tests
pytest backend/tests/ -m integration

# Run only end-to-end tests
pytest backend/tests/ -m e2e

# Run only Celery tests
pytest backend/tests/ -m celery

# Run only slow tests
pytest backend/tests/ -m slow
```

### Running Individual Test Files

```bash
# Run embedding integration tests
pytest backend/tests/test_integration_embeddings.py

# Run LLM response tests
pytest backend/tests/test_integration_llm_responses.py

# Run MongoDB tests
pytest backend/tests/test_integration_mongodb.py

# Run Celery task tests
pytest backend/tests/test_integration_celery_tasks.py

# Run end-to-end workflow tests
pytest backend/tests/test_e2e_evaluation_workflow.py
```

## Test Data and Fixtures

### Sample Data

The tests use various sample data fixtures:

- **Code Functions**: Sample Python functions for code evaluation
- **Q&A Pairs**: Sample question-answer pairs for text evaluation
- **Embeddings**: Mock embedding vectors for testing
- **API Responses**: Mock API responses for external services

### Mock Services

Tests use comprehensive mocking for:

- **OpenAI API**: Mock embedding and LLM responses
- **Ollama API**: Mock local LLM responses
- **MongoDB**: Mock database operations
- **Redis**: Mock Celery broker operations
- **File Operations**: Mock file upload and processing

## Test Coverage

### Embedding Operations
- ✅ Embedding generation (OpenAI, Ollama)
- ✅ Embedding format validation
- ✅ Embedding storage and retrieval
- ✅ Batch embedding operations
- ✅ Similarity computation
- ✅ Error handling

### LLM Operations
- ✅ Response generation
- ✅ Format validation
- ✅ Content quality assessment
- ✅ Token usage tracking
- ✅ Error handling and retry
- ✅ Batch processing

### Database Operations
- ✅ Connection management
- ✅ CRUD operations
- ✅ Batch operations
- ✅ Data consistency
- ✅ Performance testing
- ✅ Error recovery

### Celery Operations
- ✅ Task distribution
- ✅ Parallel execution
- ✅ Result aggregation
- ✅ Error handling
- ✅ Retry mechanisms
- ✅ Performance testing

### End-to-End Workflows
- ✅ Complete code evaluation
- ✅ Complete text evaluation
- ✅ Error handling
- ✅ Performance measurement
- ✅ Data persistence
- ✅ Result export

## Test Markers

Tests are marked with the following categories:

- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.celery`: Celery-specific tests
- `@pytest.mark.slow`: Performance and slow tests

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

1. **Unit Tests**: Fast, isolated tests
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test system performance

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure MongoDB is running and accessible
2. **API Keys**: Set valid test API keys in environment variables
3. **Redis Connection**: Ensure Redis is running for Celery tests
4. **File Permissions**: Ensure test files can be created and deleted

### Debug Mode

Run tests in debug mode for detailed output:

```bash
pytest backend/tests/ -v -s --tb=long
```

### Test Isolation

Each test is designed to be isolated and can run independently:

- Tests use temporary files and directories
- Database operations use test collections
- Mock services prevent external dependencies
- Cleanup is performed after each test

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use appropriate fixtures from `conftest.py`
3. Add proper test markers
4. Include comprehensive error handling tests
5. Document test purpose and coverage
6. Ensure tests are isolated and repeatable
