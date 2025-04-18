# AI Assignment Checker Architecture

## System Architecture Diagram

```
┌─────────────────┐     ┌────────────────────┐    ┌─────────────────────┐
│                 │     │                    │    │                     │
│  Web Interface  │────▶│   FastAPI Server   │───▶│   File Processing   │
│  (Upload Form)  │     │                    │    │                     │
│                 │     │                    │    └─────────┬───────────┘
└─────────────────┘     └────────────────────┘              │
                                                           ▼
┌─────────────────┐     ┌────────────────────┐    ┌─────────────────────┐
│                 │     │                    │    │                     │
│ Results Display │◀────│  Scoring Engine    │◀───│   Code Analysis     │
│                 │     │                    │    │                     │
└─────────────────┘     └────────────────────┘    └─────────┬───────────┘
                                 ▲                          │
                                 │                          ▼
┌─────────────────┐     ┌────────┴───────────┐    ┌─────────────────────┐
│                 │     │                    │    │                     │
│  MongoDB Atlas  │◀───▶│   RAG Processor    │◀───│  Embedding Generator │
│  Vector Store   │     │                    │    │                     │
│                 │     │                    │    └─────────┬───────────┘
└─────────────────┘     └────────────────────┘              │
                                 │                          │
                                 ▼                          ▼
                        ┌────────────────────┐    ┌─────────────────────┐
                        │                    │    │                     │
                        │   LLM Feedback     │    │   LLM Models        │
                        │   Generation       │    │   (Ollama/OpenAI)   │
                        │                    │    │                     │
                        └────────────────────┘    └─────────────────────┘
```

## RAG & LLM Implementation Document

### Project Overview

The AI Assignment Checker is an automated system that evaluates student code submissions against ideal solutions using artificial intelligence. The system leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to provide detailed, context-aware feedback and grading.

### Core Components

#### 1. File Processing System
- Accepts ZIP file uploads containing Python code submissions
- Extracts function definitions using Python's AST (Abstract Syntax Tree) module
- Maps student functions to ideal functions by name

#### 2. Embedding Generation
- **Purpose**: Converts code into vector representations for semantic comparison
- **Models**:
  - **Primary**: Ollama (Llama 3.2 3B) - runs locally
  - **Alternative**: OpenAI embeddings API - higher quality, requires API key
- **Process**: Code is vectorized to capture semantic meaning, not just syntax

#### 3. Vector Database (MongoDB)
- Stores code embeddings for retrieval
- Enables semantic search for similar code implementations
- Each document contains function name, code, embedding vector, and metadata

#### 4. RAG Processor
The RAG (Retrieval-Augmented Generation) system is the core of the evaluation process:

##### a. Retrieval Component
- **Function**: `retrieve_similar_contexts()`
- **Process**: 
  1. Calculates cosine similarity between student code embedding and stored embeddings
  2. Retrieves most similar code examples from the database
  3. Provides contextual examples for feedback generation

##### b. Context Preparation
- **Function**: `_analyze_code_structure()`
- **Process**:
  1. Performs detailed structure analysis between student and ideal code
  2. Identifies differences in variables, control structures, and function calls
  3. Prepares structured context for the LLM

##### c. Augmentation
- **Process**:
  1. Combines code structure analysis with retrieved similar examples
  2. Creates a comprehensive prompt using `FEEDBACK_PROMPT_TEMPLATE`
  3. Ensures token management for effective LLM processing

##### d. Generation
- **Function**: `generate_feedback()`
- **Process**:
  1. Selects between OpenAI API or Ollama local model
  2. Sends augmented prompt to selected LLM
  3. Receives detailed feedback and recommendations

#### 5. Scoring Engine
- Calculates similarity scores between student and ideal implementations
- Determines correctness based on similarity thresholds
- Combines direct similarity with structure analysis

#### 6. LLM Feedback Generation
- **Models**:
  - **Ollama**: Local LLM using Llama 3.2 3B
  - **OpenAI**: GPT-4o (with fallback to GPT-3.5-turbo)
- **Feedback Generated**:
  - Code quality assessment
  - Identification of missing concepts or patterns
  - Suggested improvements
  - Comparison with ideal implementation

### Data Flow in the RAG Process

```
1. Code → Embedding Generation → Vector Representation
2. Vector → Similarity Search → Retrieved Similar Examples
3. Code + Structure Analysis + Similar Examples → LLM → Detailed Feedback
4. All Components → Scoring Engine → Final Evaluation & Grade
```

### Key Features of the RAG Implementation

1. **Semantic Understanding**: The system understands code semantics, not just syntax matches, enabling evaluation of different implementations with the same logical approach.

2. **Contextual Feedback**: By retrieving similar examples, the LLM can reference alternative approaches when generating feedback.

3. **Detailed Structure Analysis**: The AST-based analysis provides granular insights into implementation differences.

4. **Adaptive Token Management**: The system dynamically adjusts context based on available token limits, prioritizing the most relevant information.

5. **Model Fallback Mechanism**: If the primary feedback model (OpenAI) fails, the system automatically falls back to the secondary model (Ollama).

### Sample Evaluation Flow

1. Student submits solution for a `calculate_median()` function
2. System generates embeddings for the submission
3. RAG retrieves similar implementations from the database
4. Structure analysis identifies missing edge case handling for even-length lists
5. Combined context is sent to the LLM
6. LLM generates specific feedback about the median calculation algorithm
7. System calculates a similarity score of 0.7921 (79.21%)
8. Final evaluation shows the function as "Correct" but with specific improvement recommendations

### Technical Implementation

The core RAG workflow is implemented in the `generate_comparison_report()` method, which:

1. Retrieves similar code contexts using vector similarity
2. Calculates direct similarity between student and ideal code
3. Performs detailed code structure analysis
4. Generates LLM-based feedback using the retrieved contexts
5. Combines all components into a comprehensive evaluation report

This implementation demonstrates how RAG can enhance traditional code similarity scoring with context-aware feedback, providing students with meaningful insights beyond simple correct/incorrect binaries.
