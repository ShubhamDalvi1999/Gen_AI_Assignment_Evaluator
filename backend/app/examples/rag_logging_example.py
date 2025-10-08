"""
Example of how to use the RAG logging system in the AI Assignment Checker.

This file demonstrates how to use the various RAG logging decorators and session logging
for comprehensive tracking of RAG operations.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from ..core.rag_logging import (
    log_embedding_operation,
    log_llm_response,
    log_augmentation_operation,
    log_similarity_computation,
    log_retrieval_operation,
    log_openai_api_call,
    RAGSessionLogger
)
from ..services.embedding_service import get_embedding, EmbeddingModel
from ..services.feedback_service import generate_code_feedback


# Example 1: Using individual decorators
@log_embedding_operation("example_embedding")
def generate_example_embedding(text: str) -> str:
    """Example function showing embedding generation logging."""
    # This would normally call the actual embedding service
    return f"Generated embedding for: {text[:50]}..."


@log_llm_response("example_llm_response")
def generate_example_response(prompt: str) -> str:
    """Example function showing LLM response logging."""
    # This would normally call the actual LLM service
    return f"Generated response for prompt: {prompt[:50]}..."


@log_similarity_computation("example_similarity")
def compute_example_similarity(score: float) -> float:
    """Example function showing similarity computation logging."""
    return score * 0.95  # Simulate similarity computation


@log_retrieval_operation("example_retrieval")
def retrieve_example_contexts(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Example function showing context retrieval logging."""
    return [
        {"context": f"Context {i} for query: {query[:30]}...", "similarity": 0.9 - i*0.1}
        for i in range(top_k)
    ]


@log_augmentation_operation("example_augmentation")
def augment_example_context(contexts: List[Dict[str, Any]], query: str) -> str:
    """Example function showing context augmentation logging."""
    context_text = "\n".join([ctx["context"] for ctx in contexts])
    return f"Augmented context for query '{query[:30]}...':\n{context_text[:200]}..."


# Example 2: Using RAG session logging
async def example_rag_session():
    """Example of a complete RAG session with comprehensive logging."""
    
    with RAGSessionLogger("example_rag_session") as rag_session:
        # Log session start
        rag_session.log_operation("session_start", {
            "timestamp": datetime.now().isoformat(),
            "description": "Starting example RAG session"
        })
        
        # Step 1: Generate embeddings
        rag_session.log_operation("embedding_generation", {
            "text_length": 100,
            "model": "ollama"
        })
        embedding_result = generate_example_embedding("Sample text for embedding generation")
        
        # Step 2: Retrieve similar contexts
        rag_session.log_operation("context_retrieval", {
            "query": "sample query",
            "top_k": 5
        })
        contexts = retrieve_example_contexts("sample query", top_k=5)
        
        # Step 3: Compute similarities
        rag_session.log_operation("similarity_computation", {
            "contexts_count": len(contexts),
            "similarity_threshold": 0.8
        })
        similarities = [compute_example_similarity(ctx["similarity"]) for ctx in contexts]
        
        # Step 4: Augment context
        rag_session.log_operation("context_augmentation", {
            "input_contexts": len(contexts),
            "query": "sample query"
        })
        augmented_context = augment_example_context(contexts, "sample query")
        
        # Step 5: Generate LLM response
        rag_session.log_operation("llm_response_generation", {
            "prompt_length": len(augmented_context),
            "model": "gpt-3.5-turbo"
        })
        response = generate_example_response(augmented_context)
        
        # Log session completion
        rag_session.log_operation("session_completion", {
            "total_operations": 5,
            "final_response_length": len(response)
        })
        
        return {
            "embedding_result": embedding_result,
            "contexts": contexts,
            "similarities": similarities,
            "augmented_context": augmented_context,
            "final_response": response
        }


# Example 3: Real-world usage in evaluation service
async def example_code_evaluation_with_logging():
    """Example showing how to integrate RAG logging into actual evaluation services."""
    
    with RAGSessionLogger("code_evaluation") as rag_session:
        try:
            # Log the start of code evaluation
            rag_session.log_operation("evaluation_start", {
                "student_file": "student_code.py",
                "ideal_file": "ideal_code.py",
                "model": "openai"
            })
            
            # Simulate embedding generation for ideal code
            rag_session.log_operation("ideal_embedding_generation", {
                "functions_count": 5,
                "model": "openai"
            })
            # This would call: get_embedding(ideal_code, EmbeddingModel.OPENAI)
            
            # Simulate embedding generation for student code
            rag_session.log_operation("student_embedding_generation", {
                "functions_count": 4,
                "model": "openai"
            })
            # This would call: get_embedding(student_code, EmbeddingModel.OPENAI)
            
            # Simulate similarity computation
            rag_session.log_operation("similarity_computation", {
                "comparisons_count": 20,
                "average_similarity": 0.85
            })
            
            # Simulate context retrieval
            rag_session.log_operation("context_retrieval", {
                "query_embedding_dimension": 1536,
                "retrieved_contexts": 9
            })
            
            # Simulate feedback generation
            rag_session.log_operation("feedback_generation", {
                "use_openai": True,
                "prompt_length": 2000,
                "model": "gpt-3.5-turbo"
            })
            # This would call: generate_code_feedback(...)
            
            # Log successful completion
            rag_session.log_operation("evaluation_complete", {
                "total_functions_evaluated": 4,
                "average_similarity": 0.85,
                "feedback_generated": True
            })
            
            return {
                "status": "success",
                "functions_evaluated": 4,
                "average_similarity": 0.85,
                "feedback": "Generated feedback successfully"
            }
            
        except Exception as e:
            # Log error
            rag_session.log_operation("evaluation_error", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise


# Example 4: Text evaluation with RAG logging
async def example_text_evaluation_with_logging():
    """Example showing RAG logging for text Q&A evaluation."""
    
    with RAGSessionLogger("text_evaluation") as rag_session:
        try:
            # Log document processing
            rag_session.log_operation("document_processing", {
                "ideal_document": "ideal_qa.docx",
                "student_document": "student_qa.docx",
                "model": "openai"
            })
            
            # Simulate Q&A pair extraction and embedding generation
            rag_session.log_operation("qa_embedding_generation", {
                "ideal_qa_pairs": 10,
                "student_qa_pairs": 8,
                "model": "openai"
            })
            
            # Simulate Q&A matching
            rag_session.log_operation("qa_matching", {
                "matches_found": 7,
                "high_quality_matches": 3,
                "medium_quality_matches": 2,
                "low_quality_matches": 2
            })
            
            # Simulate feedback generation for each Q&A pair
            for i in range(7):
                rag_session.log_operation("qa_feedback_generation", {
                    "qa_pair_index": i,
                    "similarity_score": 0.9 - i*0.1,
                    "use_openai": True
                })
            
            # Simulate summary generation
            rag_session.log_operation("summary_generation", {
                "total_questions": 10,
                "answered_questions": 7,
                "overall_score": 75.5,
                "use_openai": True
            })
            
            return {
                "status": "success",
                "total_questions": 10,
                "answered_questions": 7,
                "overall_score": 75.5,
                "high_quality": 3,
                "medium_quality": 2,
                "low_quality": 2
            }
            
        except Exception as e:
            rag_session.log_operation("text_evaluation_error", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise


if __name__ == "__main__":
    # Run examples
    print("Running RAG logging examples...")
    
    # Example 1: Individual decorators
    print("\n1. Testing individual decorators:")
    generate_example_embedding("This is a sample text for embedding generation")
    generate_example_response("This is a sample prompt for LLM response generation")
    compute_example_similarity(0.85)
    retrieve_example_contexts("sample query", top_k=3)
    augment_example_context([{"context": "Sample context"}], "sample query")
    
    # Example 2: RAG session
    print("\n2. Testing RAG session logging:")
    asyncio.run(example_rag_session())
    
    # Example 3: Code evaluation
    print("\n3. Testing code evaluation with logging:")
    asyncio.run(example_code_evaluation_with_logging())
    
    # Example 4: Text evaluation
    print("\n4. Testing text evaluation with logging:")
    asyncio.run(example_text_evaluation_with_logging())
    
    print("\nAll examples completed! Check the logs directory for detailed RAG logs.")
