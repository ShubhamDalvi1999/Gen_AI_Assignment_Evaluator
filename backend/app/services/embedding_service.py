import numpy as np
import requests
from scipy.spatial.distance import cosine
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.config import settings
from ..core.logging import service_logger as logger
from ..core.rag_logging import (
    log_embedding_operation, 
    log_similarity_computation,
    log_openai_api_call
)


class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


@log_embedding_operation("ollama_embedding_generation")
def get_embedding_ollama(text: str) -> np.ndarray:
    """Generate embedding using Ollama's local model."""
    try:
        logger.debug(f"Generating Ollama embedding for text of length {len(text)}")
        response = requests.post(
            settings.ollama_api_url,
            json={"model": settings.ollama_embedding_model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        if "embedding" not in result:
            raise ValueError("No embedding in Ollama response")
        embedding = np.array(result["embedding"])
        logger.debug(f"Successfully generated Ollama embedding with dimension {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Ollama embedding generation failed: {e}")
        raise ValueError(f"Ollama embedding generation failed: {str(e)}")


@log_embedding_operation("openai_embedding_generation")
@log_openai_api_call("openai_embedding_api")
def get_embedding_openai(text: str) -> np.ndarray:
    """Generate embedding using OpenAI's API."""
    if not settings.openai_api_key:
        logger.error("OpenAI API key not configured")
        raise ValueError("OpenAI API key not configured")
    
    try:
        logger.debug(f"Generating OpenAI embedding for text of length {len(text)}")
        response = requests.post(
            settings.openai_api_url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-ada-002",
                "input": text
            },
            timeout=10
        )
        response.raise_for_status()
        embedding = np.array(response.json()["data"][0]["embedding"])
        logger.debug(f"Successfully generated OpenAI embedding with dimension {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"OpenAI embedding generation failed: {e}")
        raise ValueError(f"OpenAI embedding generation failed: {str(e)}")


@log_embedding_operation("embedding_generation")
def get_embedding(text: str, model: EmbeddingModel = EmbeddingModel.OLLAMA) -> np.ndarray:
    """Generate embedding based on selected model."""
    logger.debug(f"Generating embedding using model: {model}")
    start_time = datetime.now()
    
    logger.info("========== EMBEDDING GENERATION STAGE ==========")
    logger.info(f"Generating embedding for text of length {len(text)} using model: {model}")
    
    try:
        if model == EmbeddingModel.OLLAMA:
            embedding = get_embedding_ollama(text)
        else:
            embedding = get_embedding_openai(text)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Embedding generation successful. Vector dimension: {len(embedding)}, Time: {elapsed:.2f}s")
        logger.debug(f"Generated embedding successfully in {elapsed:.2f}s ({len(text)} chars, model: {model})")
        return embedding
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Embedding generation failed after {elapsed:.2f}s: {e}")
        raise


@log_similarity_computation("cosine_similarity")
def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    try:
        similarity = float(1 - cosine(emb1, emb2))
        logger.debug(f"Computed similarity: {similarity:.4f}")
        return similarity
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0
