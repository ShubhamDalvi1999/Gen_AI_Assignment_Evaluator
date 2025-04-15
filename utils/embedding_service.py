import os
import sys
import numpy as np
import requests
from scipy.spatial.distance import cosine
from enum import Enum
from utils.logger import embedding_logger as logger
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Configure logging first before using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a custom logger with more detailed settings
logger = logging.getLogger("app")

# Set log level from environment variable if available, otherwise default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, log_level))
    logger.info(f"Log level set to {log_level}")
except AttributeError:
    logger.setLevel(logging.INFO)
    logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")

# Add a file handler to save logs to a file
try:
    log_file_path = os.path.join("logs", "app.log")
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - [%(funcName)s] - %(message)s'
    ))
    logger.addHandler(file_handler)
    logger.info(f"Log file created at: {log_file_path}")
except Exception as e:
    logger.warning(f"Failed to create log file: {e}")

# Log startup info
logger.info("=" * 60)
logger.info("AI Assignment Checker Starting")
logger.info("=" * 60)

# Load environment variables
load_dotenv()
class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

# API URLs and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))


def get_embedding_ollama(text: str) -> np.ndarray:
    """Generate embedding using Ollama's local model."""
    try:
        logger.debug(f"Generating Ollama embedding for text of length {len(text)}")
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "llama3.2:3b", "prompt": text},
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

def get_embedding_openai(text: str) -> np.ndarray:
    """Generate embedding using OpenAI's API."""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not configured")
        raise ValueError("OpenAI API key not configured")
    
    try:
        logger.debug(f"Generating OpenAI embedding for text of length {len(text)}")
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
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

def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    try:
        similarity = float(1 - cosine(emb1, emb2))
        logger.debug(f"Computed similarity: {similarity:.4f}")
        return similarity
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0
