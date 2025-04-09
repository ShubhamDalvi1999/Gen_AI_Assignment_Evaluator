import requests
import json
import os
import sys
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List
import traceback
import logging


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


class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

# API URLs and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))


def generate_feedback(student_code: str, ideal_code: str, similarity: float, 
                    structure_analysis: Dict[str, Any], similar_contexts: List[Dict[str, Any]]) -> str:
    """Generate feedback using Ollama."""
    logger.info(f"Generating feedback with Ollama (similarity score: {similarity:.4f})")
    start_time = datetime.now()
    
    try:
        # Format the prompt
        logger.debug("Formatting prompt for feedback generation")
        prompt = f"""
        You are an AI code reviewer evaluating a student's function implementation against an ideal solution. 
        Your task is to provide constructive feedback on the correctness and quality of the student's code.

        ### Ideal Code:
        ```python
        {ideal_code}
        ```

        ### Student Code:
        ```python
        {student_code}
        ```

        ### Similarity Score: {similarity:.2f}

        ### Structure Analysis:
        {json.dumps(structure_analysis, indent=2)}

        Based on this information, provide 2-3 paragraphs of constructive feedback for the student, 
        highlighting strengths and areas for improvement.
        """
        
        # Call Ollama API
        logger.debug(f"Calling Ollama API at {OLLAMA_BASE_URL}")
        api_start = datetime.now()
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": "You are a helpful programming assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "stream": False
            },
            timeout=30
        )
        api_time = (datetime.now() - api_start).total_seconds()
        logger.debug(f"Ollama API response received in {api_time:.2f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                feedback = result["message"]["content"]
                feedback_length = len(feedback)
                logger.info(f"Generated feedback successfully ({feedback_length} chars)")
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                return feedback
            elif "response" in result:
                feedback = result["response"]
                feedback_length = len(feedback)
                logger.info(f"Generated feedback successfully ({feedback_length} chars)")
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                return feedback
            else:
                logger.error(f"Unexpected response format from Ollama: {result}")
                return "Error generating feedback."
        else:
            error_msg = f"Error generating feedback: {response.status_code}"
            logger.error(f"Ollama API error: {response.status_code}, {response.text}")
            return error_msg
            
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error generating feedback after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating feedback: {str(e)}"
