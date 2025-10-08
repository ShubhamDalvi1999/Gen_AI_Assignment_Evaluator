import tiktoken
import logging
from typing import Optional

from ..core.logging import service_logger as logger

# Global tokenizer cache to avoid repeatedly creating tokenizers
_TOKENIZER_CACHE = {}


def _get_tokenizer(model: str = "gpt-4o"):
    """
    Get a tokenizer from the cache or create a new one.
    
    Args:
        model: Model name to get tokenizer for
        
    Returns:
        Tokenizer encoding object
    """
    if model in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model]
    
    try:
        # Try to get the specific model encoding
        try:
            enc = tiktoken.encoding_for_model(model)
        except (KeyError, ImportError, AttributeError):
            # Fall back to cl100k_base if the model isn't available
            try:
                logger.debug(f"Model {model} not found, falling back to cl100k_base")
                enc = tiktoken.get_encoding("cl100k_base")
                model = "cl100k_base"  # Store in cache under this key
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not initialize tiktoken: {e}")
                return None
        
        # Cache the tokenizer
        _TOKENIZER_CACHE[model] = enc
        return enc
    except Exception as e:
        logger.error(f"Error initializing tokenizer: {e}")
        return None


def initialize_tokenizers():
    """Initialize common tokenizers on module import to speed up first use."""
    try:
        for model_name in ["gpt-4o", "gpt-3.5-turbo", "cl100k_base"]:
            _get_tokenizer(model_name)
        logger.info(f"Pre-initialized {len(_TOKENIZER_CACHE)} tokenizers")
    except Exception as e:
        logger.warning(f"Error pre-initializing tokenizers: {e}")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in the provided text using the specified model's tokenizer.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for tokenization (default: gpt-4o)
        
    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0
        
    enc = _get_tokenizer(model)
    if enc is None:
        # Fallback: rough estimation (4 chars per token on average)
        logger.warning("No tokenizer available, using character-based estimation")
        return len(text) // 4
    
    try:
        tokens = enc.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback: rough estimation
        return len(text) // 4


def safe_truncate_code(code: str, max_tokens: int = 8000, model: str = "gpt-4o") -> str:
    """
    Safely truncate code to fit within token limits while preserving structure.
    
    Args:
        code: Code string to truncate
        max_tokens: Maximum number of tokens allowed
        model: Model name to use for tokenization
        
    Returns:
        Truncated code string
    """
    if not code:
        return code
    
    current_tokens = count_tokens(code, model)
    if current_tokens <= max_tokens:
        return code
    
    logger.warning(f"Code has {current_tokens} tokens, truncating to {max_tokens}")
    
    # Calculate the rough proportion to keep
    keep_ratio = max_tokens / current_tokens
    
    # Start with a slightly smaller ratio to account for inexact estimation
    keep_ratio *= 0.9
    
    # Split into lines and keep the first portion
    lines = code.split('\n')
    keep_lines = int(len(lines) * keep_ratio)
    
    if keep_lines < 1:
        keep_lines = 1
    
    truncated = '\n'.join(lines[:keep_lines])
    
    # Add a comment indicating truncation
    truncated += "\n\n# ... (code truncated due to token limit) ..."
    
    # Double-check token count
    final_tokens = count_tokens(truncated, model)
    logger.info(f"Truncated code from {current_tokens} to {final_tokens} tokens")
    
    return truncated


def estimate_cost(tokens: int, model: str = "gpt-4o") -> Optional[float]:
    """
    Estimate the cost of processing the given number of tokens.
    
    Args:
        tokens: Number of tokens
        model: Model name
        
    Returns:
        Estimated cost in USD, or None if cost estimation is not available
    """
    # Rough cost estimates (these may change and should be updated)
    cost_per_1k_tokens = {
        "gpt-4o": 0.03,  # Input tokens
        "gpt-3.5-turbo": 0.001,
        "text-embedding-ada-002": 0.0001,
        "cl100k_base": 0.001  # Default fallback
    }
    
    rate = cost_per_1k_tokens.get(model, cost_per_1k_tokens["cl100k_base"])
    return (tokens / 1000) * rate


# Initialize tokenizers on module import
initialize_tokenizers()
