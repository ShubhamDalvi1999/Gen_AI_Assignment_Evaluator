import tiktoken
import logging

# Configure logging
logger = logging.getLogger(__name__)

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

# Initialize common tokenizers at import time to reduce latency during runtime
def initialize_tokenizers():
    """Initialize common tokenizers on module import to speed up first use."""
    try:
        for model_name in ["gpt-4o", "gpt-3.5-turbo", "cl100k_base"]:
            _get_tokenizer(model_name)
        logger.info(f"Pre-initialized {len(_TOKENIZER_CACHE)} tokenizers")
    except Exception as e:
        logger.warning(f"Error pre-initializing tokenizers: {e}")

# Initialize common tokenizers when this module is imported
initialize_tokenizers()

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model to use for tokenization
        
    Returns:
        The number of tokens in the text
    """
    if not text:
        return 0
    
    # Get tokenizer from cache
    enc = _get_tokenizer(model)
    if enc is None:
        # Fall back to approximate counting if no tokenizer is available
        return _approximate_token_count(text)
        
    try:
        # For large documents, process in chunks to avoid memory issues
        if len(text) > 100000:  # For very large texts (>100K chars)
            chunk_size = 50000  # Process 50K characters at a time
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            total_tokens = 0
            for chunk in chunks:
                total_tokens += len(enc.encode(chunk))
            return total_tokens
        else:
            # Process normally for smaller texts
            return len(enc.encode(text))
    except Exception as e:
        # Last resort: use approximate counting
        logger.warning(f"Error counting tokens: {e}. Falling back to approximate count.")
        return _approximate_token_count(text)

def _approximate_token_count(text: str) -> int:
    """
    Approximate token count when tiktoken is unavailable.
    This is a rough estimate based on word count and punctuation.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Split by whitespace to count words
    words = text.split()
    # Count punctuation and special characters
    punctuation = sum(1 for char in text if char in ".,;:!?()[]{}\"'+-*/=<>@#$%^&|~`")
    # Typical English has ~1.3 tokens per word
    estimated_tokens = int(len(words) * 1.3) + punctuation
    return max(1, estimated_tokens)  # At least 1 token

def truncate_by_token_limit(texts: list[str], max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate a list of text chunks to stay within token limit."""
    # Get tokenizer from cache
    enc = _get_tokenizer(model)
    if enc is None:
        # If no tokenizer is available, use approximate counting to truncate
        current_tokens = 0
        output_chunks = []
        for text in texts:
            approx_tokens = _approximate_token_count(text)
            if current_tokens + approx_tokens > max_tokens:
                break
            output_chunks.append(text)
            current_tokens += approx_tokens
        return "\n\n".join(output_chunks)
        
    current_tokens = 0
    output_chunks = []

    for text in texts:
        tokens = enc.encode(text)
        if current_tokens + len(tokens) > max_tokens:
            # If we can't fit the entire chunk, see if we can fit part of it
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # If we have space for at least 50 tokens
                partial_text = enc.decode(tokens[:remaining_tokens])
                output_chunks.append(partial_text + "... [truncated]")
            break
        output_chunks.append(text)
        current_tokens += len(tokens)

    return "\n\n".join(output_chunks)

def safe_truncate_code(code: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Safely truncate code to a maximum token count."""
    # Get tokenizer from cache
    enc = _get_tokenizer(model)
    if enc is None:
        # Fall back to approximate truncation
        approx_tokens = _approximate_token_count(code)
        if approx_tokens <= max_tokens:
            return code
        # Roughly truncate the code based on character count (approximation)
        ratio = max_tokens / approx_tokens
        char_limit = int(len(code) * ratio * 0.85)  # Apply a safety factor
        return code[:char_limit] + "\n# ... [code truncated due to length]"
    
    tokens = enc.encode(code)
    if len(tokens) <= max_tokens:
        return code
    
    # When truncating, add a comment indicating truncation
    truncated = enc.decode(tokens[:max_tokens])
    return truncated + "\n# ... [code truncated due to length]" 