from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings using pydantic-settings for type safety."""
    
    # MongoDB Configuration
    mongodb_uri: Optional[str] = None
    mongodb_db_name: str = "assignment_checker"
    mongodb_embeddings_collection: str = "embeddings"
    mongodb_qa_collection: str = "qa_embeddings"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_api_url: str = "https://api.openai.com/v1/embeddings"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_url: str = "http://localhost:11434/api/embeddings"
    ollama_embedding_model: str = "llama3"
    
    # Similarity Thresholds
    similarity_threshold: float = 0.8
    
    # CORS Configuration (for future React frontend)
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    
    # Upload Configuration
    upload_folder: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")  # Absolute path to .env
        case_sensitive = False
        extra = "ignore"


# Create settings instance
settings = Settings()
