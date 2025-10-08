from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime


class EmbeddingModel(BaseModel):
    """Schema for embedding data."""
    function_name: Optional[str] = Field(None, description="Function name for code embeddings")
    code: Optional[str] = Field(None, description="Code content")
    embedding: List[float] = Field(..., description="Embedding vector")
    question_embedding: Optional[List[float]] = Field(None, description="Question embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class QAPair(BaseModel):
    """Schema for Q&A pair data."""
    qa_id: str = Field(..., description="Unique Q&A pair identifier")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    embedding: List[float] = Field(..., description="Answer embedding vector")
    question_embedding: Optional[List[float]] = Field(None, description="Question embedding vector")
    is_ideal: bool = Field(..., description="Whether this is an ideal answer")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class StudentSubmission(BaseModel):
    """Schema for student submission data."""
    submission_id: int = Field(..., description="Unique submission identifier")
    qa_id: str = Field(..., description="Q&A pair identifier")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    embedding: List[float] = Field(..., description="Answer embedding vector")
    question_embedding: Optional[List[float]] = Field(None, description="Question embedding vector")
    is_ideal: bool = Field(False, description="Always False for student submissions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class SubmissionCounter(BaseModel):
    """Schema for submission counter data."""
    _id: str = Field(..., description="Counter identifier")
    value: int = Field(..., description="Current counter value")


class EmbeddingRequest(BaseModel):
    """Schema for embedding generation request."""
    text: str = Field(..., description="Text to embed")
    model: str = Field(..., description="Model to use for embedding")


class EmbeddingResponse(BaseModel):
    """Schema for embedding generation response."""
    embedding: List[float] = Field(..., description="Generated embedding vector")
    model: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Embedding dimensions")
