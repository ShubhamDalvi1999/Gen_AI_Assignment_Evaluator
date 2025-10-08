from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class CodeEvaluateRequest(BaseModel):
    """Request schema for code evaluation."""
    submission: bytes = Field(..., description="Student submission ZIP file")
    ideal: bytes = Field(..., description="Ideal solution ZIP file")
    model: EmbeddingModel = Field(default=EmbeddingModel.OLLAMA, description="Embedding model to use")
    use_openai_feedback: bool = Field(default=False, description="Use OpenAI for enhanced feedback")


class TextEvaluateRequest(BaseModel):
    """Request schema for text Q&A evaluation."""
    submission: bytes = Field(..., description="Student submission DOCX file")
    ideal: bytes = Field(..., description="Ideal solution DOCX file")
    model: EmbeddingModel = Field(default=EmbeddingModel.OLLAMA, description="Embedding model to use")


class TokenEstimateRequest(BaseModel):
    """Request schema for token estimation."""
    submission: bytes = Field(..., description="Student submission file")
    ideal: bytes = Field(..., description="Ideal solution file")
    model: EmbeddingModel = Field(default=EmbeddingModel.OLLAMA, description="Embedding model to use")


class EvaluationResult(BaseModel):
    """Base evaluation result schema."""
    status: str = Field(..., description="Evaluation status")
    message: Optional[str] = Field(None, description="Status message")


class CodeEvaluationResult(EvaluationResult):
    """Code evaluation result schema."""
    functions_evaluated: int = Field(0, description="Number of functions evaluated")
    average_similarity: float = Field(0.0, description="Average similarity score")
    function_results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual function results")
    extra_functions: List[str] = Field(default_factory=list, description="Extra functions in submission")
    missing_functions: List[str] = Field(default_factory=list, description="Missing functions in submission")


class TextEvaluationResult(EvaluationResult):
    """Text Q&A evaluation result schema."""
    session_id: Optional[str] = Field(None, description="Evaluation session ID")
    matched_questions: int = Field(0, description="Number of matched questions")
    average_similarity: float = Field(0.0, description="Average similarity score")
    processed_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Processed question results")
    model_used: str = Field(..., description="Model used for evaluation")
    overall_score: float = Field(0.0, description="Overall evaluation score")
    evaluations: List[Dict[str, Any]] = Field(default_factory=list, description="Individual question evaluations")
    summary: str = Field("", description="Evaluation summary")
    stats: Dict[str, int] = Field(default_factory=dict, description="Evaluation statistics")


class TokenEstimateResult(BaseModel):
    """Token estimation result schema."""
    status: str = Field(..., description="Estimation status")
    message: Optional[str] = Field(None, description="Status message")
    estimated_tokens: int = Field(0, description="Estimated token count")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")
    warnings: List[str] = Field(default_factory=list, description="Any warnings about the estimation")
