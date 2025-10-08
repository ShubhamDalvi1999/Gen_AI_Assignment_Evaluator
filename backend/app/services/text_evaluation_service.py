from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import tempfile
import os
import traceback
import json

from ..schemas.evaluate import TextEvaluationResult, EmbeddingModel
from ..repositories.text_rag_repository import TextRAGRepository
from ..utils.docx_processor import DocxProcessor
from ..core.config import settings
from ..core.logging import service_logger as logger
from ..core.benchmarking import benchmark_operation, benchmark_context
from ..core.rag_logging import RAGSessionLogger, start_fresh_rag_session


class TextEvaluationService:
    """Service for handling text Q&A evaluation logic."""
    
    def __init__(self):
        self.text_rag_repository = TextRAGRepository()
        
        # Define similarity thresholds
        self.question_similarity_threshold = 0.7
        self.high_quality_threshold = 0.92
        self.medium_quality_threshold = 0.75
        self.low_quality_threshold = 0.60
    
    @benchmark_operation("text_evaluation")
    async def evaluate_text(
        self,
        submission_path: str,
        ideal_path: str,
        model: EmbeddingModel
    ) -> TextEvaluationResult:
        """Evaluate text Q&A submission against ideal solution."""
        
        # Start fresh RAG session with cleared logs
        start_fresh_rag_session()
        
        try:
            logger.info("Starting text Q&A evaluation")
            start_time = datetime.now()
            
            # Configure model
            use_openai = (model == EmbeddingModel.OPENAI)
            self.text_rag_repository.use_openai = use_openai
            
            # Process documents - extract Q&A pairs and generate embeddings
            logger.info("Processing ideal document")
            ideal_qa_pairs = self.text_rag_repository.process_qa_document(ideal_path, is_ideal=True)
            
            if not ideal_qa_pairs:
                return TextEvaluationResult(
                    status="error",
                    message="No Q&A pairs found in ideal document",
                    model_used=model.value
                )
            
            logger.info("Processing student submission")
            submission_id = self.text_rag_repository.generate_submission_id()
            submission_qa_pairs = self.text_rag_repository.process_qa_document(
                submission_path, is_ideal=False, submission_id=submission_id
            )
            
            if not submission_qa_pairs:
                return TextEvaluationResult(
                    status="error",
                    message="No Q&A pairs found in student submission",
                model_used=model.value
            )
            
            # Map and evaluate Q&A pairs
            logger.info("Mapping Q&A pairs")
            qa_mappings = self._map_qa_pairs(submission_qa_pairs)
            
            # Calculate scores
            logger.info("Calculating scores")
            total_questions = len(ideal_qa_pairs)
            high_matches = sum(1 for m in qa_mappings if m["quality"] == "high")
            medium_matches = sum(1 for m in qa_mappings if m["quality"] == "medium")
            low_matches = sum(1 for m in qa_mappings if m["quality"] == "low")
            poor_matches = sum(1 for m in qa_mappings if m["quality"] == "poor")
            missing = sum(1 for m in qa_mappings if m["quality"] == "missing")
            
            # Calculate overall score
            if total_questions > 0:
                overall_score = round(
                    (high_matches * 1.0 + medium_matches * 0.7 + low_matches * 0.4 + poor_matches * 0.1) 
                    / total_questions * 100
                )
            else:
                overall_score = 0
            
            # Generate summary and evaluations
            logger.info("Generating feedback")
            summary = self._generate_summary("", total_questions, high_matches, medium_matches, low_matches, overall_score)
            evaluations = self._format_evaluations_for_ui(qa_mappings, submission_qa_pairs, ideal_qa_pairs)
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Text evaluation completed in {evaluation_time:.2f}s")
            
            return TextEvaluationResult(
                status="success",
                session_id=str(submission_id),
                matched_questions=len([m for m in qa_mappings if m["quality"] != "missing"]),
                average_similarity=sum(m.get("similarity", 0) for m in qa_mappings) / len(qa_mappings) if qa_mappings else 0,
                processed_questions=qa_mappings,
                model_used=model.value,
                overall_score=overall_score,
                evaluations=evaluations,
                summary=summary,
                stats={
                    "total_questions": total_questions,
                    "high_count": high_matches,
                    "medium_count": medium_matches,
                    "low_count": low_matches,
                    "poor_count": poor_matches,
                    "missing_count": missing
                }
            )
            
        except Exception as e:
            logger.error(f"Text evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return TextEvaluationResult(
                status="error",
                message=f"Evaluation failed: {str(e)}",
                model_used=model.value
            )
    
    def _map_qa_pairs(self, submission_qa_pairs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map student Q&A pairs to ideal Q&A pairs based on similarity."""
        logger.info("Starting Q&A pair mapping")
        
        qa_mappings = []
        
        for sub_qa_id, sub_qa_data in submission_qa_pairs.items():
            try:
                # Find the best matching ideal Q&A pair
                best_match = self.text_rag_repository.find_best_qa_match(
                    sub_qa_data["question"], 
                    sub_qa_data["answer"]
                )
                
                if best_match:
                    # Determine quality based on similarity scores
                    answer_similarity = best_match.get("answer_similarity", 0)
                    
                    if answer_similarity >= self.high_quality_threshold:
                        quality = "high"
                    elif answer_similarity >= self.medium_quality_threshold:
                        quality = "medium"
                    elif answer_similarity >= self.low_quality_threshold:
                        quality = "low"
                    else:
                        quality = "poor"
                    
                    qa_mappings.append({
                        "student_qa_id": sub_qa_id,
                        "ideal_qa_id": best_match.get("qa_id"),
                        "question_similarity": best_match.get("question_similarity", 0),
                        "answer_similarity": answer_similarity,
                        "similarity": answer_similarity,  # For compatibility
                        "quality": quality,
                        "student_question": sub_qa_data["question"],
                        "student_answer": sub_qa_data["answer"]
                    })
                else:
                    # No matching ideal Q&A found
                    qa_mappings.append({
                        "student_qa_id": sub_qa_id,
                        "ideal_qa_id": None,
                        "question_similarity": 0,
                        "answer_similarity": 0,
                        "similarity": 0,
                        "quality": "missing",
                        "student_question": sub_qa_data["question"],
                        "student_answer": sub_qa_data["answer"]
                    })
                    
            except Exception as e:
                logger.error(f"Error mapping Q&A pair {sub_qa_id}: {e}")
                qa_mappings.append({
                    "student_qa_id": sub_qa_id,
                    "ideal_qa_id": None,
                    "question_similarity": 0,
                    "answer_similarity": 0,
                    "similarity": 0,
                    "quality": "missing",
                    "student_question": sub_qa_data.get("question", ""),
                    "student_answer": sub_qa_data.get("answer", ""),
                    "error": str(e)
                })
        
        logger.info(f"Mapped {len(qa_mappings)} Q&A pairs")
        return qa_mappings
    
    def _generate_summary(self, session_id: str, total_questions: int, high_count: int, 
                         medium_count: int, low_count: int, overall_score: int) -> str:
        """Generate evaluation summary."""
        
        summary_parts = [
            f"Overall Score: {overall_score}%",
            f"Total Questions Evaluated: {total_questions}",
            f"High Quality Matches: {high_count}",
            f"Medium Quality Matches: {medium_count}",
            f"Low Quality Matches: {low_count}"
        ]
        
        if overall_score >= 90:
            performance = "Excellent work! Your answers demonstrate strong understanding."
        elif overall_score >= 75:
            performance = "Good work! Most of your answers are well-aligned with the expected responses."
        elif overall_score >= 60:
            performance = "Fair performance. Some answers need improvement for better accuracy."
        else:
            performance = "Needs improvement. Consider reviewing the material and refining your answers."
        
        summary_parts.append(f"Performance Assessment: {performance}")
        
        return "\n".join(summary_parts)
    
    def _format_evaluations_for_ui(self, qa_mappings: List[Dict[str, Any]], 
                                  submission_qa_pairs: Dict[str, Dict[str, Any]], 
                                  ideal_qa_pairs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format evaluations for UI display."""
        
        evaluations = []
        
        for mapping in qa_mappings:
            evaluation = {
                "student_qa_id": mapping["student_qa_id"],
                "student_question": mapping["student_question"],
                "student_answer": mapping["student_answer"],
                "quality": mapping["quality"],
                "similarity": mapping["similarity"],
                "feedback": self._generate_feedback_for_qa(mapping, ideal_qa_pairs)
            }
            
            # Add ideal answer if available
            if mapping["ideal_qa_id"] and mapping["ideal_qa_id"] in ideal_qa_pairs:
                ideal_qa = ideal_qa_pairs[mapping["ideal_qa_id"]]
                evaluation["ideal_question"] = ideal_qa["question"]
                evaluation["ideal_answer"] = ideal_qa["answer"]
            
            evaluations.append(evaluation)
        
        return evaluations
    
    def _generate_feedback_for_qa(self, mapping: Dict[str, Any], 
                                 ideal_qa_pairs: Dict[str, Dict[str, Any]]) -> str:
        """Generate feedback for a specific Q&A pair."""
        
        quality = mapping["quality"]
        similarity = mapping["similarity"]
        
        if quality == "high":
            return f"Excellent answer! Your response shows strong understanding and aligns well with the expected answer (similarity: {similarity:.2f})."
        elif quality == "medium":
            return f"Good answer with room for improvement. Consider refining your response for better accuracy (similarity: {similarity:.2f})."
        elif quality == "low":
            return f"Your answer partially addresses the question but needs significant improvement (similarity: {similarity:.2f})."
        elif quality == "poor":
            return f"Your answer needs substantial revision to better match the expected response (similarity: {similarity:.2f})."
        else:  # missing
            return "No matching question found in the ideal answers. Please ensure your question is clearly stated."
