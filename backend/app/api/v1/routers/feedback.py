from fastapi import APIRouter, HTTPException, Form
from typing import Dict, Any, List
import traceback
from datetime import datetime

from ....schemas.evaluate import EmbeddingModel
from ....services.feedback_service import (
    generate_code_feedback,
    generate_text_feedback,
    generate_summary_feedback,
    generate_code_summary_feedback
)
from ....core.logging import app_logger as logger

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("/code")
async def generate_code_feedback_endpoint(
    student_code: str = Form(...),
    ideal_code: str = Form(...),
    similarity: float = Form(...),
    structure_analysis: str = Form(...),  # JSON string
    similar_contexts: str = Form("[]"),  # JSON string, optional
    use_openai: bool = Form(False)
) -> Dict[str, Any]:
    """Generate feedback for code evaluation."""
    logger.info(f"Generating code feedback (similarity: {similarity:.4f}, use_openai: {use_openai})")
    
    try:
        import json
        
        # Parse JSON strings
        try:
            structure_analysis_dict = json.loads(structure_analysis)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid structure_analysis JSON")
        
        try:
            similar_contexts_list = json.loads(similar_contexts)
        except json.JSONDecodeError:
            similar_contexts_list = []
        
        # Generate feedback
        feedback = generate_code_feedback(
            student_code=student_code,
            ideal_code=ideal_code,
            similarity=similarity,
            structure_analysis=structure_analysis_dict,
            similar_contexts=similar_contexts_list,
            use_openai=use_openai
        )
        
        return {
            "status": "success",
            "feedback": feedback,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating code feedback: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")


@router.post("/text")
async def generate_text_feedback_endpoint(
    student_answer: str = Form(...),
    reference_answer: str = Form(...),
    similarity: float = Form(...),
    use_openai: bool = Form(False)
) -> Dict[str, Any]:
    """Generate feedback for text Q&A evaluation."""
    logger.info(f"Generating text feedback (similarity: {similarity:.4f}, use_openai: {use_openai})")
    
    try:
        # Generate feedback
        feedback = generate_text_feedback(
            student_answer=student_answer,
            reference_answer=reference_answer,
            similarity=similarity,
            use_openai=use_openai
        )
        
        return {
            "status": "success",
            "feedback": feedback,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating text feedback: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")


@router.post("/summary")
async def generate_summary_feedback_endpoint(
    question_evaluations: str = Form(...),  # JSON string
    total_questions: int = Form(...),
    high_count: int = Form(...),
    medium_count: int = Form(...),
    low_count: int = Form(...),
    overall_score: float = Form(...),
    use_openai: bool = Form(False)
) -> Dict[str, Any]:
    """Generate comprehensive summary feedback for text Q&A evaluation."""
    logger.info(f"Generating summary feedback (overall_score: {overall_score:.2f}%, use_openai: {use_openai})")
    
    try:
        import json
        
        # Parse JSON string
        try:
            question_evaluations_list = json.loads(question_evaluations)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid question_evaluations JSON")
        
        # Generate feedback
        feedback = generate_summary_feedback(
            question_evaluations=question_evaluations_list,
            total_questions=total_questions,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            overall_score=overall_score,
            use_openai=use_openai
        )
        
        return {
            "status": "success",
            "feedback": feedback,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating summary feedback: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")


@router.post("/code-summary")
async def generate_code_summary_feedback_endpoint(
    function_evaluations: str = Form(...),  # JSON string
    total_functions: int = Form(...),
    high_count: int = Form(...),
    medium_count: int = Form(...),
    low_count: int = Form(...),
    poor_count: int = Form(...),
    missing_count: int = Form(...),
    average_similarity: float = Form(...),
    extra_functions: str = Form("[]"),  # JSON string
    missing_functions: str = Form("[]"),  # JSON string
    use_openai: bool = Form(False)
) -> Dict[str, Any]:
    """Generate comprehensive summary feedback for code evaluation."""
    logger.info(f"Generating code summary feedback (average_similarity: {average_similarity:.4f}, use_openai: {use_openai})")
    
    try:
        import json
        
        # Parse JSON strings
        try:
            function_evaluations_list = json.loads(function_evaluations)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid function_evaluations JSON")
        
        try:
            extra_functions_list = json.loads(extra_functions)
        except json.JSONDecodeError:
            extra_functions_list = []
        
        try:
            missing_functions_list = json.loads(missing_functions)
        except json.JSONDecodeError:
            missing_functions_list = []
        
        # Generate feedback
        feedback = generate_code_summary_feedback(
            function_evaluations=function_evaluations_list,
            total_functions=total_functions,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            poor_count=poor_count,
            missing_count=missing_count,
            average_similarity=average_similarity,
            extra_functions=extra_functions_list,
            missing_functions=missing_functions_list,
            use_openai=use_openai
        )
        
        return {
            "status": "success",
            "feedback": feedback,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating code summary feedback: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")
