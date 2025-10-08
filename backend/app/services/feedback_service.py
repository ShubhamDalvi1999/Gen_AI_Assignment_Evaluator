import requests
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..core.config import settings
from ..core.logging import service_logger as logger
from ..core.rag_logging import (
    log_llm_response,
    log_openai_api_call,
    RAGSessionLogger
)
from ..utils.prompts import (
    OLLAMA_CODE_FEEDBACK_PROMPT,
    OPENAI_CODE_FEEDBACK_PROMPT,
    QA_EVALUATION_PROMPT,
    QA_SUMMARY_PROMPT
)


@log_openai_api_call("openai_chat_completion")
def _make_openai_request(prompt: str, model: str, max_tokens: int) -> requests.Response:
    """Make an OpenAI API request with proper logging."""
    return requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert programming instructor providing constructive feedback."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        },
        timeout=30
    )


@log_llm_response("code_feedback_generation")
def generate_code_feedback(
    student_code: str, 
    ideal_code: str, 
    similarity: float, 
    structure_analysis: Dict[str, Any], 
    similar_contexts: List[Dict[str, Any]],
    use_openai: bool = False
) -> str:
    """Generate feedback for code evaluation using Ollama or OpenAI."""
    logger.info(f"Generating code feedback (similarity score: {similarity:.4f}, use_openai: {use_openai})")
    start_time = datetime.now()
    
    try:
        # RETRIEVAL STAGE
        logger.info("========== RETRIEVAL STAGE ==========")
        logger.info(f"Processing {len(similar_contexts)} similar contexts")
        
        # AUGMENTATION STAGE
        logger.info("========== AUGMENTATION STAGE ==========")
        logger.debug("Formatting prompt for feedback generation")
        
        if use_openai and settings.openai_api_key:
            # Use OpenAI with detailed prompt
            prompt = OPENAI_CODE_FEEDBACK_PROMPT.format(
                student_code=student_code,
                ideal_code=ideal_code,
                similarity=similarity,
                missing_variables=structure_analysis.get('missing_variables', []),
                extra_variables=structure_analysis.get('extra_variables', []),
                missing_control_structures=structure_analysis.get('missing_control_structures', []),
                extra_control_structures=structure_analysis.get('extra_control_structures', []),
                missing_function_calls=structure_analysis.get('missing_function_calls', []),
                extra_function_calls=structure_analysis.get('extra_function_calls', [])
            )
        else:
            # Use Ollama with enhanced prompt
            prompt = OLLAMA_CODE_FEEDBACK_PROMPT.format(
                student_code=student_code,
                ideal_code=ideal_code,
                similarity=similarity,
                structure_analysis=json.dumps(structure_analysis, indent=2)
            )
        
        # GENERATION STAGE
        logger.info("========== GENERATION STAGE ==========")
        
        if use_openai and settings.openai_api_key:
            # Use OpenAI API
            logger.debug("Calling OpenAI API for feedback generation")
            api_start = datetime.now()
            response = _make_openai_request(prompt, "gpt-3.5-turbo", 1000)
        else:
            # Use Ollama API
            logger.debug(f"Calling Ollama API at {settings.ollama_base_url}")
            api_start = datetime.now()
            response = requests.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_embedding_model,
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
        logger.debug(f"API response received in {api_time:.2f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            
            if use_openai and settings.openai_api_key:
                # Handle OpenAI response
                if "choices" in result and len(result["choices"]) > 0:
                    feedback = result["choices"][0]["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated OpenAI feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected OpenAI response format: {result}")
                    return "Error generating feedback."
            else:
                # Handle Ollama response
                if "message" in result and "content" in result["message"]:
                    feedback = result["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                    return feedback
                elif "response" in result:
                    feedback = result["response"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected Ollama response format: {result}")
                    return "Error generating feedback."
        else:
            error_msg = f"Error generating feedback: {response.status_code}"
            logger.error(f"API error: {response.status_code}, {response.text}")
            return error_msg
            
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error generating feedback after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating feedback: {str(e)}"


@log_llm_response("text_feedback_generation")
def generate_text_feedback(
    student_answer: str,
    reference_answer: str,
    similarity: float,
    use_openai: bool = False
) -> str:
    """Generate feedback for text Q&A evaluation using Ollama or OpenAI."""
    logger.info(f"Generating text feedback (similarity score: {similarity:.4f}, use_openai: {use_openai})")
    start_time = datetime.now()
    
    try:
        # Format the prompt
        prompt = QA_EVALUATION_PROMPT.format(
            student_answer=student_answer,
            reference_answer=reference_answer
        )
        
        # GENERATION STAGE
        logger.info("========== GENERATION STAGE ==========")
        
        if use_openai and settings.openai_api_key:
            # Use OpenAI API
            logger.debug("Calling OpenAI API for text feedback generation")
            api_start = datetime.now()
            response = _make_openai_request(prompt, "gpt-3.5-turbo", 1500)
        else:
            # Use Ollama API
            logger.debug(f"Calling Ollama API at {settings.ollama_base_url}")
            api_start = datetime.now()
            response = requests.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_embedding_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert educational evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=30
            )
        
        api_time = (datetime.now() - api_start).total_seconds()
        logger.debug(f"API response received in {api_time:.2f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            
            if use_openai and settings.openai_api_key:
                # Handle OpenAI response
                if "choices" in result and len(result["choices"]) > 0:
                    feedback = result["choices"][0]["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated OpenAI text feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Text feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected OpenAI response format: {result}")
                    return "Error generating text feedback."
            else:
                # Handle Ollama response
                if "message" in result and "content" in result["message"]:
                    feedback = result["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama text feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Text feedback generation completed in {elapsed:.2f}s")
                    return feedback
                elif "response" in result:
                    feedback = result["response"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama text feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Text feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected Ollama response format: {result}")
                    return "Error generating text feedback."
        else:
            error_msg = f"Error generating text feedback: {response.status_code}"
            logger.error(f"API error: {response.status_code}, {response.text}")
            return error_msg
            
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error generating text feedback after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating text feedback: {str(e)}"


@log_llm_response("summary_feedback_generation")
def generate_summary_feedback(
    question_evaluations: List[Dict[str, Any]],
    total_questions: int,
    high_count: int,
    medium_count: int,
    low_count: int,
    overall_score: float,
    use_openai: bool = False
) -> str:
    """Generate comprehensive summary feedback for text Q&A evaluation."""
    logger.info(f"Generating summary feedback (overall score: {overall_score:.2f}%, use_openai: {use_openai})")
    start_time = datetime.now()
    
    try:
        # Format the prompt
        prompt = QA_SUMMARY_PROMPT.format(
            question_evaluations=json.dumps(question_evaluations, indent=2),
            total_questions=total_questions,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            overall_score=overall_score
        )
        
        # GENERATION STAGE
        logger.info("========== GENERATION STAGE ==========")
        
        if use_openai and settings.openai_api_key:
            # Use OpenAI API
            logger.debug("Calling OpenAI API for summary feedback generation")
            api_start = datetime.now()
            response = _make_openai_request(prompt, "gpt-3.5-turbo", 2000)
        else:
            # Use Ollama API
            logger.debug(f"Calling Ollama API at {settings.ollama_base_url}")
            api_start = datetime.now()
            response = requests.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_embedding_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert educational analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=30
            )
        
        api_time = (datetime.now() - api_start).total_seconds()
        logger.debug(f"API response received in {api_time:.2f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            
            if use_openai and settings.openai_api_key:
                # Handle OpenAI response
                if "choices" in result and len(result["choices"]) > 0:
                    feedback = result["choices"][0]["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated OpenAI summary feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Summary feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected OpenAI response format: {result}")
                    return "Error generating summary feedback."
            else:
                # Handle Ollama response
                if "message" in result and "content" in result["message"]:
                    feedback = result["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama summary feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Summary feedback generation completed in {elapsed:.2f}s")
                    return feedback
                elif "response" in result:
                    feedback = result["response"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama summary feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Summary feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected Ollama response format: {result}")
                    return "Error generating summary feedback."
        else:
            error_msg = f"Error generating summary feedback: {response.status_code}"
            logger.error(f"API error: {response.status_code}, {response.text}")
            return error_msg
            
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error generating summary feedback after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating summary feedback: {str(e)}"


# Legacy function for backward compatibility
def generate_feedback(student_code: str, ideal_code: str, similarity: float, 
                    structure_analysis: Dict[str, Any], similar_contexts: List[Dict[str, Any]]) -> str:
    """Legacy function for backward compatibility - generates code feedback using Ollama."""
    return generate_code_feedback(student_code, ideal_code, similarity, structure_analysis, similar_contexts, use_openai=False)


def generate_code_summary_feedback(
    function_evaluations: List[Dict[str, Any]],
    total_functions: int,
    high_count: int,
    medium_count: int,
    low_count: int,
    poor_count: int,
    missing_count: int,
    average_similarity: float,
    extra_functions: List[str],
    missing_functions: List[str],
    use_openai: bool = False
) -> str:
    """Generate comprehensive summary feedback for code evaluation."""
    logger.info(f"Generating code summary feedback (average similarity: {average_similarity:.4f}, use_openai: {use_openai})")
    start_time = datetime.now()
    
    try:
        # Create a comprehensive prompt for code evaluation summary
        prompt = f"""
You are an expert programming instructor providing comprehensive feedback on a student's code submission.

## EVALUATION SUMMARY
- Total Functions Evaluated: {total_functions}
- Average Similarity Score: {average_similarity:.2%}
- High Quality Functions: {high_count}
- Medium Quality Functions: {medium_count}
- Low Quality Functions: {low_count}
- Poor Quality Functions: {poor_count}
- Missing Functions: {missing_count}

## FUNCTION DETAILS
{json.dumps(function_evaluations, indent=2)}

## ADDITIONAL ANALYSIS
- Extra Functions in Submission: {extra_functions if extra_functions else 'None'}
- Missing Required Functions: {missing_functions if missing_functions else 'None'}

## TASK
Provide a comprehensive analysis that includes:

1. **Overall Assessment**: Evaluate the student's overall performance
2. **Strengths**: Identify what the student did well
3. **Areas for Improvement**: Point out specific areas that need work
4. **Function-by-Function Analysis**: Brief analysis of each function's quality
5. **Recommendations**: Specific suggestions for improvement
6. **Learning Objectives**: What the student should focus on next

Format your response in a clear, structured manner with appropriate headings and bullet points.
Be constructive and encouraging while being honest about areas that need improvement.
"""

        # GENERATION STAGE
        logger.info("========== GENERATION STAGE ==========")
        
        if use_openai and settings.openai_api_key:
            # Use OpenAI API
            logger.debug("Calling OpenAI API for code summary feedback generation")
            api_start = datetime.now()
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are an expert programming instructor providing comprehensive feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2500
                },
                timeout=30
            )
        else:
            # Use Ollama API
            logger.debug(f"Calling Ollama API at {settings.ollama_base_url}")
            api_start = datetime.now()
            response = requests.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_embedding_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert programming instructor providing comprehensive feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=30
            )
        
        api_time = (datetime.now() - api_start).total_seconds()
        logger.debug(f"API response received in {api_time:.2f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            
            if use_openai and settings.openai_api_key:
                # Handle OpenAI response
                if "choices" in result and len(result["choices"]) > 0:
                    feedback = result["choices"][0]["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated OpenAI code summary feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Code summary feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected OpenAI response format: {result}")
                    return "Error generating code summary feedback."
            else:
                # Handle Ollama response
                if "message" in result and "content" in result["message"]:
                    feedback = result["message"]["content"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama code summary feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Code summary feedback generation completed in {elapsed:.2f}s")
                    return feedback
                elif "response" in result:
                    feedback = result["response"]
                    feedback_length = len(feedback)
                    logger.info(f"Generated Ollama code summary feedback successfully ({feedback_length} chars)")
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Code summary feedback generation completed in {elapsed:.2f}s")
                    return feedback
                else:
                    logger.error(f"Unexpected Ollama response format: {result}")
                    return "Error generating code summary feedback."
        else:
            error_msg = f"Error generating code summary feedback: {response.status_code}"
            logger.error(f"API error: {response.status_code}, {response.text}")
            return error_msg
            
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error generating code summary feedback after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating code summary feedback: {str(e)}"
