from typing import Dict, Any, Optional
import os
import traceback
from datetime import datetime

from ..schemas.evaluate import TokenEstimateResult, EmbeddingModel
from ..utils.docx_processor import DocxProcessor
from ..utils.code_analyzer import extract_functions_from_zip
from ..utils.tokenizer_utils import count_tokens
from ..core.logging import service_logger as logger


class TokenEstimationService:
    """Service for estimating token usage for file processing."""
    
    def __init__(self):
        # Token limits for safety checks
        self.safe_token_limit = 16000
        
        # Overhead multipliers
        self.text_comparison_overhead = 0.3
        self.text_qa_extraction_overhead = 0.5
        self.text_prompt_overhead = 1000
        
        self.code_structure_overhead = 0.2
        self.code_retrieval_overhead = 0.3
        self.code_prompt_overhead = 800
        
        # Safety margin
        self.safety_margin_multiplier = 0.1
    
    async def estimate_tokens(
        self,
        submission_path: str,
        ideal_path: str,
        model: EmbeddingModel
    ) -> TokenEstimateResult:
        """Estimate token usage for uploaded files."""
        
        try:
            logger.info(f"Estimating tokens for files: {os.path.basename(submission_path)} and {os.path.basename(ideal_path)}")
            
            # Determine file type based on extension
            submission_ext = os.path.splitext(submission_path)[1].lower()
            ideal_ext = os.path.splitext(ideal_path)[1].lower()
            
            is_text_file = (submission_ext in ['.docx', '.doc', '.txt'] or 
                           ideal_ext in ['.docx', '.doc', '.txt'])
            
            # Force text mode for DOCX files
            if submission_ext == '.docx' or ideal_ext == '.docx':
                is_text_file = True
                logger.info("Forced text mode due to .docx extension")
            
            if is_text_file:
                return await self._estimate_text_tokens(submission_path, ideal_path, model)
            else:
                return await self._estimate_code_tokens(submission_path, ideal_path, model)
                
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            logger.error(traceback.format_exc())
            return TokenEstimateResult(
                status="error",
                message=f"Could not estimate tokens: {str(e)}"
            )
    
    async def _estimate_text_tokens(
        self, 
        submission_path: str, 
        ideal_path: str, 
        model: EmbeddingModel
    ) -> TokenEstimateResult:
        """Estimate tokens for text/DOCX files."""
        
        try:
            logger.info("Processing as text files")
            
            # Extract text from files
            logger.debug(f"Extracting text from submission: {submission_path}")
            student_text = DocxProcessor.extract_text_from_docx(submission_path)
            
            logger.debug(f"Extracting text from ideal: {ideal_path}")
            ideal_text = DocxProcessor.extract_text_from_docx(ideal_path)
            
            # Log text lengths
            logger.debug(f"Student text length: {len(student_text) if student_text else 0} characters")
            logger.debug(f"Ideal text length: {len(ideal_text) if ideal_text else 0} characters")
            
            if not student_text:
                logger.warning(f"No text extracted from submission file: {submission_path}")
                student_text = ""
                
            if not ideal_text:
                logger.warning(f"No text extracted from ideal file: {ideal_path}")
                ideal_text = ""
            
            # Check if both texts are empty
            if not student_text and not ideal_text:
                if submission_path.lower().endswith('.docx') or ideal_path.lower().endswith('.docx'):
                    logger.warning("No text extracted from DOCX files - possible corrupt files")
                    return TokenEstimateResult(
                        status="success",
                        message="Text token estimation completed (file_type: text) - empty files detected",
                        estimated_tokens=0,
                        warnings=["No text could be extracted from the DOCX files. The files may be corrupt or empty."]
                    )
            
            # Count tokens for the extracted text
            logger.debug("Counting tokens for text files")
            student_tokens = count_tokens(student_text) if student_text else 0
            ideal_tokens = count_tokens(ideal_text) if ideal_text else 0
            
            logger.info(f"Text token counts: student={student_tokens}, ideal={ideal_tokens}")
            
            # Calculate overhead for text evaluation
            comparison_overhead = int((student_tokens + ideal_tokens) * self.text_comparison_overhead)
            qa_extraction_overhead = int(max(student_tokens, ideal_tokens) * self.text_qa_extraction_overhead)
            prompt_overhead = self.text_prompt_overhead
            
            # Calculate total estimate
            total_tokens = student_tokens + ideal_tokens + comparison_overhead + qa_extraction_overhead + prompt_overhead
            safety_margin = int(total_tokens * self.safety_margin_multiplier)
            total_estimate = total_tokens + safety_margin
            
            is_safe = total_estimate < self.safe_token_limit
            
            # Calculate estimated cost (rough estimate based on OpenAI pricing)
            cost_estimate = None
            if model == EmbeddingModel.OPENAI:
                # Rough estimate: $0.0001 per 1K tokens for embeddings
                cost_estimate = (total_estimate / 1000) * 0.0001
            
            warnings = []
            if not is_safe:
                warnings.append(f"Total token estimate ({total_estimate}) exceeds safe limit ({self.safe_token_limit})")
            
            if student_tokens == 0:
                warnings.append("No tokens found in student submission")
            
            if ideal_tokens == 0:
                warnings.append("No tokens found in ideal solution")
            
            return TokenEstimateResult(
                status="success",
                message=f"Text token estimation completed (file_type: text)",
                estimated_tokens=total_estimate,
                cost_estimate=cost_estimate,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Text token estimation failed: {e}")
            return TokenEstimateResult(
                status="error",
                message=f"Failed to process text files: {str(e)}"
            )
    
    async def _estimate_code_tokens(
        self, 
        submission_path: str, 
        ideal_path: str, 
        model: EmbeddingModel
    ) -> TokenEstimateResult:
        """Estimate tokens for code/ZIP files."""
        
        try:
            logger.info("Processing as code files")
            
            # Extract functions from ZIP files
            student_functions = extract_functions_from_zip(submission_path)
            ideal_functions = extract_functions_from_zip(ideal_path)
            
            logger.info(f"Extracted functions: student={len(student_functions)}, ideal={len(ideal_functions)}")
            
            # Count tokens in each function
            student_tokens = 0
            ideal_tokens = 0
            
            for func_name, func_code in student_functions.items():
                student_tokens += count_tokens(func_code)
            
            for func_name, func_code in ideal_functions.items():
                ideal_tokens += count_tokens(func_code)
            
            logger.info(f"Code token counts: student={student_tokens}, ideal={ideal_tokens}")
            
            # Calculate overhead estimates
            structure_overhead = int((student_tokens + ideal_tokens) * self.code_structure_overhead)
            retrieval_overhead = int(ideal_tokens * self.code_retrieval_overhead)
            prompt_overhead = self.code_prompt_overhead
            
            # Calculate total estimate
            total_tokens = student_tokens + ideal_tokens + structure_overhead + retrieval_overhead + prompt_overhead
            safety_margin = int(total_tokens * self.safety_margin_multiplier)
            total_estimate = total_tokens + safety_margin
            
            is_safe = total_estimate < self.safe_token_limit
            
            # Calculate estimated cost
            cost_estimate = None
            if model == EmbeddingModel.OPENAI:
                # Rough estimate: $0.0001 per 1K tokens for embeddings
                cost_estimate = (total_estimate / 1000) * 0.0001
            
            warnings = []
            if not is_safe:
                warnings.append(f"Total token estimate ({total_estimate}) exceeds safe limit ({self.safe_token_limit})")
            
            if len(student_functions) == 0:
                warnings.append("No functions found in student submission")
            
            if len(ideal_functions) == 0:
                warnings.append("No functions found in ideal solution")
            
            return TokenEstimateResult(
                status="success",
                message=f"Code token estimation completed (file_type: code)",
                estimated_tokens=total_estimate,
                cost_estimate=cost_estimate,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Code token estimation failed: {e}")
            return TokenEstimateResult(
                status="error",
                message=f"Failed to process code files: {str(e)}"
            )
