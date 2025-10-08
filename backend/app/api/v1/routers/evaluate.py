from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import tempfile
import os
import shutil
import traceback
from datetime import datetime

from ....schemas.evaluate import EmbeddingModel, CodeEvaluationResult
from ....services.code_evaluation_service import CodeEvaluationService
from ....core.logging import app_logger as logger

router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluate"])


@router.post("/code")
async def evaluate_code(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA),
    use_openai_feedback: bool = Form(False)
) -> CodeEvaluationResult:
    """Handle code evaluation."""
    logger.info(f"Starting code evaluation - Student: {submission.filename}, Ideal: {ideal.filename}, Model: {model}")
    logger.info(f"OpenAI feedback enabled: {use_openai_feedback}")
    
    evaluation_start_time = datetime.now()
    
    # Validate file extensions
    for upload_file, file_name in [(submission, "Student submission"), (ideal, "Ideal solution")]:
        if not upload_file.filename.lower().endswith('.zip'):
            logger.error(f"{file_name} has invalid extension: {upload_file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} must be a ZIP file. Please upload a file with .zip extension."
            )
        
        # Check file size (10MB limit)
        file_size = 0
        try:
            file_content = await upload_file.read()
            file_size = len(file_content)
            logger.info(f"Read {file_name} file: {file_size / 1024:.1f} KB")
            # Reset the file pointer for later reading
            await upload_file.seek(0)
        except Exception as e:
            logger.error(f"Error reading {file_name}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading {file_name}: {str(e)}"
            )
            
        if file_size > 10 * 1024 * 1024:  # 10MB in bytes
            logger.error(f"{file_name} exceeds size limit: {file_size / (1024 * 1024):.1f} MB > 10 MB")
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} exceeds the 10MB size limit. Please upload a smaller file."
            )
            
        # Basic ZIP validation check
        try:
            # Read a small portion to check if it's a valid ZIP
            signature = file_content[:4]
            # ZIP files start with PK\x03\x04 signature
            if signature != b'PK\x03\x04':
                logger.error(f"{file_name} has invalid ZIP signature: {signature}")
                raise HTTPException(
                    status_code=400,
                    detail=f"{file_name} does not appear to be a valid ZIP file. Please check the file format."
                )
        except Exception as e:
            logger.error(f"Error validating {file_name}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error validating {file_name}: {str(e)}"
            )
    
    # Initialize file paths before try block
    student_zip_path = None
    ideal_path = None
    
    # Create temporary files to extract the uploads
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for evaluation: {temp_dir}")
    
    try:
        # Save uploaded files to temporary location
        logger.info("Saving uploaded files to temporary location")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as student_temp_file:
            content = await submission.read()
            student_temp_file.write(content)
            student_zip_path = student_temp_file.name
            logger.info(f"Saved student submission to: {student_zip_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as ideal_temp:
            content = await ideal.read()
            ideal_temp.write(content)
            ideal_path = ideal_temp.name
            logger.info(f"Saved ideal solution to: {ideal_path}")

        # Initialize evaluation service
        evaluation_service = CodeEvaluationService()
        
        # Perform evaluation
        result = await evaluation_service.evaluate_code(
            student_zip_path=student_zip_path,
            ideal_zip_path=ideal_path,
            model=model,
            use_openai_feedback=use_openai_feedback
        )
        
        total_evaluation_time = (datetime.now() - evaluation_start_time).total_seconds()
        logger.info(f"Completed full evaluation in {total_evaluation_time:.2f}s")
        
        return result

    except Exception as e:
        error_time = (datetime.now() - evaluation_start_time).total_seconds()
        logger.error(f"Evaluation failed after {error_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        logger.info("Cleaning up temporary files")
        for path in [student_zip_path, ideal_path]:
            if path is not None and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temporary file {path}: {cleanup_error}")
        
        # Clean up temp directory if it exists
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Removed temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove temporary directory {temp_dir}: {cleanup_error}")


@router.post("/text")
async def evaluate_text(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA)
):
    """Handle text Q&A evaluation."""
    from ....schemas.evaluate import TextEvaluationResult
    from ....services.text_evaluation_service import TextEvaluationService
    
    logger.info(f"Starting text evaluation - Student: {submission.filename}, Ideal: {ideal.filename}, Model: {model}")
    
    evaluation_start_time = datetime.now()
    
    # Validate file extensions
    for upload_file, file_name in [(submission, "Student submission"), (ideal, "Ideal solution")]:
        if not upload_file.filename.lower().endswith('.docx'):
            logger.error(f"{file_name} has invalid extension: {upload_file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} must be a DOCX file. Please upload a file with .docx extension."
            )
        
        # Check file size (10MB limit)
        file_size = 0
        try:
            file_content = await upload_file.read()
            file_size = len(file_content)
            logger.info(f"Read {file_name} file: {file_size / 1024:.1f} KB")
            # Reset the file pointer for later reading
            await upload_file.seek(0)
        except Exception as e:
            logger.error(f"Error reading {file_name}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading {file_name}: {str(e)}"
            )
            
        if file_size > 10 * 1024 * 1024:  # 10MB in bytes
            logger.error(f"{file_name} exceeds size limit: {file_size / (1024 * 1024):.1f} MB > 10 MB")
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} exceeds the 10MB size limit. Please upload a smaller file."
            )
    
    # Initialize file paths before try block
    submission_path = None
    ideal_path = None
    
    try:
        # Save uploaded files to temporary location
        logger.info("Saving uploaded files to temporary location")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as submission_temp:
            content = await submission.read()
            submission_temp.write(content)
            submission_path = submission_temp.name
            logger.info(f"Saved student submission to: {submission_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as ideal_temp:
            content = await ideal.read()
            ideal_temp.write(content)
            ideal_path = ideal_temp.name
            logger.info(f"Saved ideal solution to: {ideal_path}")

        # Initialize evaluation service
        evaluation_service = TextEvaluationService()
        
        # Perform evaluation
        result = await evaluation_service.evaluate_text(
            submission_path=submission_path,
            ideal_path=ideal_path,
            model=model
        )
        
        total_evaluation_time = (datetime.now() - evaluation_start_time).total_seconds()
        logger.info(f"Completed text evaluation in {total_evaluation_time:.2f}s")
        
        return result

    except Exception as e:
        error_time = (datetime.now() - evaluation_start_time).total_seconds()
        logger.error(f"Text evaluation failed after {error_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Text evaluation failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        logger.info("Cleaning up temporary files")
        for path in [submission_path, ideal_path]:
            if path is not None and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temporary file {path}: {cleanup_error}")


# Create a separate router for estimate endpoints
router_tokens = APIRouter(prefix="/api/v1/estimate", tags=["estimate"])


@router_tokens.post("/tokens")
async def estimate_tokens(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA)
):
    """Estimate token usage for uploaded files."""
    from ....schemas.evaluate import TokenEstimateResult
    from ....services.token_estimation_service import TokenEstimationService
    
    logger.info(f"Starting token estimation - Student: {submission.filename}, Ideal: {ideal.filename}, Model: {model}")
    
    estimation_start_time = datetime.now()
    
    # Validate files are not empty
    for upload_file, file_name in [(submission, "Student submission"), (ideal, "Ideal solution")]:
        try:
            file_content = await upload_file.read()
            file_size = len(file_content)
            logger.info(f"Read {file_name} file: {file_size / 1024:.1f} KB")
            # Reset the file pointer for later reading
            await upload_file.seek(0)
            
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded {file_name.lower()} file is empty."
                )
        except Exception as e:
            logger.error(f"Error reading {file_name}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading {file_name}: {str(e)}"
            )
    
    # Initialize file paths before try block
    submission_path = None
    ideal_path = None
    
    try:
        # Create temp directory
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files to temporary location
        submission_path = os.path.join(temp_dir, os.path.basename(submission.filename))
        ideal_path = os.path.join(temp_dir, os.path.basename(ideal.filename))
        
        with open(submission_path, "wb") as f:
            submission_bytes = await submission.read()
            f.write(submission_bytes)
            logger.info(f"Saved submission file: {submission_path} ({len(submission_bytes)} bytes)")
        
        with open(ideal_path, "wb") as f:
            ideal_bytes = await ideal.read()
            f.write(ideal_bytes)
            logger.info(f"Saved ideal file: {ideal_path} ({len(ideal_bytes)} bytes)")

        # Initialize estimation service
        estimation_service = TokenEstimationService()
        
        # Perform estimation
        result = await estimation_service.estimate_tokens(
            submission_path=submission_path,
            ideal_path=ideal_path,
            model=model
        )
        
        total_estimation_time = (datetime.now() - estimation_start_time).total_seconds()
        logger.info(f"Completed token estimation in {total_estimation_time:.2f}s")
        
        return result

    except Exception as e:
        error_time = (datetime.now() - estimation_start_time).total_seconds()
        logger.error(f"Token estimation failed after {error_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Token estimation failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        logger.info("Cleaning up temporary files")
        for path in [submission_path, ideal_path]:
            if path is not None and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temporary file {path}: {cleanup_error}")
