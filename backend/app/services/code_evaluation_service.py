from typing import Dict, Any, List
from datetime import datetime
import tempfile
import os
import shutil
import traceback

from ..schemas.evaluate import CodeEvaluationResult, EmbeddingModel
from ..services.embedding_service import get_embedding, compute_similarity, EmbeddingModel as ServiceEmbeddingModel
from ..services.feedback_service import generate_feedback
from ..repositories.embedding_repository import EmbeddingRepository
from ..utils.code_analyzer import extract_functions_from_zip, analyze_code_structure, generate_recommendations
from ..core.config import settings
from ..core.logging import service_logger as logger
from ..core.benchmarking import benchmark_operation, benchmark_context
from ..core.rag_logging import RAGSessionLogger, start_fresh_rag_session


class CodeEvaluationService:
    """Service for handling code evaluation logic."""
    
    def __init__(self):
        self.embedding_repository = EmbeddingRepository()
    
    @benchmark_operation("code_evaluation")
    async def evaluate_code(
        self,
        student_zip_path: str,
        ideal_zip_path: str,
        model: EmbeddingModel,
        use_openai_feedback: bool = False
    ) -> CodeEvaluationResult:
        """Evaluate student code against ideal solution."""
        
        # Start fresh RAG session with cleared logs
        start_fresh_rag_session()
        
        try:
            # Clear previous embeddings
            self.embedding_repository.clear_embeddings()
            logger.info("Cleared previous embeddings from database")
            
            # Extract functions from both ZIP files
            logger.info("Extracting functions from ZIP files")
            extraction_start = datetime.now()
            
            ideal_funcs = extract_functions_from_zip(ideal_zip_path)
            logger.info(f"Extracted {len(ideal_funcs)} functions from ideal solution")
            
            student_functions = extract_functions_from_zip(student_zip_path)
            logger.info(f"Extracted {len(student_functions)} functions from student submission")
            
            extraction_time = (datetime.now() - extraction_start).total_seconds()
            logger.info(f"Function extraction completed in {extraction_time:.2f}s")
            
            if not ideal_funcs:
                return CodeEvaluationResult(
                    status="error",
                    message="No Python functions found in ideal answer"
                )
            
            if not student_functions:
                return CodeEvaluationResult(
                    status="error",
                    message="No Python functions found in student submission"
                )
            
            # Generate embeddings for ideal functions and store them
            logger.info(f"Generating embeddings for {len(ideal_funcs)} ideal functions using {model}")
            ideal_embedding_start = datetime.now()
            ideal_embeddings = {}
        
            for name, code in ideal_funcs.items():
                logger.debug(f"Generating embedding for ideal function: {name}")
                embedding = get_embedding(code, ServiceEmbeddingModel(model))
                self.embedding_repository.store_embedding(name, code, embedding)
                ideal_embeddings[name] = embedding
            
            ideal_embedding_time = (datetime.now() - ideal_embedding_start).total_seconds()
            logger.info(f"Generated embeddings for ideal functions in {ideal_embedding_time:.2f}s")
        
            # Process student functions and generate reports
            logger.info("Starting function comparison and report generation")
            function_reports = {}
            total_similarity = 0
            func_count = len(ideal_funcs)
        
            comparison_start = datetime.now()
            
            # Log function match overview
            matching_funcs = set(student_functions.keys()).intersection(set(ideal_funcs.keys()))
            missing_funcs = set(ideal_funcs.keys()) - set(student_functions.keys())
            extra_funcs = set(student_functions.keys()) - set(ideal_funcs.keys())
            
            logger.info(f"Function match overview: {len(matching_funcs)} matching, {len(missing_funcs)} missing, {len(extra_funcs)} extra")
            if missing_funcs:
                logger.info(f"Missing functions: {', '.join(missing_funcs)}")
            if extra_funcs:
                logger.info(f"Extra functions in submission: {', '.join(extra_funcs)}")
        
            # Evaluate each function
            for func_name, ideal_code in ideal_funcs.items():
                func_start_time = datetime.now()
                logger.info(f"Evaluating function: {func_name}")
                
                if func_name in student_functions:
                    # Function exists in both - compare them
                    student_code = student_functions[func_name]
                    logger.debug(f"Function {func_name} found in both submissions - comparing")
                    
                    # Generate embedding for student code
                    logger.debug(f"Generating embedding for student function: {func_name}")
                    student_embedding_start = datetime.now()
                    student_embedding = get_embedding(student_code, ServiceEmbeddingModel(model))
                    student_embedding_time = (datetime.now() - student_embedding_start).total_seconds()
                    logger.debug(f"Generated student embedding in {student_embedding_time:.2f}s")
                    
                    # Calculate similarity
                    logger.debug(f"Calculating similarity for function: {func_name}")
                    similarity_start = datetime.now()
                    similarity = compute_similarity(student_embedding, ideal_embeddings[func_name])
                    similarity_time = (datetime.now() - similarity_start).total_seconds()
                    logger.info(f"Similarity for {func_name}: {similarity:.4f} (calculated in {similarity_time:.2f}s)")
                    
                    # Analyze code structure
                    logger.debug(f"Analyzing code structure for function: {func_name}")
                    structure_start = datetime.now()
                    structure_analysis = analyze_code_structure(student_code, ideal_code)
                    structure_time = (datetime.now() - structure_start).total_seconds()
                    logger.debug(f"Structure analysis completed in {structure_time:.2f}s")
                    
                    # Generate recommendations
                    logger.debug(f"Generating recommendations for function: {func_name}")
                    recommendations = generate_recommendations(structure_analysis)
                    
                    # Retrieve similar contexts
                    logger.debug(f"Retrieving similar contexts for function: {func_name}")
                    similar_contexts = self.embedding_repository.retrieve_similar_contexts(student_embedding)
                    if similar_contexts:
                        context_names = [ctx.get("function_name", "Unknown") for ctx in similar_contexts]
                        logger.debug(f"Found similar contexts: {', '.join(context_names)}")
                    
                    # Generate feedback
                    feedback = generate_feedback(student_code, ideal_code, similarity, structure_analysis, similar_contexts)
                    
                    # Create function report
                    logger.debug(f"Creating report for function: {func_name}")
                    
                    # Determine quality based on similarity score
                    if similarity >= 0.8:
                        quality = "high"
                    elif similarity >= 0.6:
                        quality = "medium"
                    elif similarity >= 0.4:
                        quality = "low"
                    else:
                        quality = "poor"
                    
                    function_reports[func_name] = {
                        "function_name": func_name,
                        "status": "Correct" if similarity >= settings.similarity_threshold else "Incorrect",
                        "similarity": float(similarity),
                        "quality": quality,
                        "structure_analysis": structure_analysis,
                        "recommendations": recommendations,
                        "similar_contexts": [
                            {
                                "function_name": ctx.get("function_name", "Unknown"),
                                "similarity": float(ctx.get("similarity", 0))
                            } for ctx in similar_contexts
                        ],
                        "feedback": feedback
                    }
                    
                    total_similarity += similarity
                else:
                    # Function in ideal but not in student submission
                    logger.info(f"Function {func_name} is missing from student submission")
                    function_reports[func_name] = {
                        "function_name": func_name,
                        "status": "Missing",
                        "similarity": 0.0,
                        "quality": "missing",
                        "structure_analysis": {
                            "variables": {"missing_variables": [], "extra_variables": []},
                            "control_flow": {"missing_control_structures": [], "extra_control_structures": []},
                            "function_calls": {"missing_calls": [], "extra_calls": []}
                        },
                        "recommendations": ["Implement this required function"],
                        "similar_contexts": [],
                        "feedback": "This function is missing from your submission."
                    }
                
                func_time = (datetime.now() - func_start_time).total_seconds()
                logger.info(f"Completed evaluation of function {func_name} in {func_time:.2f}s")
            
            # Calculate overall score
            overall_score = round((total_similarity / func_count * 100) if func_count > 0 else 0, 2)
            logger.info(f"Overall similarity score: {overall_score}%")
            
            comparison_time = (datetime.now() - comparison_start).total_seconds()
            logger.info(f"Completed function comparison in {comparison_time:.2f}s")
            
            # Get count of correctly matched functions (not missing)
            matched_count = len([f for f in function_reports if function_reports[f]["status"] != "Missing"])
            
            return CodeEvaluationResult(
                status="success",
                functions_evaluated=func_count,
                average_similarity=total_similarity / func_count if func_count > 0 else 0.0,
                function_results=list(function_reports.values()),
                extra_functions=list(extra_funcs),
                missing_functions=list(missing_funcs)
            )
            
        except Exception as e:
            logger.error(f"Code evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return CodeEvaluationResult(
                status="error",
                message=f"Evaluation failed: {str(e)}"
            )
