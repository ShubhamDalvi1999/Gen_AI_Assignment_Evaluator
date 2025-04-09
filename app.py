import ast
import zipfile
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import requests
from scipy.spatial.distance import cosine
import logging
import tempfile
from enum import Enum
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from utils.tokenizer_utils import count_tokens, initialize_tokenizers
from utils.text_rag_processor import TextRAGProcessor
import traceback
import sys
import io
import re
import urllib.parse
import shutil
from fastapi.responses import JSONResponse
from utils.prompts import OPENAI_CODE_FEEDBACK_PROMPT

# Configure logging first before using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a custom logger with more detailed settings
logger = logging.getLogger("app")

# Set log level from environment variable if available, otherwise default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, log_level))
    logger.info(f"Log level set to {log_level}")
except AttributeError:
    logger.setLevel(logging.INFO)
    logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")

# Add a file handler to save logs to a file
try:
    log_file_path = os.path.join("logs", "app.log")
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - [%(funcName)s] - %(message)s'
    ))
    logger.addHandler(file_handler)
    logger.info(f"Log file created at: {log_file_path}")
except Exception as e:
    logger.warning(f"Failed to create log file: {e}")

# Log startup info
logger.info("=" * 60)
logger.info("AI Assignment Checker Starting")
logger.info("=" * 60)

# Load environment variables
load_dotenv()

# Log MongoDB connection details (removing passwords for security)
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    logger.error("MONGODB_URI not found in .env file. Remote MongoDB connection is required.")
    # Don't set default local connection - use None to indicate missing connection
    mongodb_uri = None
elif "://" in mongodb_uri:
    # Parse the URI but don't log it for security
    parts = mongodb_uri.split("://", 1)
    if "@" in parts[1]:
        user_pass, host_part = parts[1].split("@", 1)
        if ":" in user_pass:
            username = user_pass.split(":", 1)[0]
            # Create masked_uri but don't log it
            masked_uri = f"{parts[0]}://{username}:****@{host_part}"
        else:
            masked_uri = f"{parts[0]}://{user_pass}@{host_part}"
    else:
        masked_uri = mongodb_uri
    # Don't log the URI at all, even masked
    logger.info("MongoDB URI loaded from .env file")

# MongoDB configuration from environment variables
mongodb_db_name = os.getenv("MONGODB_DB_NAME", "assignment_checker")
mongodb_embeddings_collection = os.getenv("MONGODB_COLLECTION_NAME", "embeddings")
mongodb_qa_collection = os.getenv("MONGODB_QA_COLLECTION_NAME", "qa_embeddings")

# Initialize FastAPI app
app = FastAPI(title="AI Assignment Checker")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

# API URLs and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

# MongoDB setup
mongo_client = None
db = None
embeddings_collection = None

if mongodb_uri:
    try:
        # Use the URI from .env
        mongo_client = MongoClient(mongodb_uri)
        # Test connection
        mongo_client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB database: {mongodb_db_name}")
        
        # Initialize database and collections for easier access
        db = mongo_client[mongodb_db_name]
        embeddings_collection = db[mongodb_embeddings_collection]
        
        logger.info(f"MongoDB collections initialized: {mongodb_embeddings_collection}, {mongodb_qa_collection}")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        # Don't log the connection string, even masked
        mongo_client = None
        db = None
        embeddings_collection = None
else:
    logger.error("MongoDB connection not attempted. MONGODB_URI is missing from .env file.")

# Initialize tokenizers
logger.info("Pre-initializing tokenizers for improved performance...")
initialize_tokenizers()

# Initialize text RAG processor
try:
    if mongo_client:
        # Use the new constructor format with MongoDB URI
        from utils.text_rag_processor import TextRAGProcessor
        
        # Get MongoDB URI from environment
        mongodb_uri = os.getenv("MONGODB_URI")
        
        # Create TextRAGProcessor with URI and database name
        text_rag_processor = TextRAGProcessor(
            mongodb_uri=mongodb_uri,
            use_openai=os.getenv("USE_OPENAI", "False").lower() == "true",
            db_name=mongodb_db_name
        )
        logger.info("TextRAGProcessor initialized successfully with new constructor")
    else:
        text_rag_processor = None
        logger.warning("TextRAGProcessor not initialized due to MongoDB connection failure")
except Exception as e:
    logger.error(f"Error initializing TextRAGProcessor: {e}")
    text_rag_processor = None

def extract_functions_from_file(file_path: str) -> Dict[str, str]:
    """Extract function definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        tree = ast.parse(code)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        return {func.name: ast.get_source_segment(code, func) for func in functions}
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return {}

def extract_functions_from_zip(zip_path: str) -> Dict[str, str]:
    """Extract all functions from Python files in a zip."""
    logger.info(f"Starting function extraction from ZIP: {os.path.basename(zip_path)}")
    func_codes = {}
    with tempfile.TemporaryDirectory() as extract_dir:
        try:
            # Validate zip file before extracting
            if not zipfile.is_zipfile(zip_path):
                logger.error(f"Invalid ZIP file: {zip_path}")
                raise ValueError(f"The provided file is not a valid ZIP archive.")
                
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract with careful error handling
                try:
                    logger.info(f"Extracting ZIP file to temporary directory: {extract_dir}")
                    zip_ref.extractall(extract_dir)
                    logger.info(f"Successfully extracted {len(zip_ref.namelist())} files from ZIP")
                except zipfile.BadZipFile:
                    logger.error(f"Bad ZIP file format: {zip_path}")
                    raise ValueError(f"The ZIP file is corrupt or in an unsupported format.")
                except PermissionError:
                    logger.error(f"Permission denied when extracting: {zip_path}")
                    raise ValueError(f"Permission denied when extracting the ZIP file.")
                except Exception as extract_error:
                    logger.error(f"Error extracting ZIP: {extract_error}")
                    raise ValueError(f"Failed to extract the ZIP file: {str(extract_error)}")
            
            # Check if any Python files were found
            python_files_found = False
            python_files_count = 0
            
            logger.info(f"Scanning directory for Python files: {extract_dir}")
            for root, dirs, files in os.walk(extract_dir):
                logger.debug(f"Scanning directory: {root} (contains {len(files)} files)")
                for file in files:
                    if file.endswith('.py'):
                        python_files_count += 1
                        python_files_found = True
                        file_path = os.path.join(root, file)
                        logger.info(f"Processing Python file: {os.path.relpath(file_path, extract_dir)}")
                        try:
                            file_functions = extract_functions_from_file(file_path)
                            if file_functions:
                                logger.info(f"Found {len(file_functions)} functions in {os.path.relpath(file_path, extract_dir)}")
                                for func_name in file_functions:
                                    logger.debug(f"  - Function: {func_name}")
                            else:
                                logger.info(f"No functions found in {os.path.relpath(file_path, extract_dir)}")
                            func_codes.update(file_functions)
                        except Exception as file_error:
                            logger.warning(f"Skipping file {file_path}: {file_error}")
            
            logger.info(f"Found {python_files_count} Python files with {len(func_codes)} total functions")
            
            if not python_files_found:
                logger.warning(f"No Python files found in ZIP: {zip_path}")
                raise ValueError(f"No Python (.py) files found in the ZIP archive. Please check the contents of your submission.")
                
        except ValueError as value_error:
            # Re-raise ValueError with the message intact
            raise value_error
        except Exception as e:
            logger.error(f"Unexpected error processing ZIP file {zip_path}: {e}")
            raise ValueError(f"Error processing ZIP file: {str(e)}")
    
    if not func_codes:
        logger.warning(f"No functions found in any Python files from ZIP: {zip_path}")
        raise ValueError(f"No Python functions were found in the submitted files. Please ensure your code contains function definitions.")
    
    logger.info(f"Successfully extracted {len(func_codes)} functions: {', '.join(func_codes.keys())}")    
    return func_codes

def get_embedding_ollama(text: str) -> np.ndarray:
    """Generate embedding using Ollama's local model."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "llama3.2:3b", "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        if "embedding" not in result:
            raise ValueError("No embedding in Ollama response")
        return np.array(result["embedding"])
    except Exception as e:
        logger.error(f"Ollama embedding generation failed: {e}")
        raise ValueError(f"Ollama embedding generation failed: {str(e)}")

def get_embedding_openai(text: str) -> np.ndarray:
    """Generate embedding using OpenAI's API."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured")
    
    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-ada-002",
                "input": text
            },
            timeout=10
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])
    except Exception as e:
        logger.error(f"OpenAI embedding generation failed: {e}")
        raise ValueError(f"OpenAI embedding generation failed: {str(e)}")

def get_embedding(text: str, model: EmbeddingModel = EmbeddingModel.OLLAMA) -> np.ndarray:
    """Generate embedding based on selected model."""
    logger.debug(f"Generating embedding using model: {model}")
    start_time = datetime.now()
    
    try:
        if model == EmbeddingModel.OLLAMA:
            embedding = get_embedding_ollama(text)
        else:
            embedding = get_embedding_openai(text)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Generated embedding successfully in {elapsed:.2f}s ({len(text)} chars, model: {model})")
        return embedding
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Embedding generation failed after {elapsed:.2f}s: {e}")
        raise

def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    try:
        return float(1 - cosine(emb1, emb2))
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def analyze_code_structure(student_code: str, ideal_code: str) -> Dict[str, Any]:
    """Analyze and compare code structure between student and ideal solutions."""
    logger.info("Analyzing and comparing code structure")
    start_time = datetime.now()
    
    try:
        # Parse both code snippets
        logger.debug("Parsing student and ideal code with ast")
        student_tree = ast.parse(student_code)
        ideal_tree = ast.parse(ideal_code)
        
        # Extract variables from both
        logger.debug("Extracting variables")
        student_vars = set()
        ideal_vars = set()
        for node in ast.walk(student_tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                student_vars.add(node.id)
        for node in ast.walk(ideal_tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                ideal_vars.add(node.id)
        
        # Extract control flow structures
        logger.debug("Extracting control flow structures")
        student_ctrl = []
        ideal_ctrl = []
        for node in ast.walk(student_tree):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                student_ctrl.append(type(node).__name__)
        for node in ast.walk(ideal_tree):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                ideal_ctrl.append(type(node).__name__)
        
        # Extract function calls
        logger.debug("Extracting function calls")
        student_calls = set()
        ideal_calls = set()
        for node in ast.walk(student_tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                student_calls.add(node.func.id)
        for node in ast.walk(ideal_tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                ideal_calls.add(node.func.id)
        
        # Compare and return differences
        result = {
            "variables": {
                "missing_variables": list(ideal_vars - student_vars),
                "extra_variables": list(student_vars - ideal_vars)
            },
            "control_flow": {
                "missing_control_structures": [c for c in ideal_ctrl if c not in student_ctrl],
                "extra_control_structures": [c for c in student_ctrl if c not in ideal_ctrl]
            },
            "function_calls": {
                "missing_calls": list(ideal_calls - student_calls),
                "extra_calls": list(student_calls - ideal_calls)
            }
        }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Code structure analysis completed in {elapsed:.2f}s")
        logger.debug(f"Analysis results: {json.dumps(result, indent=2)}")
        
        return result
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error analyzing code structure after {elapsed:.2f}s: {e}")
        return {
            "variables": {"missing_variables": [], "extra_variables": []},
            "control_flow": {"missing_control_structures": [], "extra_control_structures": []},
            "function_calls": {"missing_calls": [], "extra_calls": []}
        }

def generate_recommendations(structure_analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on structure analysis."""
    recommendations = []
    
    # Variable recommendations
    missing_vars = structure_analysis["variables"]["missing_variables"]
    if missing_vars:
        recommendations.append(f"Add missing variables: {', '.join(missing_vars)}")
    
    extra_vars = structure_analysis["variables"]["extra_variables"]
    if extra_vars:
        recommendations.append(f"Consider removing unnecessary variables: {', '.join(extra_vars)}")
    
    # Control flow recommendations
    missing_ctrl = structure_analysis["control_flow"]["missing_control_structures"]
    if missing_ctrl:
        recommendations.append(f"Add missing control structures: {', '.join(missing_ctrl)}")
    
    # Function call recommendations
    missing_calls = structure_analysis["function_calls"]["missing_calls"]
    if missing_calls:
        recommendations.append(f"Add missing function calls: {', '.join(missing_calls)}")
    
    if not recommendations:
        recommendations.append("Your implementation looks good structurally!")
    
    return recommendations

def store_embedding(func_name: str, code: str, embedding: np.ndarray) -> None:
    """Store embedding in MongoDB."""
    try:
        if mongo_client is not None and embeddings_collection is not None:
            embeddings_collection.insert_one({
                "function_name": func_name,
                "code": code,
                "embedding": embedding.tolist(),
                "timestamp": datetime.now()
            })
            logger.debug(f"Embedding stored for function: {func_name}")
        else:
            logger.warning(f"Embedding not stored for function {func_name} - MongoDB connection unavailable")
    except Exception as e:
        logger.error(f"Error storing embedding: {e}")

def retrieve_similar_contexts(query_embedding: np.ndarray, top_k: int = 9) -> List[Dict[str, Any]]:
    """Retrieve similar code contexts based on embedding similarity."""
    logger.info(f"Retrieving similar contexts (top_{top_k})")
    start_time = datetime.now()
    
    try:
        if mongo_client is not None and embeddings_collection is not None:
            # Get all stored embeddings from MongoDB
            logger.debug("Querying MongoDB for stored embeddings")
            stored_docs = list(embeddings_collection.find({}, {"embedding": 1, "code": 1, "function_name": 1}))
            logger.debug(f"Retrieved {len(stored_docs)} embeddings from MongoDB for similarity comparison")
            
            if not stored_docs:
                logger.warning("No stored embeddings found in MongoDB")
                return []
            
            # Calculate similarities
            logger.debug("Calculating similarity scores between query and stored embeddings")
            calc_start = datetime.now()
            similarities = []
            for doc in stored_docs:
                similarity = compute_similarity(query_embedding, np.array(doc["embedding"]))
                doc["similarity"] = similarity
                similarities.append((similarity, doc))
            
            calc_time = (datetime.now() - calc_start).total_seconds()
            logger.debug(f"Calculated {len(similarities)} similarity scores in {calc_time:.2f}s")
                
            # Sort by similarity and get top k
            similarities.sort(reverse=True, key=lambda x: x[0])
            result = [doc for _, doc in similarities[:top_k]]
            
            # Log result summary
            if result:
                top_results = [(doc.get("function_name", "Unknown"), doc.get("similarity", 0)) for doc in result]
                result_summary = ", ".join([f"{name}({sim:.4f})" for name, sim in top_results])
                logger.info(f"Top {len(result)} similar contexts: {result_summary}")
            else:
                logger.warning("No similar contexts found after filtering")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Similar contexts retrieval completed in {elapsed:.2f}s")
            return result
        else:
            logger.warning("MongoDB connection unavailable - cannot retrieve embeddings")
            return []
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error retrieving similar contexts after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return []

def generate_feedback(student_code: str, ideal_code: str, similarity: float, 
                    structure_analysis: Dict[str, Any], similar_contexts: List[Dict[str, Any]]) -> str:
    """Generate feedback using Ollama."""
    logger.info(f"Generating feedback with Ollama (similarity score: {similarity:.4f})")
    start_time = datetime.now()
    
    try:
        # Format the prompt
        logger.debug("Formatting prompt for feedback generation")
        prompt = f"""
        You are an AI code reviewer evaluating a student's function implementation against an ideal solution. 
        Your task is to provide constructive feedback on the correctness and quality of the student's code.

        ### Ideal Code:
        ```python
        {ideal_code}
        ```

        ### Student Code:
        ```python
        {student_code}
        ```

        ### Similarity Score: {similarity:.2f}

        ### Structure Analysis:
        {json.dumps(structure_analysis, indent=2)}

        Based on this information, provide 2-3 paragraphs of constructive feedback for the student, 
        highlighting strengths and areas for improvement.
        """
        
        # Call Ollama API
        logger.debug(f"Calling Ollama API at {OLLAMA_BASE_URL}")
        api_start = datetime.now()
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": "llama3.2:3b",
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
        logger.debug(f"Ollama API response received in {api_time:.2f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                feedback = result["message"]["content"]
                feedback_length = len(feedback)
                logger.info(f"Generated feedback successfully ({feedback_length} chars)")
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                return feedback
            elif "response" in result:
                feedback = result["response"]
                feedback_length = len(feedback)
                logger.info(f"Generated feedback successfully ({feedback_length} chars)")
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Feedback generation completed in {elapsed:.2f}s")
                return feedback
            else:
                logger.error(f"Unexpected response format from Ollama: {result}")
                return "Error generating feedback."
        else:
            error_msg = f"Error generating feedback: {response.status_code}"
            logger.error(f"Ollama API error: {response.status_code}, {response.text}")
            return error_msg
            
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error generating feedback after {elapsed:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating feedback: {str(e)}"

@app.post("/evaluate")
async def evaluate(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA),
    use_openai_feedback: bool = Form(False)
) -> Dict[str, Any]:
    """Handle file uploads and perform evaluation."""
    logger.info(f"Starting code evaluation - Student: {submission.filename}, Ideal: {ideal.filename}, Model: {model}")
    logger.info(f"OpenAI feedback enabled: {use_openai_feedback}")
    
    evaluation_start_time = datetime.now()
    
    # Check if MongoDB is available - required for code evaluation
    if mongo_client is None:
        logger.error("MongoDB connection unavailable - required for evaluation")
        return {
            "status": "error",
            "message": "MongoDB connection is required for code evaluation. Please check your MongoDB URI in the .env file."
        }
    
    # Validate file extensions
    for upload_file, file_name in [(submission, "Student submission"), (ideal, "Ideal solution")]:
        if not upload_file.filename.lower().endswith('.zip'):
            logger.error(f"{file_name} has invalid extension: {upload_file.filename}")
            return {
                "status": "error",
                "message": f"{file_name} must be a ZIP file. Please upload a file with .zip extension."
            }
        
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
            return {
                "status": "error",
                "message": f"Error reading {file_name}: {str(e)}"
            }
            
        if file_size > 10 * 1024 * 1024:  # 10MB in bytes
            logger.error(f"{file_name} exceeds size limit: {file_size / (1024 * 1024):.1f} MB > 10 MB")
            return {
                "status": "error",
                "message": f"{file_name} exceeds the 10MB size limit. Please upload a smaller file."
            }
            
        # Basic ZIP validation check
        try:
            # Read a small portion to check if it's a valid ZIP
            signature = file_content[:4]
            # ZIP files start with PK\x03\x04 signature
            if signature != b'PK\x03\x04':
                logger.error(f"{file_name} has invalid ZIP signature: {signature}")
                return {
                    "status": "error", 
                    "message": f"{file_name} does not appear to be a valid ZIP file. Please check the file format."
                }
        except Exception as e:
            logger.error(f"Error validating {file_name}: {e}")
            return {
                "status": "error",
                "message": f"Error validating {file_name}: {str(e)}"
            }
    
    # Initialize file paths before try block
    student_zip_path = None
    ideal_path = None
    
    # Create temporary files to extract the uploads
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for evaluation: {temp_dir}")
    
    try:
        # Clear previous embeddings
        if embeddings_collection is not None:
            logger.info("Clearing previous embeddings from database")
            delete_result = embeddings_collection.delete_many({})
            logger.info(f"Deleted {delete_result.deleted_count} previous embeddings")
        
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

        # Extract functions from both ZIP files
        logger.info("Extracting functions from ZIP files")
        extraction_start = datetime.now()
        
        try:
            ideal_funcs = extract_functions_from_zip(ideal_path)
            logger.info(f"Extracted {len(ideal_funcs)} functions from ideal solution")
            
            student_functions = extract_functions_from_zip(student_zip_path)
            logger.info(f"Extracted {len(student_functions)} functions from student submission")
            
            extraction_time = (datetime.now() - extraction_start).total_seconds()
            logger.info(f"Function extraction completed in {extraction_time:.2f}s")
        except Exception as extraction_error:
            logger.error(f"Function extraction failed: {extraction_error}")
            raise

        if not ideal_funcs:
            logger.error("No Python functions found in ideal answer")
            return {"error": "No Python functions found in ideal answer"}
        if not student_functions:
            logger.error("No Python functions found in student submission")
            return {"error": "No Python functions found in submission"}

        # Generate embeddings for ideal functions and store them
        logger.info(f"Generating embeddings for {len(ideal_funcs)} ideal functions using {model}")
        ideal_embedding_start = datetime.now()
        ideal_embeddings = {}
        for name, code in ideal_funcs.items():
            logger.debug(f"Generating embedding for ideal function: {name}")
            embedding = get_embedding(code, model)
            store_embedding(name, code, embedding)
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
                student_embedding = get_embedding(student_code, model)
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
                similar_contexts = retrieve_similar_contexts(student_embedding)
                if similar_contexts:
                    context_names = [ctx.get("function_name", "Unknown") for ctx in similar_contexts]
                    logger.debug(f"Found similar contexts: {', '.join(context_names)}")
                
                # Generate feedback if enabled
                feedback = ""
                feedback_start = datetime.now()
                
                if use_openai_feedback and OPENAI_API_KEY:
                    # Use OpenAI for better feedback
                    logger.info(f"Generating OpenAI feedback for function: {func_name}")
                    try:
                        # Format the prompt with the specific data
                        prompt = OPENAI_CODE_FEEDBACK_PROMPT.format(
                            student_code=student_code,
                            ideal_code=ideal_code,
                            missing_variables=', '.join(structure_analysis['variables']['missing_variables']) or 'None',
                            extra_variables=', '.join(structure_analysis['variables']['extra_variables']) or 'None',
                            missing_control_structures=', '.join(structure_analysis['control_flow']['missing_control_structures']) or 'None',
                            extra_control_structures=', '.join(structure_analysis['control_flow']['extra_control_structures']) or 'None',
                            missing_function_calls=', '.join(structure_analysis['function_calls']['missing_calls']) or 'None',
                            extra_function_calls=', '.join(structure_analysis['function_calls']['extra_calls']) or 'None',
                            similarity=f"{similarity:.2f}"
                        )
                        
                        # Call OpenAI API for feedback
                        logger.debug(f"Calling OpenAI API for feedback on function: {func_name}")
                        openai_start = datetime.now()
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENAI_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "gpt-3.5-turbo",
                                "messages": [
                                    {"role": "system", "content": "You are an educational AI assistant that provides helpful, detailed feedback on student code submissions."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": 0.3,
                                "max_tokens": 800
                            },
                            timeout=30
                        )
                        openai_time = (datetime.now() - openai_start).total_seconds()
                        logger.debug(f"OpenAI API call completed in {openai_time:.2f}s with status: {response.status_code}")
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            if "choices" in response_data and len(response_data["choices"]) > 0:
                                feedback = response_data["choices"][0]["message"]["content"].strip()
                                feedback_length = len(feedback)
                                logger.info(f"Generated OpenAI feedback for function: {func_name} ({feedback_length} chars)")
                            else:
                                logger.error(f"Unexpected response format from OpenAI: {response_data}")
                                logger.info(f"Falling back to local feedback for function: {func_name}")
                                feedback = generate_feedback(student_code, ideal_code, similarity, 
                                                         structure_analysis, similar_contexts)
                        else:
                            logger.error(f"OpenAI API error: {response.status_code}, {response.text}")
                            logger.info(f"Falling back to local feedback for function: {func_name}")
                            feedback = generate_feedback(student_code, ideal_code, similarity, 
                                                     structure_analysis, similar_contexts)
                    except Exception as e:
                        logger.error(f"Error generating OpenAI feedback: {e}")
                        # Fall back to local feedback
                        logger.info(f"Falling back to local feedback for function: {func_name}")
                        feedback = generate_feedback(student_code, ideal_code, similarity, 
                                                  structure_analysis, similar_contexts)
                else:
                    # Use Ollama for feedback
                    logger.info(f"Generating local feedback for function: {func_name}")
                    feedback = generate_feedback(student_code, ideal_code, similarity, 
                                              structure_analysis, similar_contexts)
                
                feedback_time = (datetime.now() - feedback_start).total_seconds()
                logger.debug(f"Feedback generation completed in {feedback_time:.2f}s")
                
                # Create function report
                logger.debug(f"Creating report for function: {func_name}")
                function_reports[func_name] = {
                    "status": "Correct" if similarity >= SIMILARITY_THRESHOLD else "Incorrect",
                    "similarity": float(similarity),
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
                    "status": "Missing",
                    "similarity": 0.0,
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
        
        total_evaluation_time = (datetime.now() - evaluation_start_time).total_seconds()
        logger.info(f"Completed full evaluation in {total_evaluation_time:.2f}s")

        # Get count of correctly matched functions (not missing)
        matched_count = len([f for f in function_reports if function_reports[f]["status"] != "Missing"])
        
        # Create the response with more explicit structure
        return {
            "report": function_reports,
            "overall_score": f"{overall_score}%",
            "total_functions": func_count,
            "matched_functions": matched_count,
            "function_stats": {
                "total": func_count,
                "matched": matched_count,
                "missing": func_count - matched_count
            },
            "model_used": model
        }

    except Exception as e:
        error_time = (datetime.now() - evaluation_start_time).total_seconds()
        logger.error(f"Evaluation failed after {error_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Evaluation failed: {str(e)}"}
    
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

@app.post("/evaluate-text")
async def evaluate_text(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA)
) -> Dict[str, Any]:
    """Handle DOCX file uploads and perform Q&A evaluation."""
    
    # Check if MongoDB is available - required for text evaluation
    if mongo_client is None or text_rag_processor is None:
        return {
            "status": "error",
            "message": "MongoDB connection is required for text evaluation. Please check your MongoDB URI in the .env file."
        }
    
    # Create temporary files
    submission_path = None
    ideal_path = None
    
    try:
        # Process submission file
        submission_path = os.path.join(tempfile.gettempdir(), f"submission_{submission.filename}")
        with open(submission_path, "wb") as temp_file:
            temp_file.write(await submission.read())
        
        # Process ideal file
        ideal_path = os.path.join(tempfile.gettempdir(), f"ideal_{ideal.filename}")
        with open(ideal_path, "wb") as temp_file:
            temp_file.write(await ideal.read())
        
        # Log file sizes for debugging
        submission_size = os.path.getsize(submission_path)
        ideal_size = os.path.getsize(ideal_path)
        logger.info(f"Evaluating text files - submission: {submission.filename} ({submission_size} bytes), ideal: {ideal.filename} ({ideal_size} bytes)")
        
        # Set model based on the dropdown selection only
        if model == EmbeddingModel.OPENAI:
            # When OpenAI is selected as the model, ensure we use OpenAI
            text_rag_processor.use_openai = True
            logger.info("Using OpenAI for text evaluation (selected from model dropdown)")
            
            # Verify OpenAI API key is available - safely access the attribute
            openai_api_key = getattr(text_rag_processor, 'openai_api_key', None)
            if not openai_api_key:
                return {
                    "status": "error",
                    "message": "OpenAI API key is required when using OpenAI model. Please check your .env file."
                }
        else:  # Ollama is the default
            text_rag_processor.use_openai = False
            logger.info("Using Ollama for text evaluation (selected from model dropdown)")
        
        # Log final model selection
        logger.info(f"Final model selection: {'OpenAI' if text_rag_processor.use_openai else 'Ollama'}")
        
        # Perform the evaluation
        logger.info("Starting Q&A evaluation process")
        evaluation_result = text_rag_processor.evaluate_qa_submission(submission_path, ideal_path)
        
        # Ensure the overall_score appears at top level as some UI components expect it
        if evaluation_result.get("status") == "success" and "overall_score" not in evaluation_result:
            if "result" in evaluation_result and "overall_score" in evaluation_result["result"]:
                evaluation_result["overall_score"] = evaluation_result["result"]["overall_score"]
                logger.debug("Added overall_score at top level for UI compatibility")
        
        # Log evaluation completion
        if evaluation_result.get("status") == "success":
            score = evaluation_result.get("overall_score", evaluation_result.get("result", {}).get("overall_score", 0))
            logger.info(f"Q&A evaluation completed successfully with score: {score}%")
        else:
            logger.warning(f"Q&A evaluation failed: {evaluation_result.get('message', 'Unknown error')}")
            
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Error in text evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    finally:
        # Clean up temporary files
        try:
            for path in [submission_path, ideal_path]:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up temporary files: {cleanup_error}")

@app.post("/estimate-tokens")
async def estimate_tokens(submission: UploadFile = File(...), ideal: UploadFile = File(...)):
    """Estimate token usage for uploaded files."""
    
    submission_path = None
    ideal_path = None
    
    try:
        logger.info(f"Estimating tokens for files: {submission.filename} and {ideal.filename}")
        
        # Determine if we're in text or code mode based on file extension
        is_text_file = any(ext in submission.filename.lower() for ext in ['.docx', '.doc', '.txt']) or \
                      any(ext in ideal.filename.lower() for ext in ['.docx', '.doc', '.txt'])
        
        # Forced text mode if files have .docx extension - this is crucial for the text evaluation tab
        if submission.filename.lower().endswith('.docx') or ideal.filename.lower().endswith('.docx'):
            is_text_file = True
            logger.info("Forced text mode due to .docx extension")
        
        # Create temp directory if it doesn't exist
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files
        submission_path = os.path.join(temp_dir, os.path.basename(submission.filename))
        ideal_path = os.path.join(temp_dir, os.path.basename(ideal.filename))
        
        with open(submission_path, "wb") as f:
            submission_bytes = await submission.read()
            if len(submission_bytes) == 0:
                return {
                    "status": "error",
                    "message": "Uploaded student submission file is empty."
                }
            f.write(submission_bytes)
            logger.info(f"Saved submission file: {submission_path} ({len(submission_bytes)} bytes)")
        
        with open(ideal_path, "wb") as f:
            ideal_bytes = await ideal.read()
            if len(ideal_bytes) == 0:
                return {
                    "status": "error",
                    "message": "Uploaded ideal answer file is empty."
                }
            f.write(ideal_bytes)
            logger.info(f"Saved ideal file: {ideal_path} ({len(ideal_bytes)} bytes)")
        
        # Initialize token counts
        student_tokens = 0
        ideal_tokens = 0
        
        # Process as text files if that's what we determined earlier
        if is_text_file:
            logger.info("Processing as text files")
            try:
                # Import in the try block to handle import errors gracefully
                from utils.docx_processor import DocxProcessor
                
                # Extract text from files
                logger.debug(f"Extracting text from submission: {submission_path}")
                student_text = DocxProcessor.extract_text_from_docx(submission_path)
                logger.debug(f"Extracting text from ideal: {ideal_path}")
                ideal_text = DocxProcessor.extract_text_from_docx(ideal_path)
                
                # Log text lengths to help diagnose issues
                logger.debug(f"Student text length: {len(student_text) if student_text else 0} characters")
                logger.debug(f"Ideal text length: {len(ideal_text) if ideal_text else 0} characters")
                
                if not student_text:
                    logger.warning(f"No text extracted from submission file: {submission_path}")
                    student_text = ""
                    
                if not ideal_text:
                    logger.warning(f"No text extracted from ideal file: {ideal_path}")
                    ideal_text = ""
                
                # Both texts are empty - this might indicate a problem with text extraction
                if not student_text and not ideal_text:
                    # If these are definitely DOCX files but we got no text, report error
                    if submission.filename.lower().endswith('.docx') or ideal.filename.lower().endswith('.docx'):
                        logger.error("No text extracted from DOCX files - possible corrupt files")
                        return {
                            "status": "error",
                            "message": "No text could be extracted from the DOCX files. The files may be corrupt or empty."
                        }
                    else:
                        # Try code processing instead if these might be code files
                        logger.warning("No text extracted from both files, switching to code processing")
                        raise ValueError("Empty text extracted from files")
                
                # Count tokens for the extracted text
                logger.debug("Counting tokens for text files")
                student_tokens = count_tokens(student_text)
                ideal_tokens = count_tokens(ideal_text)
                logger.info(f"Text token counts: student={student_tokens}, ideal={ideal_tokens}")
                
                # Calculate overhead for text evaluation
                comparison_overhead = int((student_tokens + ideal_tokens) * 0.3)
                prompt_overhead = 1000  # Text evaluation prompts are slightly larger
                qa_extraction_overhead = int(max(student_tokens, ideal_tokens) * 0.5)  # Extracting Q&A pairs
                
                total_estimate = student_tokens + ideal_tokens + comparison_overhead + prompt_overhead + qa_extraction_overhead
                safety_margin = int(total_estimate * 0.1)
                total_estimate += safety_margin
                
                is_safe = total_estimate < 16000
                
                return {
                    "status": "success",
                    "file_type": "text",
                    "student_tokens": student_tokens,
                    "ideal_tokens": ideal_tokens,
                    "comparison_overhead": comparison_overhead,
                    "qa_extraction_overhead": qa_extraction_overhead,
                    "prompt_overhead": prompt_overhead,
                    "safety_margin": safety_margin,
                    "total_estimate": total_estimate,
                    "is_safe": is_safe
                }
            except Exception as docx_error:
                # If this is explicitly a DOCX file but processing failed, return an error
                if submission.filename.lower().endswith('.docx') or ideal.filename.lower().endswith('.docx'):
                    logger.error(f"DOCX processing failed for explicit DOCX files: {docx_error}")
                    logger.error(traceback.format_exc())
                    return {
                        "status": "error",
                        "message": f"Failed to process DOCX files: {str(docx_error)}"
                    }
                else:
                    logger.warning(f"Text processing failed: {docx_error}, trying code processing")
        
        # Only try code processing if not explicit DOCX files
        if not (submission.filename.lower().endswith('.docx') or ideal.filename.lower().endswith('.docx')):
            try:
                logger.info("Trying code file processing")
                # Extract student code from submission
                student_functions = extract_functions_from_zip(submission_path)
                ideal_functions = extract_functions_from_zip(ideal_path)
                
                logger.info(f"Extracted functions: student={len(student_functions)}, ideal={len(ideal_functions)}")
                
                # Count tokens in each function
                for func_name, func_code in student_functions.items():
                    student_tokens += count_tokens(func_code)
                
                for func_name, func_code in ideal_functions.items():
                    ideal_tokens += count_tokens(func_code)
                
                logger.info(f"Code token counts: student={student_tokens}, ideal={ideal_tokens}")
                
                # Calculate overhead estimates
                # 1. Structure analysis overhead (comparing the code)
                structure_overhead = int((student_tokens + ideal_tokens) * 0.2)
                
                # 2. RAG retrieval overhead (embedding searches, etc.)
                retrieval_overhead = int(ideal_tokens * 0.3)
                
                # 3. Prompt overhead (prompts for analysis, recommendations, etc.)
                prompt_overhead = 800  # Fixed token count for prompts
                
                # Calculate total estimate with safety margin
                total_estimate = student_tokens + ideal_tokens + structure_overhead + retrieval_overhead + prompt_overhead
                safety_margin = int(total_estimate * 0.1)
                total_estimate += safety_margin
                
                # Check if it's safe to process (under token limit)
                is_safe = total_estimate < 16000  # Assuming a limit of 16K tokens for most operations
                
                return {
                    "status": "success",
                    "file_type": "code",
                    "student_tokens": student_tokens,
                    "ideal_tokens": ideal_tokens,
                    "structure_overhead": structure_overhead,
                    "retrieval_overhead": retrieval_overhead,
                    "prompt_overhead": prompt_overhead,
                    "safety_margin": safety_margin,
                    "total_estimate": total_estimate,
                    "is_safe": is_safe
                }
            except Exception as zip_error:
                logger.error(f"ZIP processing failed: {zip_error}")
                logger.error(traceback.format_exc())
                
                return {
                    "status": "error",
                    "message": f"Could not estimate tokens: Unable to process the provided files. Please ensure you're uploading valid files.",
                    "details": str(zip_error)
                }
        else:
            # This case should never happen since we would have returned from the DOCX section
            # But adding as a fallback
            return {
                "status": "error",
                "message": "DOCX files detected but processing failed. Please try again."
            }
    except Exception as e:
        logger.error(f"Error estimating tokens: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error estimating tokens: {str(e)}"
        }
    finally:
        # Cleanup temporary files
        for path in [submission_path, ideal_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Deleted temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Error cleaning up file {path}: {e}")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/process-qa")
async def process_qa():
    """Process Q&A evaluation request."""
    try:
        # Parse the JSON data from the request
        data = await request.json()
        
        if not data:
            return JSONResponse(
                status_code=400,
                content={"error": "No data provided"}
            )
            
        # Extract data fields
        qa_pairs = data.get('qa_pairs', [])
        document_id = data.get('document_id')
        course_context = data.get('course_context', '')
        emphasize_embedding = data.get('emphasize_embedding', False)
        
        if not qa_pairs:
            return JSONResponse(
                status_code=400,
                content={"error": "No Q&A pairs provided"}
            )
            
        if not document_id:
            return JSONResponse(
                status_code=400,
                content={"error": "No document ID provided"}
            )
        
        # Log the request
        logger.info(f"Processing {len(qa_pairs)} Q&A pairs for document {document_id}")
        if emphasize_embedding:
            logger.info("Using embedding-focused similarity for evaluation")
        
        # Check if text_rag_processor is available
        if text_rag_processor is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Text RAG processor is not initialized. Check MongoDB connection."}
            )
        
        # Format Q&A pairs for processing
        formatted_qa = {}
        for i, qa in enumerate(qa_pairs):
            formatted_qa[f"qa_{i}"] = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", "")
            }
        
        # Process the Q&A pairs
        result = text_rag_processor.process_qa_submission(
            document_id=document_id,
            qa_pairs=formatted_qa,
            emphasize_embedding=emphasize_embedding
        )

        return result
    except Exception as e:
        logger.error(f"Error in process_qa: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Entry point to run the application
if __name__ == "__main__":
    import uvicorn
    import platform
    
    # Print startup message
    print("\n" + "=" * 80)
    print("AI Assignment Checker")
    print("=" * 80)
    
    # Custom host and port
    host = "127.0.0.1"  # Using 127.0.0.1 instead of 0.0.0.0 for better Windows compatibility
    port = 8001
    
    # OS-specific instructions
    if platform.system() == "Windows":
        print(f"\nServer starting on http://{host}:{port}")
        print(f"Open your browser and navigate to: http://localhost:{port}")
    else:
        print(f"\nServer starting on http://{host}:{port}")
        print(f"You can access the application at: http://localhost:{port}")
    
    print("\nPress CTRL+C to stop the server")
    print("=" * 80 + "\n")
    
    # Start the server
    uvicorn.run(app, host=host, port=port) 