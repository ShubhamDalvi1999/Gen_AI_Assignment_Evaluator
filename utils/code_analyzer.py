import os
import numpy as np
import ast
import sys
import zipfile
import tempfile
from scipy.spatial.distance import cosine
from enum import Enum
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

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
class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

# API URLs and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))




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
