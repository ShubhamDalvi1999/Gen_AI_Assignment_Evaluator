from typing import List, Dict, Any
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
import os
import ast
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from utils.prompts import FEEDBACK_PROMPT_TEMPLATE
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGProcessor:
    def __init__(self, mongo_client: MongoClient):
        """Initialize the RAG processor with MongoDB connection."""
        self.db = mongo_client["assignment_checker"]
        self.collection = self.db["embeddings"]
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.use_openai = False
        
    def store_code_context(self, function_name: str, code: str, embedding: np.ndarray, 
                          metadata: Dict[str, Any] = None) -> None:
        """Store code context with its embedding and metadata."""
        logger.info("========== CODE CONTEXT STORAGE STAGE ==========")
        logger.info(f"Storing code context with embedding for function: {function_name}")
        
        document = {
            "function_name": function_name,
            "code": code,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        self.collection.insert_one(document)
        logger.info(f"Successfully stored code context for function: {function_name}")
        
    def retrieve_similar_contexts(self, query_embedding: np.ndarray, 
                                top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar code contexts based on embedding similarity."""
        # Get all stored embeddings
        stored_docs = list(self.collection.find({}, {"embedding": 1, "code": 1, "metadata": 1, "function_name": 1}))
        
        # Calculate similarities
        similarities = []
        for doc in stored_docs:
            similarity = 1 - cosine(query_embedding, np.array(doc["embedding"]))
            doc["similarity"] = similarity
            similarities.append((similarity, doc))
            
        # Sort by similarity and get top k
        similarities.sort(reverse=True)
        return [doc for _, doc in similarities[:top_k]]
    
    def analyze_code_structure(self, student_code: str, ideal_code: str) -> Dict[str, Any]:
        """Analyze and compare code structure between student and ideal solutions."""
        try:
            # Parse both code snippets
            student_tree = ast.parse(student_code)
            ideal_tree = ast.parse(ideal_code)
            
            # Extract variables from both
            student_vars = set()
            ideal_vars = set()
            for node in ast.walk(student_tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    student_vars.add(node.id)
            for node in ast.walk(ideal_tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    ideal_vars.add(node.id)
            
            # Extract control flow structures
            student_ctrl = []
            ideal_ctrl = []
            for node in ast.walk(student_tree):
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    student_ctrl.append(type(node).__name__)
            for node in ast.walk(ideal_tree):
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    ideal_ctrl.append(type(node).__name__)
            
            # Extract function calls
            student_calls = set()
            ideal_calls = set()
            for node in ast.walk(student_tree):
                if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                    student_calls.add(node.func.id)
            for node in ast.walk(ideal_tree):
                if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                    ideal_calls.add(node.func.id)
            
            # Compare and return differences
            return {
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
        except Exception as e:
            logger.error(f"Error analyzing code structure: {e}")
            return {
                "variables": {"missing_variables": [], "extra_variables": []},
                "control_flow": {"missing_control_structures": [], "extra_control_structures": []},
                "function_calls": {"missing_calls": [], "extra_calls": []}
            }
    
    def generate_recommendations(self, structure_analysis: Dict[str, Any]) -> List[str]:
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
    
    def generate_comparison_report(self, student_code: str, ideal_code: str, 
                                  student_embedding: np.ndarray, ideal_embedding: np.ndarray) -> Dict[str, Any]:
        """Generate a detailed comparison report using RAG."""
        # Retrieve similar contexts
        similar_contexts = self.retrieve_similar_contexts(student_embedding)
        
        # Calculate direct similarity
        direct_similarity = 1 - cosine(student_embedding, ideal_embedding)
        
        # Analyze code structure
        structure_analysis = self.analyze_code_structure(student_code, ideal_code)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(structure_analysis)
        
        # Return the compiled report
        return {
            "status": "Correct" if direct_similarity >= self.similarity_threshold else "Incorrect",
            "similarity": direct_similarity,
            "structure_analysis": structure_analysis,
            "recommendations": recommendations,
            "similar_contexts": similar_contexts,
            "feedback": "Detailed feedback not available in this simplified version."
        } 