"""
Text-based Retrieval-Augmented Generation (RAG) processor for evaluating Q&A submissions.
"""
import os
import json
import difflib
import logging
import traceback
from collections import Counter
import numpy as np
import requests
import random
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from utils.docx_processor import DocxProcessor
from utils.tokenizer_utils import count_tokens, safe_truncate_code, _get_tokenizer
from utils.prompts import QA_SUMMARY_PROMPT, QA_EVALUATION_PROMPT,QA_EXTRACTION_PROMPT
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for question embeddings to reduce API calls
_QUESTION_EMBEDDING_CACHE = {}

# Method to create consistent cache keys
def _create_cache_key(text, prefix=""):
    """Create a more reliable cache key than using hash()."""
    import hashlib
    # Create a consistent hash using sha256
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    # Use only first 16 chars of the hex digest to keep keys manageable
    return f"{prefix}_{hash_obj.hexdigest()[:16]}"


# Core processor for text-based retrieval-augmented generation (RAG) that handles embedding generation, 
# similarity calculations, and evaluation of student Q&A submissions against reference answers
class TextRAGProcessor:
    """Text-based RAG processor for evaluating Q&A submissions."""
    
    def __init__(self, mongodb_uri=None, use_openai=True, db_name="assignment_checker"):
        """
        Initialize TextRAGProcessor with MongoDB connection and similarity thresholds.
        
        Args:
            mongodb_uri: MongoDB connection URI (uses environment variable if None)
            use_openai: Whether to use OpenAI for embeddings (default: True)
            db_name: MongoDB database name (default: "assignment_checker")
        """
        # Set up logging
        self.logger = logger
        
        # Connect to MongoDB
        self.mongodb_client = None
        self.qa_embeddings = None
        self.student_submissions = None  # New collection for student submissions
        self.counters = None  # Collection for tracking counters
        self.connect_mongodb(mongodb_uri, db_name)
        
        # Set up local file storage for Q&A pairs
        self.qa_files_dir = os.path.join(os.getcwd(), "qa_files")
        self.setup_qa_directories()
        
        # Define similarity thresholds
        self.question_similarity_threshold = 0.7  # Threshold for matching questions
        self.high_quality_threshold = 0.92  # Threshold for high quality matches
        self.medium_quality_threshold = 0.75  # Threshold for medium quality matches
        self.low_quality_threshold = 0.60  # Threshold for low quality matches
        
        # Set up embedding service based on configuration
        self.use_openai = use_openai
        
        # Get OpenAI API key for embedding generation
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # If OpenAI is enabled but no API key is available, log a warning
        if self.use_openai and not self.openai_api_key:
            logger.warning("OpenAI embeddings requested but no API key provided - falling back to Ollama")
            self.use_openai = False
        
        # Initialize embedding models at startup
        self._initialize_embedding_models()
    
    # Initializes embedding models for OpenAI and Ollama by testing connectivity and warming up models
    def _initialize_embedding_models(self):
        """Pre-initialize embedding models to improve performance."""
        logger.info(f"Initializing embedding models... (use_openai={self.use_openai})")
        
        try:
            # Check if we can connect to OpenAI - always try to initialize if available
            if self.openai_api_key:
                try:
                    # Verify OpenAI connectivity with a small request
                    self.generate_embedding_openai("This is a test")
                    logger.info("OpenAI embedding model initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing OpenAI embedding model: {e}")
                
            # Check if we can connect to Ollama (needed if OpenAI not used)
            if not self.use_openai or not self.openai_api_key:
                try:
                    response = requests.get(f"{self.ollama_api_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name") for m in models]
                        
                        if self.ollama_model in model_names:
                            logger.info(f"Found Ollama model: {self.ollama_model}")
                        else:
                            available_models = ", ".join(model_names[:5])
                            logger.warning(f"Ollama model {self.ollama_model} not found. Available models: {available_models}")
                            
                        # Send a small request to warm up the model
                        self.generate_embedding_ollama("This is a test")
                        logger.info("Ollama embedding model initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing Ollama embedding model: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
    


    
    ## Generates text embeddings using OpenAI's API with caching to reduce API calls
    def generate_embedding_openai(self, text: str, bypass_cache: bool = False) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not set")
        
        # Create a more reliable cache key specifically for OpenAI embeddings
        cache_key = _create_cache_key(text, prefix="openai_embed")
        
        # Only use cache if bypass_cache is False
        if not bypass_cache and cache_key in _QUESTION_EMBEDDING_CACHE:
            logger.debug(f"Using cached OpenAI embedding for text: {text[:30]}...")
            return _QUESTION_EMBEDDING_CACHE[cache_key]
        
        try:
            logger.info(f"Generating new OpenAI embedding for text: {text[:30]}...")
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-ada-002",
                    "input": text
                }
            )
            response.raise_for_status()
            embedding = np.array(response.json()["data"][0]["embedding"])
            
            # Cache the embedding
            _QUESTION_EMBEDDING_CACHE[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise
    
    # Generates text embeddings using Ollama's API with caching to reduce API calls
    def generate_embedding_ollama(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama."""
        # Create a more reliable cache key specifically for Ollama embeddings
        cache_key = _create_cache_key(text, prefix=f"ollama_{self.ollama_model}_embed")
        
        if cache_key in _QUESTION_EMBEDDING_CACHE:
            logger.debug(f"Using cached Ollama embedding for text: {text[:30]}...")
            return _QUESTION_EMBEDDING_CACHE[cache_key]
        
        try:
            logger.info(f"Generating new Ollama embedding for text: {text[:30]}...")
            response = requests.post(
                f"{self.ollama_api_url}/api/embeddings",
                json={"model": self.ollama_model, "prompt": text}
            )
            response.raise_for_status()
            embedding = np.array(response.json()["embedding"])
            
            # Cache the embedding
            _QUESTION_EMBEDDING_CACHE[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {e}")
            raise
    
    # Primary embedding generation function that selects between OpenAI and Ollama based on configuration
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using selected model."""
        # Additional logging to help diagnose model selection issues
        logger.debug(f"Generating embedding with use_openai={self.use_openai}")
        
        # Ensure OpenAI is available if selected
        if self.use_openai:
            if not self.openai_api_key:
                logger.warning("OpenAI selected but API key not available. Falling back to Ollama.")
                return self.generate_embedding_ollama(text)
            return self.generate_embedding_openai(text)
        else:
            return self.generate_embedding_ollama(text)
    


    # Stores Q&A pair with embeddings in MongoDB for later retrieval and comparison
    def store_qa_embedding(self, qa_id: str, question: str, answer: str, 
                          embedding: np.ndarray, is_ideal: bool = True, question_embedding: np.ndarray = None,
                          submission_id: int = None) -> None:
        """Store Q&A embedding in MongoDB.
        
        Args:
            qa_id: Identifier for the Q&A pair
            question: The question text
            answer: The answer text
            embedding: The ANSWER embedding
            is_ideal: Whether this is from the ideal document
            question_embedding: The QUESTION embedding (if available)
            submission_id: The submission ID for student submissions (only used when is_ideal=False)
        """
        try:
            logger.info("========== DOCUMENT EMBEDDING STORAGE STAGE ==========")
            
            # Convert numpy arrays to lists for MongoDB storage
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            if question_embedding is not None:
                question_embedding_list = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
            else:
                question_embedding_list = None
            
            # Store in the appropriate collection based on type
            if is_ideal:
                # Store ideal Q&A pair in qa_embeddings collection
                # Use qa_id as the MongoDB document _id for ideal answers
                document = {
                                        "_id": qa_id,  # Use qa_id directly as the document ID
                    "qa_id": qa_id,
                    "question": question,
                    "answer": answer,
                                        "embedding": embedding_list,
                                        "question_embedding": question_embedding_list,
                                        "is_ideal": True,
                    "timestamp": datetime.now()
                }
            
                # Insert or update in MongoDB
                self.qa_embeddings.replace_one(
                        {"_id": qa_id},
                        document,
                        upsert=True
                )
                
                logger.info(f"Stored ideal Q&A pair in MongoDB: {qa_id}")
            else:
                # Store student submission in student_submissions collection
                # For student submissions, create a compound ID using both submission_id and qa_id
                compound_id = f"{submission_id}_{qa_id}"
                
                document = {
                    "_id": compound_id,  # Use compound ID to avoid duplicates
                    "qa_id": qa_id,
                    "submission_id": submission_id,
                    "question": question,
                    "answer": answer,
                    "embedding": embedding_list,
                    "question_embedding": question_embedding_list,
                    "is_ideal": False,
                    "timestamp": datetime.now()
                }
                
                # Insert or update in MongoDB
                self.student_submissions.replace_one(
                    {"_id": compound_id},
                    document,
                    upsert=True
                )
                
                logger.info(f"Stored student Q&A pair in MongoDB: {qa_id} (submission_id: {submission_id})")
        
        except Exception as e:
            logger.error(f"Error storing Q&A embedding: {e}")
            logger.error(traceback.format_exc())
    
    # Clears all embedding records from MongoDB to reset the system
    def clear_embeddings(self) -> None:
        """Clear all embeddings from MongoDB."""
        self.qa_embeddings.delete_many({})
    

    # Calculates cosine similarity between two embeddings for comparing text semantic similarity

    
    # 2. Processes a DOCX file to extract Q&A pairs and store embeddings.
    def process_qa_document(self, file_path: str, is_ideal: bool = True, submission_id: int = None) -> Dict[str, Dict[str, Any]]:
        """Process a DOCX file to extract Q&A pairs and store embeddings.
         
        Args:
            file_path: The path to the DOCX file
            is_ideal: Whether this is from the ideal document
            submission_id: The submission ID for student submissions (only used when is_ideal=False)
            
        Returns:
            Dictionary of question-answer pairs
        """
        # 2.1. Extract text from document
        text = DocxProcessor.extract_text_from_docx(file_path)
        
        if not text:
            logger.error(f"No text extracted from {file_path}")
            return {}
        
        # 2.2. Extract Q&A pairs from text
        qa_pairs = DocxProcessor.extract_qa_pairs(text)
        
        if not qa_pairs:
            logger.warning(f"No Q&A pairs found in document: {file_path}")
            return {}
            
        # 2.3. Normalize and deduplicate
        processed_qa = {}
        
        if is_ideal:
            # Ideal document processing
            logger.info(f"Processing ideal document with {len(qa_pairs)} Q&A pairs")
            id_prefix = "ideal"
        else:
            # Student submission processing
            logger.info(f"Processing student submission with {len(qa_pairs)} Q&A pairs")
            if submission_id is None:
                # Generate a submission ID if not provided
                submission_id = self.generate_submission_id()
            id_prefix = f"student_{submission_id}"
        
        # Process each Q&A pair
        # Check if qa_pairs is a dictionary of dictionaries or a list of tuples
        if isinstance(qa_pairs, dict):
            # It's a dictionary with nested Q&A dictionaries
            qa_items = qa_pairs.items()
        elif isinstance(qa_pairs, list):
            # It's a list of tuples (index, qa_dict)
            qa_items = enumerate(qa_pairs)
        else:
            logger.error(f"Unexpected qa_pairs format: {type(qa_pairs)}")
            return {}
        
        for i, qa_data in qa_items:
            if isinstance(qa_data, dict):
                # Direct access to question and answer
                question = qa_data.get("question", "")
                answer = qa_data.get("answer", "")
            else:
                # For enumerated list items, qa_data is the actual QA dict
                question = qa_data.get("question", "")
                answer = qa_data.get("answer", "")
            
            # Generate unique ID for the QA pair
            new_qa_id = f"{i+1}" if isinstance(i, int) else f"{i}"
            qa_id_with_prefix = f"{id_prefix}_{new_qa_id}"
            
            # Skip if question or answer is empty
            if not question.strip() or not answer.strip():
                logger.warning(f"Skipping empty Q&A pair: {qa_id_with_prefix}")
                continue
            
            # Process and store the embeddings
            try:
                # Generate embeddings for the answer
                answer_embedding = self.generate_embedding(answer)
            
                # Generate embeddings for the question (cached to avoid duplicate work)
                question_embedding = self.generate_embedding(question)
            
                # Store in MongoDB with the appropriate submission_id
                self.store_qa_embedding(
                    new_qa_id, question, answer, answer_embedding, 
                    is_ideal=is_ideal, 
                    question_embedding=question_embedding,
                    submission_id=None if is_ideal else submission_id
                )
            
                # Store in the results dictionary
                processed_qa[qa_id_with_prefix] = {
                    "question": question,
                    "answer": answer,
                    "embedding": answer_embedding,
                    "question_embedding": question_embedding
                }
            
                # Log the Q&A pair (first 100 chars only)
                q_preview = question[:100] + "..." if len(question) > 100 else question
                a_preview = answer[:100] + "..." if len(answer) > 100 else answer
            
                # Convert i to string before concatenation to avoid type errors
                index_str = str(i) if isinstance(i, str) else str(i+1)
                logger.info(f"  [{index_str}] Question: {q_preview}")
                logger.info(f"      Answer: {a_preview}")
            except Exception as e:
                logger.error(f"Error processing Q&A pair {qa_id_with_prefix}: {e}")
                logger.error(traceback.format_exc())
        
        # Save Q&A pairs to a local JSON file
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Create qa_files directory if it doesn't exist
        qa_files_dir = os.path.join(os.getcwd(), "qa_files")
        
        if is_ideal:
            # Save ideal Q&A pairs
            json_path = os.path.join(qa_files_dir, "ideal", f"{base_name}.json")
            self.save_qa_pairs_to_json(processed_qa, json_path)
        else:
            # Save student Q&A pairs with submission ID
            json_path = os.path.join(qa_files_dir, "student", f"{base_name}_submission_{submission_id}.json")
            self.save_qa_pairs_to_json(processed_qa, json_path)
        
        return processed_qa
    

    # Extracts Q&A pairs from document text using LLM when regular regex parsing fails.
    def extract_qa_pairs_with_llm(self, document_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract question-answer pairs from document text using LLM when regular parsing fails.
        
        Args:
            document_text: The raw document text
            
        Returns:
            Dictionary of question-answer pairs
        """
        from datetime import datetime
        import re
        import json
        from utils.tokenizer_utils import safe_truncate_code
        
        try:
            logger.info(f"Extracting QA pairs from document text ({len(document_text)} chars) using LLM")
            
            # Safely truncate the document text to avoid token limits
            truncated_text = safe_truncate_code(document_text, 6000)
            if len(truncated_text) < len(document_text):
                logger.warning(f"Document text truncated from {len(document_text)} to {len(truncated_text)} chars due to token limits")
            
            # Format the prompt
            prompt = QA_EXTRACTION_PROMPT.format(
                document_text=truncated_text
            )
            
            # Track start time
            start_time = datetime.now()
            
            # Use OpenAI if configured, otherwise Ollama
            if self.use_openai:
                logger.info("Using OpenAI for QA extraction")
                
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.openai_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-3.5-turbo",  # Using a more cost-effective model for extraction
                            "messages": [
                                {"role": "system", "content": "You are a document parsing assistant. Extract question-answer pairs from documents and format them as JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 2000,
                            "response_format": {"type": "json_object"}  # Request JSON format directly
                        },
                        timeout=30
                    )
                    
                    response.raise_for_status()
                    time_taken = (datetime.now() - start_time).total_seconds()
                    logger.info(f"OpenAI QA extraction completed in {time_taken:.2f} seconds")
                    
                    result = response.json()
                    # Example OpenAI response:
                    """{
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1677652288,
                        "model": "gpt-3.5-turbo",
                        "choices": [
                            {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello there, how can I help you today?"
                            },
                            "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 9,
                            "completion_tokens": 12,
                            "total_tokens": 21
                        }
                    }
                    """
                    # Safely extract content with error handling
                    if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0]:
                        content = result["choices"][0]["message"].get("content", "{}")
                        try:
                            qa_pairs = json.loads(content)
                            logger.info(f"Successfully extracted {len(qa_pairs)} QA pairs")
                            return qa_pairs
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing OpenAI JSON response: {e}")
                            # Try to extract JSON from the content if direct parsing fails
                            extracted_json = self._extract_json_from_text(content)
                            if extracted_json:
                                return extracted_json
                            return {}
                    else:
                        logger.error(f"Unexpected response format from OpenAI: {result}")
                        return {}
                except Exception as e:
                    logger.error(f"Error with OpenAI QA extraction: {e}")
                    return {}
            else:
                logger.info("Using Ollama for QA extraction")
                
                try:
                    # Ollama request with clear instructions for JSON output
                    response = requests.post(
                        f"{self.ollama_api_url}/api/generate",
                        json={
                            "model": self.ollama_model,
                            "prompt": f"System: You are a document parsing assistant. Extract question-answer pairs from documents and format them as JSON.\n\nUser: {prompt}",
                            "stream": False,
                            "options": {
                                "temperature": 0.2
                            }
                        },
                        timeout=60
                    )
                    
                    response.raise_for_status()
                    time_taken = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Ollama QA extraction completed in {time_taken:.2f} seconds")
                    
                    extraction_text = response.json().get("response", "")
                    
                    # Try multiple methods to extract valid JSON
                    qa_pairs = self._extract_json_from_text(extraction_text)
                    
                    if qa_pairs:
                        logger.info(f"Successfully extracted {len(qa_pairs)} QA pairs")
                        return qa_pairs
                    else:
                        logger.warning("No valid JSON found in extraction response")
                        return {}
                    
                except Exception as e:
                    logger.error(f"Error with Ollama QA extraction: {e}")
                    logger.error(traceback.format_exc())
                    return {}
            
        except Exception as e:
            logger.error(f"Error extracting QA pairs with LLM: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from a text response.
        """
        try:
            # Try to find JSON within the text using regex patterns
            json_pattern = r'```(?:json)?\s*({.*?})\s*```'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                # Found JSON inside code blocks
                json_str = json_match.group(1)
            else:
                # Try to extract just a plain JSON object
                json_pattern = r'({[\s\S]*})'
                json_match = re.search(json_pattern, text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    logging.warning("Could not extract JSON from text")
                    return {}
            
            # Handle escaped characters and quotes to make the JSON valid
            json_str = json_str.replace('\\"', '"').replace("\\'", "'")
            
            # Handle potential double braces ({{ }}) that may come from prompt templates
            json_str = json_str.replace('{{', '{').replace('}}', '}')
            
            # Parse the JSON string
            json_data = json.loads(json_str)
            return json_data
        except Exception as e:
            logging.error(f"Error extracting JSON from text: {e}")
            logging.debug(f"Text that failed JSON extraction: {text[:500]}...")
            return {}
    
    # Migrates old format embeddings in the database to include the embedding_type field
    def _migrate_embeddings(self):
        """Migrate old format embeddings to new format with answer_embedding field."""
        try:
            # Find records that don't have embedding_type field (old format)
            cursor = self.qa_embeddings.find({"embedding_type": {"$exists": False}})
            count = 0
            
            for doc in cursor:
                qa_id = doc.get("qa_id")
                embedding = doc.get("embedding")
                
                if not qa_id or embedding is None:
                    continue
                    
                # Update the document to include the embedding_type field
                self.qa_embeddings.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding_type": "answer"}}
                )
                count += 1
            
            if count > 0:
                logger.info(f"Migrated {count} embeddings to new format")
                
        except Exception as e:
            logger.error(f"Error migrating embeddings: {e}")




    # 3.1 Retrieves ideal Q&A embeddings from MongoDB and handles backward compatibility
    def retrieve_ideal_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve ideal embeddings from MongoDB."""
        try:
            # First, try to migrate any old format embeddings
            self._migrate_embeddings()
            
            cursor = self.qa_embeddings.find({"is_ideal": True})
            count = 0
            count_with_question_embeddings = 0
            
            ideal_qa_pairs = {}
            for doc in cursor:
                qa_id = doc.get("qa_id")
                embedding = doc.get("embedding")  # This is stored as "embedding" in MongoDB
                question_embedding = doc.get("question_embedding")  # Get question embedding if available
                
                if not qa_id or embedding is None:
                    logger.warning(f"Missing qa_id or embedding for document in MongoDB, skipping")
                    continue
                    
                # Store all the data
                ideal_qa_pairs[qa_id] = {
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "answer_embedding": embedding,  # Map the "embedding" field to "answer_embedding"
                    "embedding": embedding,  # Keep the original field for backward compatibility
                    "embedding_type": doc.get("embedding_type", "answer")  # Default to "answer" if not specified
                }
                
                # Add question embedding if available
                if question_embedding is not None:
                    ideal_qa_pairs[qa_id]["question_embedding"] = question_embedding
                    count_with_question_embeddings += 1
                
                count += 1
            
            logger.info(f"Retrieved {count} ideal Q&A pairs from MongoDB ({count_with_question_embeddings} with question embeddings)")
            
            # Generate missing question embeddings if needed
            if count > count_with_question_embeddings:
                logger.warning(f"Found {count - count_with_question_embeddings} Q&A pairs without question embeddings, generating them now")
                missing_embeddings = self._generate_missing_question_embeddings(ideal_qa_pairs)
                logger.info(f"Generated {missing_embeddings} missing question embeddings")
            
            return ideal_qa_pairs
        except Exception as e:
            logger.error(f"Error retrieving ideal embeddings: {e}")
            return {}
    
    # Generates missing question embeddings for Q&A pairs that have answer embeddings but not question embeddings
    def _generate_missing_question_embeddings(self, qa_pairs: Dict[str, Dict[str, Any]]) -> int:
        """
        Generate missing question embeddings for Q&A pairs.
        
        Args:
            qa_pairs: Dictionary of Q&A pairs
            
        Returns:
            Number of embeddings generated
        """
        count = 0
        for qa_id, qa_pair in qa_pairs.items():
            # Skip if already has question embedding
            if "question_embedding" in qa_pair:
                continue
                
            # Get question text
            question = qa_pair.get("question", "")
            if not question:
                logger.warning(f"Skipping {qa_id} - missing question text")
                continue
                
            try:
                # Generate new question embedding
                question_embedding = self.generate_embedding(question)
                
                # Add to the qa_pair
                question_embedding_list = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
                qa_pairs[qa_id]["question_embedding"] = question_embedding_list
                
                # Update MongoDB
                self.qa_embeddings.update_one(
                    {"qa_id": qa_id, "is_ideal": True},
                    {"$set": {"question_embedding": question_embedding_list}}
                )
                
                count += 1
                logger.debug(f"Generated and stored missing question embedding for {qa_id}")
            except Exception as e:
                logger.error(f"Error generating question embedding for {qa_id}: {e}")
                
        return count

    def _verify_embedding_model(self):
        """Verify that the selected embedding model is available and make adjustments if needed."""
        if self.use_openai:
            # Check if OpenAI API key is available when OpenAI is selected
            if not self.openai_api_key:
                logger.warning("OpenAI selected but API key not available. Falling back to Ollama.")
                self.use_openai = False
                return False
            else:
                logger.info("Verified OpenAI API key is available for embedding generation")
                return True
        else:
            # Check if Ollama is available
            try:
                response = requests.get(f"{self.ollama_api_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Verified Ollama is available for embedding generation")
                    return True
                else:
                    logger.warning(f"Ollama not responding correctly: status {response.status_code}")
                    # If OpenAI key is available, fall back to OpenAI
                    if self.openai_api_key:
                        logger.warning("Falling back to OpenAI due to Ollama issues")
                        self.use_openai = True
                        return True
                    return False
            except Exception as e:
                logger.warning(f"Cannot connect to Ollama: {e}")
                # If OpenAI key is available, fall back to OpenAI
                if self.openai_api_key:
                    logger.warning("Falling back to OpenAI due to Ollama issues")
                    self.use_openai = True
                    return True
                return False
        


    # 1. Evaluates a QA submission against an ideal answer.
    def evaluate_qa_submission(self, submission_path: str, ideal_path: str) -> Dict[str, Any]:
        """
        Evaluate a QA submission against an ideal answer.
        Returns a dictionary with the evaluation results.
        
        This method:
        1. Extracts Q&A pairs from both documents
        2. Generates embeddings for questions and answers
        3. Maps student answers to ideal answers based on question similarity
        4. Evaluates answer quality based on answer similarity
        5. Returns detailed evaluation results
        """
        try:
            # Generate a unique evaluation ID
            eval_id = os.urandom(4).hex()
            logger.info(f"Starting evaluation {eval_id} - submission: {os.path.basename(submission_path)}, ideal: {os.path.basename(ideal_path)}")
            
            # Generate an incremental submission ID for the student submission
            result = self.counters.find_one_and_update(
                {"_id": "student_submission_id"},
                {"$inc": {"value": 1}},
                return_document=True
            )
            submission_id = result["value"]
            submission_timestamp = datetime.now()
            logger.info(f"Generated submission ID: {submission_id} at {submission_timestamp}")
            
            # Verify embedding model availability
            model_available = self._verify_embedding_model()
            if not model_available:
                return {
                    "status": "error",
                    "message": "No embedding model available. Please check your configuration."
                }
            
            logger.info(f"Using model: {'OpenAI' if self.use_openai else 'Ollama'}")
            
            # Clear existing embeddings and cache to ensure fresh results
            self.clear_embeddings()
            global _QUESTION_EMBEDDING_CACHE
            _QUESTION_EMBEDDING_CACHE = {}
            logger.info(f"Cleared embedding cache and database for fresh evaluation")
            
            # 2.1 Process ideal document
            logger.info("========== DOCUMENT PROCESSING STAGE ==========")
            logger.info(f"Processing ideal document: {ideal_path}")
            ideal_qa_pairs = self.process_qa_document(ideal_path, is_ideal=True)
            
            if not ideal_qa_pairs:
                return {
                    "status": "error",
                    "message": "No questions found in the ideal document. Please check the document format."
                }
            
            # 2.2 Process submission document with the submission ID
            logger.info(f"Processing submission document: {submission_path}")
            submission_qa_pairs = self.process_qa_document(submission_path, is_ideal=False, submission_id=submission_id)
            
            if not submission_qa_pairs:
                return {
                    "status": "error",
                    "message": "No questions found in the submission. Please check the document format."
                }
            
            # Check for embedding errors
            if self._has_embedding_errors(ideal_qa_pairs) or self._has_embedding_errors(submission_qa_pairs):
                logger.warning("Embedding errors detected, retrieving ideal embeddings from database")
                # Try to retrieve embeddings from MongoDB
                ideal_qa_pairs = self.retrieve_ideal_embeddings()
                
                if not ideal_qa_pairs:
                    return {
                        "status": "error",
                        "message": "Error generating embeddings. Please try again or check the embedding service."
                    }
            
            # Retrieve student embeddings from MongoDB to ensure we're using the stored versions
            logger.info("Retrieving student embeddings from MongoDB")
            stored_student_qa_pairs = self.retrieve_student_embeddings(submission_id)
            
            if not stored_student_qa_pairs:
                logger.warning("No stored student embeddings found, using in-memory embeddings")
            else:
                logger.info(f"Using {len(stored_student_qa_pairs)} stored student embeddings for evaluation")
                submission_qa_pairs = stored_student_qa_pairs
            
            # 3. Map submission QA pairs to ideal QA pairs and return the quality of each mapping
            logger.info("Mapping submission QA pairs to ideal QA pairs")
            qa_mappings = self._map_qa_pairs(submission_qa_pairs)
            
            # Log similarity samples to help with tuning
            self._log_similarity_samples(qa_mappings)
            
            # Calculate overall statistics
            total_questions = len(ideal_qa_pairs)
            high_matches = sum(1 for m in qa_mappings if m["quality"] == "high")
            medium_matches = sum(1 for m in qa_mappings if m["quality"] == "medium")
            low_matches = sum(1 for m in qa_mappings if m["quality"] == "low")
            poor_matches = sum(1 for m in qa_mappings if m["quality"] == "poor")
            missing = sum(1 for m in qa_mappings if m["quality"] == "missing")
            
            # Log score distribution
            logger.info(f"Score distribution: High={high_matches}, Medium={medium_matches}, Low={low_matches}, Poor={poor_matches}, Missing={missing}")
            
            # Ensure numbers add up correctly
            if high_matches + medium_matches + low_matches + poor_matches + missing != total_questions:
                logger.warning(f"Question count mismatch: {high_matches + medium_matches + low_matches + poor_matches + missing} vs {total_questions}")
            
            # Calculate overall score (weighted)
            logger.info("========== SCORING STAGE ==========")
            logger.info(f"Calculating overall score from {total_questions} questions")
            overall_score = (high_matches * 1.0 + medium_matches * 0.7 + low_matches * 0.4 + poor_matches * 0.1) / total_questions
            overall_score = round(overall_score * 100)
            logger.info(f"Calculated overall score: {overall_score}%")
            
            # Format summary evaluation text for each question
            question_evaluations = self._format_question_evaluations(qa_mappings, submission_qa_pairs, ideal_qa_pairs)
            
            # Generate overall summary
            summary = self._generate_summary(
                question_evaluations, 
                total_questions, 
                high_matches, 
                medium_matches, 
                low_matches,
                overall_score
            )
            
            # Store submission metadata in MongoDB
            submission_metadata = {
                "submission_id": submission_id,
                "eval_id": eval_id,
                "timestamp": submission_timestamp,
                "overall_score": overall_score,
                "total_questions": total_questions,
                "high_matches": high_matches,
                "medium_matches": medium_matches,
                "low_matches": low_matches,
                "poor_matches": poor_matches,
                "missing": missing,
                "filename": os.path.basename(submission_path)
            }
            
            # Use a compound ID for metadata documents to avoid duplicate key errors
            metadata_doc_id = f"metadata_{submission_id}"
            self.student_submissions.replace_one(
                {"_id": metadata_doc_id},
                {
                    "_id": metadata_doc_id,
                    **submission_metadata
                },
                upsert=True
            )
            logger.info(f"Stored submission metadata for submission_id: {submission_id}")
            
            # Clean the mappings to ensure they can be JSON serialized
            cleaned_mappings = self._clean_for_json(qa_mappings)
            
            # Format the response to match what the UI expects
            result = {
                "status": "success",
                "eval_id": eval_id,
                "submission_id": submission_id,  # Include the submission ID in the response
                "submission_timestamp": submission_timestamp.isoformat(),  # Include the timestamp
                "overall_score": overall_score,  # Added at top level for UI compatibility
                "stats": {
                    "total_questions": total_questions,
                    "high_count": high_matches,
                    "medium_count": medium_matches,
                    "low_count": low_matches,
                    "poor_count": poor_matches,
                    "missing_count": missing,
                    "overall_score": overall_score
                },
                "evaluations": self._format_evaluations_for_ui(qa_mappings, submission_qa_pairs, ideal_qa_pairs),
                "summary": summary,
                "result": {
                    "overall_score": overall_score,
                    "total_questions": total_questions,
                    "high_quality_matches": high_matches,
                    "medium_quality_matches": medium_matches,
                    "low_quality_matches": low_matches,
                    "poor_quality_matches": poor_matches,
                    "missing_answers": missing,
                    "question_mapping": cleaned_mappings
                }
            }
            
            # Save evaluation results to a local JSON file
            submission_filename = os.path.basename(submission_path)
            base_name = os.path.splitext(submission_filename)[0]
            evaluation_path = os.path.join(self.qa_files_dir, "evaluations", f"{base_name}_eval_{submission_id}.json")
            
            # Ensure evaluations directory exists
            os.makedirs(os.path.join(self.qa_files_dir, "evaluations"), exist_ok=True)
            
            # Save the evaluation results
            import json
            with open(evaluation_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)  # default=str handles datetime serialization
            
            logger.info(f"Saved evaluation results to {evaluation_path}")
            submission_filename = os.path.basename(submission_path)
            base_name = os.path.splitext(submission_filename)[0]
            evaluation_path = os.path.join(self.qa_files_dir, "evaluations", f"{base_name}_eval_{submission_id}.json")
            
            # Ensure evaluations directory exists
            os.makedirs(os.path.join(self.qa_files_dir, "evaluations"), exist_ok=True)
            
            # Save the evaluation results
            import json
            with open(evaluation_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)  # default=str handles datetime serialization
            
            logger.info(f"Saved evaluation results to {evaluation_path}")
            
            # Return the final evaluation result
            return result
            
        except Exception as e:
            logger.error(f"Error during QA evaluation: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Error evaluating submission: {str(e)}"
            }
    
    def _format_question_evaluations(self, qa_mappings, submission_qa_pairs, ideal_qa_pairs) -> str:
        """Format question evaluations for summary generation, with proper truncation."""
        from utils.tokenizer_utils import safe_truncate_code
        
        question_evaluations = ""
        for mapping in qa_mappings:
            try:
                # Handle missing submissions (quality = "missing")
                if mapping.get("quality") == "missing" or mapping.get("submission_id") is None:
                    question = safe_truncate_code(mapping["ideal_question"], 200)
                    question_evaluations += f"Question: {question}\n"
                    question_evaluations += "Student Answer: [No answer provided]\n"
                    question_evaluations += f"Reference Answer: {safe_truncate_code(mapping['ideal_answer'], 400)}\n"
                    question_evaluations += "Quality: MISSING\n"
                    question_evaluations += "Similarity: 0.00%\n\n"
                    continue
                    
                # Truncate text if needed to prevent token overflow
                submission_id = mapping["submission_id"]
                ideal_id = mapping["ideal_id"]
                
                question = safe_truncate_code(submission_qa_pairs[submission_id]['question'], 200)
                student_answer = safe_truncate_code(submission_qa_pairs[submission_id]['answer'], 400)
                reference_answer = safe_truncate_code(ideal_qa_pairs[ideal_id]['answer'], 400)
                
                question_evaluations += f"Question: {question}\n"
                question_evaluations += f"Student Answer: {student_answer}\n"
                question_evaluations += f"Reference Answer: {reference_answer}\n"
                question_evaluations += f"Quality: {mapping['quality'].upper()}\n"
                question_evaluations += f"Similarity: {mapping['similarity'] * 100:.2f}%\n\n"
            except Exception as e:
                logger.error(f"Error formatting question evaluation: {e}")
                # Add a placeholder for this evaluation to ensure continuity
                question_evaluations += f"[Error processing this question/answer pair: {str(e)}]\n\n"
        
        return question_evaluations
    
    def _generate_summary(self, question_evaluations, total_questions, high_count, medium_count, low_count, overall_score) -> str:
        """Generate a summary of the evaluation results."""
        logger.info("========== SUMMARY GENERATION STAGE ==========")
        logger.info(f"Generating evaluation summary for {total_questions} questions (Score: {overall_score}%)")
        
        try:
            prompt = QA_SUMMARY_PROMPT.format(
                question_evaluations=question_evaluations,
                total_questions=total_questions,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                overall_score=overall_score
            )
            
            if self.use_openai and self.openai_api_key:
                logger.info("Generating summary with OpenAI")
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4-0125-preview",
                        "messages": [
                            {"role": "system", "content": "You are an education assistant helping evaluate student performance across multiple questions."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 1500
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                # Safely extract content
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0]:
                    content = result["choices"][0]["message"].get("content", "")
                    # Wrap the content in HTML for better UI display
                    return f"""
<div class="evaluation-summary ai-generated">
    {content}
</div>
"""
                else:
                    logger.error(f"Unexpected response format from OpenAI: {result}")
                    return "Error generating summary: unexpected response format"
            else:
                # Fallback to a simple summary if OpenAI is not available
                logger.warning("OpenAI API not available for summary generation, using simple summary")
                return self._generate_simple_summary(total_questions, high_count, medium_count, low_count, overall_score)
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._generate_simple_summary(total_questions, high_count, medium_count, low_count, overall_score)
    
    def _generate_simple_summary(self, total_questions, high_count, medium_count, low_count, overall_score) -> str:
        """Generate a simple summary when AI is not available."""
        return f"""
<div class="evaluation-summary">
    <h2 class="summary-title">Performance Summary</h2>
    
    <div class="score-section">
        <div class="overall-score">
            <h3>Overall Score</h3>
            <div class="score-value">{overall_score}%</div>
        </div>
        
        <div class="score-breakdown">
            <h3>Performance Breakdown</h3>
            <ul class="stats-list">
                <li><span class="stat-label">Total Questions:</span> <span class="stat-value">{total_questions}</span></li>
                <li><span class="stat-label">High Quality Answers:</span> <span class="stat-value">{high_count}</span></li>
                <li><span class="stat-label">Medium Quality Answers:</span> <span class="stat-value">{medium_count}</span></li>
                <li><span class="stat-label">Low Quality Answers:</span> <span class="stat-value">{low_count}</span></li>
                <li><span class="stat-label">Poor/Missing Answers:</span> <span class="stat-value">{total_questions - (high_count + medium_count + low_count)}</span></li>
            </ul>
        </div>
    </div>
    
    <div class="recommendations-section">
        <h3>Recommendations</h3>
        <ol class="recommendation-list">
            <li>Review the specific feedback provided for each question</li>
            <li>Focus on improving areas marked as 'low' or 'poor' quality</li>
            <li>Revisit course materials related to questions you struggled with</li>
            <li>Practice formulating more complete answers that address all aspects of each question</li>
        </ol>
    </div>
</div>
"""
    
    def _generate_answer_feedback(self, student_answer: str, reference_answer: str, quality: str) -> str:
        """Generate feedback for a student answer compared to reference answer.Args:
            student_answer: The student's submitted answer
            reference_answer: The reference/ideal answer
            quality: Quality level (high, medium, low, poor)
            
        Returns:
            Detailed feedback with specific insights on the student's answer
        """
        logger.info("========== GENERATION STAGE ==========")
        logger.info(f"Generating {quality.lower()} quality feedback using {'OpenAI' if self.use_openai else 'local'} API")
        
        start_time = datetime.now()
        
        try:
            # Truncate very long answers to avoid token limits
            student_answer = self.safe_truncate_text(student_answer, 2000)
            reference_answer = self.safe_truncate_text(reference_answer, 2000)
            
            # Get detailed similarity metrics to enrich the feedback
            similarity_metrics = self.compare_answers(student_answer, reference_answer)
            
            # Add metrics to guide the feedback generator
            metrics_suffix = (
                f"\n\nAnswer Metrics:\n"
                f"- Embedding Similarity: {similarity_metrics['embedding_similarity']:.2f}\n"
                f"- Text Similarity: {similarity_metrics['text_similarity']:.2f}\n"
                f"- Term Overlap: {similarity_metrics['token_overlap']:.2f}\n"
                f"- Combined Similarity: {similarity_metrics['combined_similarity']:.2f}\n"
                f"- Overall Quality: {quality.upper()}"
            )
            
            
            # Create enriched prompt with both answers and metrics
            enriched_prompt = QA_EVALUATION_PROMPT.format(
                student_answer=student_answer,
                reference_answer=reference_answer
            ) + metrics_suffix
            
            # Use modern OpenAI client API format compatible with v1.0.0+
            if self.openai_api_key:
                logger.info(f"Generating {quality} quality feedback using OpenAI API")
                try:
                    # New OpenAI API format (v1.0.0+)
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.openai_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            # Choose model based on quality level
                            "model": "gpt-4" if quality in ["medium", "low"] else "gpt-3.5-turbo",
                            "messages": [
                                {"role": "system", "content": "You are an educational assistant providing feedback on student answers."},
                                {"role": "user", "content": enriched_prompt}
                            ],
                            "temperature": 0.5,
                            "max_tokens": 1024 if quality in ["medium", "low"] else 512
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    
                    # Extract the content from the response
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        if "message" in response_json["choices"][0]:
                            feedback = response_json["choices"][0]["message"]["content"].strip()
                        else:
                            logger.error(f"Unexpected response format: {response_json}")
                            raise ValueError("Invalid response format from OpenAI API")
                    else:
                        logger.error(f"No choices in response: {response_json}")
                        raise ValueError("No choices in response from OpenAI API")
                    
                except Exception as e:
                    logger.error(f"Error generating feedback with OpenAI: {str(e)}")
                    # Fallback to default messages
                    feedback = self._get_default_feedback(quality)
            else:
                # No API key available, use default feedback
                logger.warning("No OpenAI API key available, using default feedback")
                feedback = self._get_default_feedback(quality)
            
            # Log performance
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"Generated {quality.lower()} quality feedback in {processing_time:.2f}s")
            
            # Add a default message if feedback generation fails or returns empty
            if not feedback or len(feedback) < 10:
                feedback = self._get_default_feedback(quality)
                
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            logger.error(traceback.format_exc())
            
            # Provide a generic fallback message if feedback generation fails
            return self._get_default_feedback(quality)
    
    def _get_default_feedback(self, quality: str) -> str:
        """Return default feedback based on quality level when AI generation fails."""
        if quality == "high":
            return "Excellent work! Your answer covers all the key points and demonstrates a strong understanding of the concepts. The response is comprehensive, accurate, and well-structured."
        elif quality == "medium":
            return "Good answer that covers many key points, but there's room for improvement in completeness and detail. Consider expanding on the core concepts and providing more specific examples to strengthen your response."
        elif quality == "low":
            return "Your answer addresses some aspects correctly, but misses important details and concepts. Review the course materials on this topic, focusing especially on the key terminology and fundamental principles that would make your answer more complete."
        else:
            return "Your answer needs significant improvement. It appears you may be missing some fundamental understanding of this topic. I recommend reviewing the course materials, focusing on the basic concepts and terminology, and then try reframing your answer to address all parts of the question."
    
    # 3. Maps student submission Q&A pairs to the most similar ideal Q&A pairs using two-phase matching
    def _map_qa_pairs(self, submission_qa_pairs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map submission QA pairs to ideal QA pairs and evaluate quality.
        
        Args:
            submission_qa_pairs: Dictionary of student QA pairs, including their embeddings
            
        Returns:
            List of QA mappings, each with quality score
        """
        logger.info("========== MAPPING STAGE ==========")
        mappings = []
        
        # Verify that student embeddings are available
        for qa_id, qa_data in submission_qa_pairs.items():
            if 'embedding' not in qa_data or qa_data['embedding'] is None:
                logger.warning(f"Student answer embedding missing for {qa_id}, skipping")
                continue
                
            if 'question_embedding' not in qa_data or qa_data['question_embedding'] is None:
                logger.warning(f"Student question embedding missing for {qa_id}, skipping")
                continue
        
        # Retrieve ideal QA pairs
        ideal_qa_pairs = self.retrieve_ideal_embeddings()
        if not ideal_qa_pairs:
            logger.error("Failed to retrieve ideal QA pairs from MongoDB")
            return []
        
        # Ensure ideal embeddings are available
        for qa_id, qa_data in ideal_qa_pairs.items():
            if 'embedding' not in qa_data or qa_data['embedding'] is None:
                logger.warning(f"Ideal answer embedding missing for {qa_id}, skipping")
                continue
                
            if 'question_embedding' not in qa_data or qa_data['question_embedding'] is None:
                logger.warning(f"Ideal question embedding missing for {qa_id}, skipping")
                continue
            
        # First pass: Map based on question similarity
        for ideal_qa_id, ideal_qa_data in ideal_qa_pairs.items():
            # Get the question embedding for the ideal question
            ideal_question_embedding = ideal_qa_data.get('question_embedding')
            if ideal_question_embedding is None:
                logger.warning(f"Ideal question embedding missing for {ideal_qa_id}, skipping")
                continue
            
            # Find the most similar student question
            best_match = None
            best_similarity = -1
            
            for sub_qa_id, sub_qa_data in submission_qa_pairs.items():
                # Get the question embedding for the student question
                sub_question_embedding = sub_qa_data.get('question_embedding')
                if sub_question_embedding is None:
                    logger.warning(f"Student question embedding missing for {sub_qa_id}, skipping")
                    continue
                
                # Calculate question similarity
                q_similarity = self._compute_similarity(sub_question_embedding, ideal_question_embedding)
                
                if q_similarity > best_similarity:
                    best_similarity = q_similarity
                    best_match = sub_qa_id
            
            # No match found
            if best_match is None or best_similarity < self.question_similarity_threshold:
                logger.info(f"No good match found for question {ideal_qa_id}")
                mappings.append({
                    'ideal_qa_id': ideal_qa_id,
                    'student_qa_id': None,
                    'question_similarity': 0,
                    'answer_similarity': 0,
                    'quality': "missing"
                })
                continue
                
            # Get the answer embeddings
            ideal_answer_embedding = ideal_qa_data.get('embedding')
            sub_answer_embedding = submission_qa_pairs[best_match].get('embedding')
            
            if ideal_answer_embedding is None or sub_answer_embedding is None:
                logger.warning(f"Answer embedding missing, skipping comparison")
                continue
                
            # Calculate answer similarity
            answer_similarity = self._compute_similarity(sub_answer_embedding, ideal_answer_embedding)
            
            # Determine quality based on answer similarity
            quality = self._determine_quality(answer_similarity)
                
            # Add mapping
            mappings.append({
                'ideal_qa_id': ideal_qa_id,
                'student_qa_id': best_match,
                'question_similarity': float(best_similarity),
                'answer_similarity': float(answer_similarity),
                'quality': quality
            })
            
            logger.info(f"Mapped question {ideal_qa_id} to {best_match} with Q-sim={best_similarity:.4f}, A-sim={answer_similarity:.4f}, quality={quality}")
                
        return mappings
        
    def _compute_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two embeddings.
        Handles both numpy arrays and lists.
        """
        import numpy as np
        
        # Convert to numpy arrays if they're lists
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
        
        # Ensure we're working with 1D arrays
        if len(vec1.shape) > 1:
            vec1 = vec1.flatten()
        if len(vec2.shape) > 1:
            vec2 = vec2.flatten()
        
        # Compute cosine similarity
        similarity = float(1 - cosine(vec1, vec2))
            
        # Ensure the result is a python float
        return float(similarity)
    
    def _determine_quality(self, similarity):
        if similarity >= self.high_quality_threshold:
            return "high"
        elif similarity >= self.medium_quality_threshold:
            return "medium"
        elif similarity >= self.low_quality_threshold:
            return "low"
        else:
            return "poor"
    
    def _clean_for_json(self, obj):
        """Clean objects for JSON serialization."""
        import numpy as np
        
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
            return obj.tolist()
        else:
            # Try to convert to a string if all else fails
            try:
                return str(obj)
            except:
                return "Non-serializable object"

    def _has_embedding_errors(self, qa_pairs: Dict[str, Dict[str, Any]]) -> bool:
        """Check if any Q&A pairs have embedding errors."""
        for qa_id, qa_pair in qa_pairs.items():
            if "error" in qa_pair or "embedding" not in qa_pair:
                return True
        return False



    # Generates comprehensive evaluation results from the mappings between student and ideal answers
    def evaluate(self, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate evaluation results from the mappings.
        Uses answer similarity as the primary metric for quality assessment.
        
        Args:
            mappings: List of mappings between submission and ideal Q&A pairs
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("========== EVALUATION STAGE ==========")
        logger.info(f"Evaluating {len(mappings)} answer mappings")
        
        if not mappings:
            logger.warning("No mappings to evaluate")
            return {
                "quality_counts": {"high": 0, "medium": 0, "low": 0, "poor": 0, "missing": 0},
                "summary": {"total": 0, "matched": 0, "missing": 0},
                "evaluations": []
            }
        
        # Count the number of Q&A pairs by quality level
        quality_counts = {
            "high": 0,
            "medium": 0, 
            "low": 0,
            "poor": 0,
            "missing": 0
        }
        
        evaluations = []
        for mapping in mappings:
            quality = mapping.get("quality", "poor")
            
            # Update quality counts
            if quality in quality_counts:
                quality_counts[quality] += 1
            else:
                logger.warning(f"Unknown quality level: {quality}")
                quality_counts["poor"] += 1
                quality = "poor"  # Default to poor if unknown
            
            # Generate evaluation text based on quality
            # Use the answer similarity score for context
            evaluation_text = self._get_evaluation_text_for_quality(
                quality, 
                mapping.get("answer_similarity", 0),  # Use answer similarity for feedback
                mapping.get("question_similarity", 0)
            )
            
            # Add evaluation details
            evaluation = {
                "ideal_id": mapping.get("ideal_id", ""),
                "ideal_question": mapping.get("ideal_question", ""),
                "ideal_answer": mapping.get("ideal_answer", ""),
                "submission_id": mapping.get("submission_id", None),
                "submission_question": mapping.get("submission_question", ""),
                "submission_answer": mapping.get("submission_answer", ""),
                "similarity": mapping.get("answer_similarity", 0),  # Primary score is answer similarity
                "question_similarity": mapping.get("question_similarity", 0),
                "quality": quality,
                "evaluation": evaluation_text
            }
            evaluations.append(evaluation)
        
        # Count total, matched, and missing Q&A pairs
        total = len(evaluations)
        missing = quality_counts["missing"]
        matched = total - missing
        
        # Generate summary
        summary = {
            "total": total,
            "matched": matched,
            "missing": missing
        }
        
        # Generate results
        results = {
            "quality_counts": quality_counts,
            "summary": summary,
            "evaluations": evaluations
        }
        
        # Log the evaluation results
        logger.info(f"Evaluation results: {total} total, {matched} matched, {missing} missing")
        logger.info(f"Quality counts: high={quality_counts['high']}, medium={quality_counts['medium']}, "
                    f"low={quality_counts['low']}, poor={quality_counts['poor']}, missing={quality_counts['missing']}")
        
        return results

    # Generates evaluation text based on the answer quality and similarity scores
    def _get_evaluation_text_for_quality(self, quality: str, answer_similarity: float, question_similarity: float) -> str:
        """
        Generate evaluation text based on the quality level and similarity scores.
        
        Args:
            quality: Quality level (high, medium, low, poor, missing)
            answer_similarity: Similarity score between submission and ideal answers
            question_similarity: Similarity score between submission and ideal questions
            
        Returns:
            Evaluation text
        """
        # Format similarity scores for display
        a_sim = f"{answer_similarity:.2f}"
        q_sim = f"{question_similarity:.2f}"
        
        if quality == "high":
            return f"Excellent answer that closely matches the reference. The content is comprehensive and accurate. (A-sim: {a_sim}, Q-sim: {q_sim})"
        
        elif quality == "medium":
            return f"Good answer that covers the main points of the reference, but may be missing some details or nuance. (A-sim: {a_sim}, Q-sim: {q_sim})"
        
        elif quality == "low":
            return f"Partial answer that addresses some aspects of the question, but misses important details or contains inaccuracies. (A-sim: {a_sim}, Q-sim: {q_sim})"
        
        elif quality == "poor":
            return f"Poor answer that does not adequately address the question or contains significant errors. (A-sim: {a_sim}, Q-sim: {q_sim})"
        
        elif quality == "missing":
            return "This question was not addressed in the submission."
        
        else:
            return f"Unable to evaluate this answer (unknown quality level: {quality})." 



    # Establishes connection to MongoDB database for storing and retrieving embeddings
    def connect_mongodb(self, mongodb_uri=None, db_name="assignment_checker"):
        """Connect to MongoDB and initialize collections."""
        # Use provided URI or get from environment
        if not mongodb_uri:
            mongodb_uri = os.getenv("MONGODB_URI")
        
        if not mongodb_uri:
            logger.warning("No MongoDB URI provided and MONGODB_URI not found in environment")
            return False
        
        try:
            # Connect to MongoDB
            self.mongodb_client = MongoClient(mongodb_uri)
            
            # Try a simple operation to verify connection
            self.mongodb_client.admin.command('ping')
            
            # Set up collections
            db = self.mongodb_client[db_name]
            self.qa_embeddings = db["qa_embeddings"]
            self.student_submissions = db["student_submissions"]
            self.counters = db["counters"]
            
            # Create indexes for efficient querying - make submission_id non-unique
            # Drop the existing unique index if it exists
            try:
                self.student_submissions.drop_index("submission_id_1")
                logger.info("Dropped existing unique index on submission_id")
            except Exception as e:
                logger.debug(f"No existing index to drop: {e}")
            
            # Create a non-unique index on submission_id
            self.student_submissions.create_index([("submission_id", 1)])
            
            # Create a compound index on submission_id and qa_id
            self.student_submissions.create_index([("submission_id", 1), ("qa_id", 1)], unique=True)
            
            self.student_submissions.create_index([("timestamp", -1)])
            
            # Initialize counter for student submission IDs if it doesn't exist
            if self.counters.count_documents({"_id": "student_submission_id"}) == 0:
                self.counters.insert_one({"_id": "student_submission_id", "value": 0})
                logger.info("Initialized student submission ID counter")
            
            logger.info(f"Connected to MongoDB database: {db_name}")
            return True
        except Exception as e:
            # Don't include the URI details in the error message
            logger.error(f"Failed to connect to MongoDB: Connection error")
            logger.debug(f"MongoDB connection error details: {e}")
            self.mongodb_client = None
            self.qa_embeddings = None
            self.student_submissions = None
            self.counters = None
            return False

    # Safely truncates text to avoid exceeding token limits while preserving readability
    def safe_truncate_text(self, text: str, max_length: int = 1500) -> str:
        """
        Safely truncate text to avoid exceeding token limits while preserving readability.
        
        Args:
            text: Input text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text with indicator if truncation occurred
        """
        if not text:
            return ""
        
        # If text is already short enough, return it as is
        if len(text) <= max_length:
            return text
        
        # Otherwise, truncate and add an indicator
        truncated_text = text[:max_length]
        
        # Try to truncate at a sentence boundary
        last_period = truncated_text.rfind('.')
        last_question = truncated_text.rfind('?')
        last_exclamation = truncated_text.rfind('!')
        
        # Find the latest sentence boundary
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        # If we found a sentence boundary that's not too far from the truncation point, use it
        if last_sentence_end > max_length * 0.8:
            truncated_text = truncated_text[:last_sentence_end + 1]
        
        # Add truncation indicator
        return truncated_text + " [...text truncated...]" 

    # Formats evaluation data for UI display with structured information
    def _format_evaluations_for_ui(self, qa_mappings, submission_qa_pairs, ideal_qa_pairs):
        """Format mappings into a list of evaluations for UI display."""
        evaluations = []
        
        for mapping in qa_mappings:
            try:
                quality = mapping.get("quality", "poor")
                
                if quality == "missing":
                    # For missing answers, create an evaluation with empty student answers
                    evaluations.append({
                                        "question": mapping.get("ideal_question", ""),
                        "student_answer": "[No answer provided]",
                                        "reference_answer": mapping.get("ideal_answer", ""),
                        "quality": "missing",
                        "similarity": 0.0,
                        "question_similarity": 0.0,
                        "combined_score": 0.0,
                        "numerical_score": 0,
                        "key_concepts_present": [],
                        "key_concepts_missing": ["All concepts missing"],
                        "feedback": "This question was not answered in your submission."
                    })
                else:
                    # For answered questions, include all the details
                    # Get the student_qa_id which should be available
                    student_qa_id = mapping.get("student_qa_id")
                    ideal_qa_id = mapping.get("ideal_qa_id")
                                
                    if not student_qa_id or student_qa_id not in submission_qa_pairs:
                        logger.warning(f"No valid student_qa_id found in mapping: {mapping}")
                        continue
                                
                    # Use student_qa_id as the key for submission_qa_pairs
                    sub_id = student_qa_id
                                
                    answer_similarity = mapping.get("answer_similarity", 0.0)
                    question_similarity = mapping.get("question_similarity", 0.0)
                                
                    # Get the ideal answer
                    ideal_answer = mapping.get("ideal_answer", "")
                    if not ideal_answer and ideal_qa_id and ideal_qa_id in ideal_qa_pairs:
                        ideal_answer = ideal_qa_pairs[ideal_qa_id].get("answer", "")
                                
                    # Get the submission question
                    submission_question = submission_qa_pairs[sub_id].get("question", "")
                    
                    # Generate the detailed feedback
                    feedback = self._generate_answer_feedback(
                    submission_qa_pairs[sub_id]["answer"],
                                        ideal_answer,
                        quality
                    )
                    
                    # Get the combined similarity if available
                    student_answer = submission_qa_pairs[sub_id]["answer"]
                    similarity_metrics = self.compare_answers(student_answer, ideal_answer)
                    combined_similarity = similarity_metrics.get("combined_similarity", answer_similarity)
                                
                        # Parse numerical score from feedback if available
                    numerical_score = None
                    key_concepts_present = []
                    key_concepts_missing = []
                    
                    try:
                        # Try to extract enhanced feedback components
                        lines = feedback.split('\n')
                        for line in lines:
                            line = line.strip()
                            
                            # Extract numerical score
                            if line.startswith("Numerical Score:"):
                                score_text = line.replace("Numerical Score:", "").strip()
                                if score_text.isdigit():
                                    numerical_score = int(score_text)
                            
                            # Extract key concepts
                            if line.startswith("Key Concepts Present:"):
                                concepts_text = line.replace("Key Concepts Present:", "").strip()
                                # Split the concepts, accounting for list formatting [item1, item2]
                                if concepts_text.startswith("[") and concepts_text.endswith("]"):
                                    concepts_text = concepts_text[1:-1]
                                key_concepts_present = [c.strip() for c in concepts_text.split(',') if c.strip()]
                                
                            if line.startswith("Key Concepts Missing:"):
                                concepts_text = line.replace("Key Concepts Missing:", "").strip()
                                # Split the concepts, accounting for list formatting [item1, item2]
                                if concepts_text.startswith("[") and concepts_text.endswith("]"):
                                    concepts_text = concepts_text[1:-1]
                                key_concepts_missing = [c.strip() for c in concepts_text.split(',') if c.strip()]
                    except Exception as e:
                        logger.warning(f"Error parsing enhanced feedback: {e}")
                    
                    # If numerical score wasn't found, provide a default based on quality
                    if numerical_score is None:
                        if quality == "high":
                            numerical_score = int(answer_similarity * 100)
                        elif quality == "medium":
                            numerical_score = int(answer_similarity * 90)
                        elif quality == "low":
                            numerical_score = int(answer_similarity * 75)
                        else:
                            numerical_score = int(answer_similarity * 50)
                    
                    # Use answer similarity as the primary score displayed
                    evaluations.append({
                                    "question": submission_question,
                        "student_answer": submission_qa_pairs[sub_id]["answer"],
                                    "reference_answer": ideal_answer,
                        "quality": quality,
                        "similarity": answer_similarity,  # Primary score is answer similarity
                        "question_similarity": question_similarity,
                                    "combined_score": combined_similarity,  # For consistency
                        "numerical_score": numerical_score,
                        "key_concepts_present": key_concepts_present,
                        "key_concepts_missing": key_concepts_missing,
                        "feedback": feedback
                    })
            except Exception as e:
                logger.error(f"Error formatting question evaluation: {e}")
                continue
        
        return evaluations

        

    # Logs detailed similarity metrics for sample mappings to aid in debugging and tuning thresholds
    def _log_similarity_samples(self, mappings: List[Dict[str, Any]], sample_size: int = 3) -> None:
        """
        Log detailed similarity metrics for a sample of mappings for better analysis.
        
        This helps with debugging and tuning the similarity thresholds by providing
        context for how mappings are being created between submission and ideal Q&A pairs.
        
        Args:
            mappings: List of mappings between submission and ideal Q&A pairs
            sample_size: Number of random samples to log
        """
        import random
        
        if not mappings:
            logger.info("No mappings available to sample - nothing to log")
            return
        
        # Filter to only include mappings with submission_id
        valid_mappings = [m for m in mappings if m.get("submission_id") is not None]
        
        if not valid_mappings:
            logger.warning("No valid mappings with submission_id found for sampling")
            return
        
        # Determine number of samples - use smallest value between sample_size and available valid mappings
        sample_count = min(sample_size, len(valid_mappings))
        
        try:
            # Sample a few mappings to log details about
            samples = random.sample(valid_mappings, sample_count)
            
            logger.info(f"===== SIMILARITY SAMPLE ({sample_count} mappings) =====")
            for idx, mapping in enumerate(samples, 1):
                sub_id = mapping.get("submission_id", "unknown")
                ideal_id = mapping.get("ideal_id", "unknown")
                
                # Get all similarity scores
                q_sim = mapping.get("question_similarity", 0)
                a_sim = mapping.get("answer_similarity", 0)
                e_sim = mapping.get("embedding_similarity", 0)
                t_sim = mapping.get("text_similarity", 0)
                o_sim = mapping.get("token_overlap", 0)
                c_sim = mapping.get("combined_similarity", 0)
                quality = mapping.get("quality", "unknown").upper()
                
                # Get snippets of questions and answers (first 80 chars)
                sub_question = mapping.get("submission_question", "")
                sub_question_preview = sub_question[:80] + "..." if len(sub_question) > 80 else sub_question
                
                sub_answer = mapping.get("submission_answer", "")
                sub_answer_preview = sub_answer[:80] + "..." if len(sub_answer) > 80 else sub_answer
                
                ideal_question = mapping.get("ideal_question", "")
                ideal_question_preview = ideal_question[:80] + "..." if len(ideal_question) > 80 else ideal_question
                
                ideal_answer = mapping.get("ideal_answer", "")
                ideal_answer_preview = ideal_answer[:80] + "..." if len(ideal_answer) > 80 else ideal_answer
                
                # Log detailed similarity information
                logger.info(f"Sample {idx}: {sub_id} -> {ideal_id} (Quality: {quality})")
                logger.info(f"  Submission Question: {sub_question_preview}")
                logger.info(f"  Ideal Question: {ideal_question_preview}")
                logger.info(f"  Similarity Metrics:")
                logger.info(f"    - Question Similarity:     {q_sim:.4f}")
                logger.info(f"    - Answer Similarity:       {a_sim:.4f} (combined)")
                logger.info(f"    - Embedding Similarity:    {e_sim:.4f}")
                logger.info(f"    - Text Similarity:         {t_sim:.4f}")
                logger.info(f"    - Token Overlap:           {o_sim:.4f}")
                logger.info(f"    - Combined Similarity:     {c_sim:.4f}")
                logger.info(f"  Student Answer:   {sub_answer_preview}")
                logger.info(f"  Ideal Answer:     {ideal_answer_preview}")
                logger.info("-----")
            
            logger.info("=======================================")
        except ValueError as e:
            # This happens if sample_count is larger than the population
            logger.warning(f"Could not sample mappings: {e}")
        except Exception as e:
            logger.warning(f"Error while logging similarity samples: {e}")
            # Continue execution even if logging fails

    def log_evaluation_stats(self, eval_id: str, stats: Dict[str, Any]) -> None:
        """
        Log detailed evaluation statistics for analysis and debugging.
        
        Args:
            eval_id: Unique identifier for the evaluation
            stats: Dictionary containing evaluation statistics
        """
        logger.info(f"===== EVALUATION STATS ({eval_id}) =====")
        logger.info(f"Total questions: {stats.get('total_questions', 0)}")
        logger.info(f"Matched questions: {stats.get('matched', 0)}")
        logger.info(f"Missing questions: {stats.get('missing', 0)}")
        logger.info(f"Quality distribution:")
        logger.info(f"  - High: {stats.get('high_count', 0)}")
        logger.info(f"  - Medium: {stats.get('medium_count', 0)}")
        logger.info(f"  - Low: {stats.get('low_count', 0)}")
        logger.info(f"  - Poor: {stats.get('poor_count', 0)}")
        logger.info(f"Overall score: {stats.get('overall_score', 0)}%")
        logger.info(f"====================================")

    def log_embedding_info(self, embedding_type: str, text_sample: str, embedding: np.ndarray) -> None:
        """
        Log information about embeddings for debugging and diagnostics.
        
        Args:
            embedding_type: Type of embedding (question, answer)
            text_sample: Sample of the text being embedded (truncated)
            embedding: The embedding vector
        """
        # Ensure embedding is a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Calculate embedding stats
        norm = np.linalg.norm(embedding)
        mean = np.mean(embedding)
        std = np.std(embedding)
        min_val = np.min(embedding)
        max_val = np.max(embedding)
        
        # Create a short sample of the text (first 50 chars)
        text_preview = text_sample[:50] + "..." if len(text_sample) > 50 else text_sample
        
        # Log the embedding stats
        logger.debug(f"Embedding [{embedding_type}] - Text: '{text_preview}'")
        logger.debug(f"  - Shape: {embedding.shape}, Norm: {norm:.4f}")
        logger.debug(f"  - Stats: Mean={mean:.4f}, StdDev={std:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")

    def log_qa_extraction_results(self, qa_pairs: Dict[str, Dict[str, Any]], source_file: str) -> None:
        """
        Log summary of Q&A pair extraction for analysis purposes.
        
        Args:
            qa_pairs: Dictionary of extracted Q&A pairs
            source_file: Path to the source document
        """
        if not qa_pairs:
            logger.warning(f"No Q&A pairs extracted from {source_file}")
            return
        
        # Log summary stats
        qa_count = len(qa_pairs)
        logger.info(f"Extracted {qa_count} Q&A pairs from {source_file}")
        
        # Calculate average lengths
        q_lengths = [len(qa.get("question", "")) for qa in qa_pairs.values()]
        a_lengths = [len(qa.get("answer", "")) for qa in qa_pairs.values()]
        
        avg_q_length = sum(q_lengths) / qa_count if qa_count > 0 else 0
        avg_a_length = sum(a_lengths) / qa_count if qa_count > 0 else 0
        
        logger.info(f"  - Average question length: {avg_q_length:.1f} chars")
        logger.info(f"  - Average answer length: {avg_a_length:.1f} chars")
        
        # Log first few Q&A pairs as samples
        sample_count = min(3, qa_count)
        logger.info(f"Sample of {sample_count} Q&A pairs:")
        
        for i, (qa_id, qa_pair) in enumerate(list(qa_pairs.items())[:sample_count]):
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            # Truncate for readability
            q_preview = question[:80] + "..." if len(question) > 80 else question
            a_preview = answer[:80] + "..." if len(answer) > 80 else answer
            
            logger.info(f"  [{i+1}] Question: {q_preview}")
            logger.info(f"      Answer: {a_preview}")

    def retrieve_student_embeddings(self, submission_id: int) -> Dict[str, Dict[str, Any]]:
        """Retrieve student embeddings from MongoDB for a specific submission_id.
        
        Args:
            submission_id: The submission ID to retrieve
            
        Returns:
            Dictionary of student Q&A pairs with embeddings
        """
        try:
            logger.info(f"Retrieving student embeddings for submission_id: {submission_id}")
            
            # Query MongoDB for student submissions with this submission_id
            cursor = self.student_submissions.find({"submission_id": submission_id})
            
            student_qa_pairs = {}
            count = 0
            
            for doc in cursor:
                qa_id = doc.get("qa_id")
                embedding = doc.get("embedding")  # This is the answer embedding
                question_embedding = doc.get("question_embedding")
                
                if not qa_id or embedding is None:
                    logger.warning(f"Missing qa_id or embedding for student document in MongoDB, skipping")
                    continue
                    
                # Store all the data
                student_qa_pairs[qa_id] = {
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "answer_embedding": embedding,  # Map the "embedding" field to "answer_embedding"
                    "embedding": embedding,  # Keep the original field for backward compatibility
                    "question_embedding": question_embedding
                }
                
                count += 1
            
            logger.info(f"Retrieved {count} student Q&A pairs from MongoDB for submission_id: {submission_id}")
            
            return student_qa_pairs
        except Exception as e:
            logger.error(f"Error retrieving student embeddings for submission_id {submission_id}: {e}")
            logger.error(traceback.format_exc())
            return {}

    def generate_submission_id(self) -> int:
        """Generate a new incremental submission ID from the counters collection.
        
        Returns:
            The new submission ID
        """
        try:
            result = self.counters.find_one_and_update(
                {"_id": "student_submission_id"},
                {"$inc": {"value": 1}},
                return_document=True
            )
            submission_id = result["value"]
            logger.info(f"Generated new submission ID: {submission_id}")
            return submission_id
        except Exception as e:
            logger.error(f"Error generating submission ID: {e}")
            logger.error(traceback.format_exc())
            # Return a random ID as fallback
            fallback_id = int(time.time())
            logger.warning(f"Using fallback submission ID: {fallback_id}")
            return fallback_id

    def save_qa_pairs_to_json(self, qa_pairs: Dict[str, Dict[str, Any]], file_path: str) -> bool:
        """Save question-answer pairs to a local JSON file.
        
        Args:
            qa_pairs: Dictionary of question-answer pairs
            file_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data for JSON serialization
            json_data = {}
            for qa_id, qa_data in qa_pairs.items():
                # Convert numpy arrays to lists for JSON serialization
                serializable_qa = {}
                for key, value in qa_data.items():
                    if isinstance(value, np.ndarray):
                        serializable_qa[key] = value.tolist()
                    elif hasattr(value, 'tolist'):  # Handle other array-like objects
                        serializable_qa[key] = value.tolist()
                    else:
                        serializable_qa[key] = value
                json_data[qa_id] = serializable_qa
            
            # Write to file with indentation for readability
            import json
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Q&A pairs to JSON file {file_path}: {e}")
            logger.error(traceback.format_exc())
            return False

    def setup_qa_directories(self):
        """Set up the directory structure for storing Q&A JSON files."""
        try:
            # Create main qa_files directory
            os.makedirs(self.qa_files_dir, exist_ok=True)
            
            # Create subdirectories for ideal and student submissions and evaluations
            ideal_dir = os.path.join(self.qa_files_dir, "ideal")
            student_dir = os.path.join(self.qa_files_dir, "student")
            evaluations_dir = os.path.join(self.qa_files_dir, "evaluations")
            
            os.makedirs(ideal_dir, exist_ok=True)
            os.makedirs(student_dir, exist_ok=True)
            os.makedirs(evaluations_dir, exist_ok=True)
            
            logger.info(f"Set up Q&A file directories at {self.qa_files_dir}")
            return True
        except Exception as e:
            logger.error(f"Error setting up Q&A directories: {e}")
            logger.error(traceback.format_exc())
            return False

    def compare_answers(self, student_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Compare student answer with reference answer and return similarity metrics.
        
        Args:
            student_answer: The student's submitted answer
            reference_answer: The reference/ideal answer
            
        Returns:
            Dictionary with similarity metrics
        """
        import difflib
        import numpy as np
        from collections import Counter
        
        # Calculate embedding similarity if possible
        try:
            # Generate embeddings
            student_embedding = self.generate_embedding(student_answer)
            reference_embedding = self.generate_embedding(reference_answer)
            
            # Calculate cosine similarity
            embedding_similarity = self._compute_similarity(student_embedding, reference_embedding)
        except Exception as e:
            logger.warning(f"Error calculating embedding similarity: {e}")
            embedding_similarity = 0.0
        
        # Calculate text similarity using difflib
        try:
            text_similarity = difflib.SequenceMatcher(None, student_answer, reference_answer).ratio()
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            text_similarity = 0.0
        
        # Calculate token overlap
        try:
            # Tokenize answers (simple whitespace tokenization for efficiency)
            student_tokens = Counter(student_answer.lower().split())
            reference_tokens = Counter(reference_answer.lower().split())
            
            # Calculate overlap
            common_tokens = sum((student_tokens & reference_tokens).values())
            total_tokens = sum(reference_tokens.values())
            
            token_overlap = common_tokens / total_tokens if total_tokens > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating token overlap: {e}")
            token_overlap = 0.0
        
        # Calculate combined similarity with specified weights
        # 50% embedding similarity, 40% text similarity, 10% token overlap
        combined_similarity = (
            0.3 * embedding_similarity +
            0.5 * text_similarity +
            0.2 * token_overlap
        )
        
        # Return all metrics
        return {
            "embedding_similarity": float(embedding_similarity),
            "text_similarity": float(text_similarity),
            "token_overlap": float(token_overlap),
            "combined_similarity": float(combined_similarity)
        }