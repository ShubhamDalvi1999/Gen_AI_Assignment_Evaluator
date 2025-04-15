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
        self.connect_mongodb(mongodb_uri, db_name)
        
        # Cache for ideal Q&A pairs
        self.ideal_qa_pairs = {}
        
        # Set similarity thresholds - adjusted to account for higher embedding weights
        # With higher embedding weights, answers with good semantic similarity will score higher
        self.high_quality_threshold = 0.85  # High quality threshold reduced from 0.92
        self.medium_quality_threshold = 0.75  # Medium quality threshold reduced from 0.80
        self.low_quality_threshold = 0.60  # Low quality threshold reduced from 0.65
        
        # For backward compatibility
        self.similarity_thresholds = {
            "high": self.high_quality_threshold,
            "medium": self.medium_quality_threshold,
            "low": self.low_quality_threshold
        }
        
        # Add simplified threshold naming for easier code readability
        self.sim_threshold_high = self.high_quality_threshold
        self.sim_threshold_medium = self.medium_quality_threshold
        self.sim_threshold_low = self.low_quality_threshold
        
        # API keys and model settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        
        # Use OpenAI or Ollama for embeddings
        self.use_openai = use_openai
        logger.info(f"Initialized TextRAGProcessor with {'OpenAI' if use_openai else 'Ollama'} for embeddings")
        
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
                          embedding: np.ndarray, is_ideal: bool = True, question_embedding: np.ndarray = None) -> None:
        """Store Q&A embedding in MongoDB.
        
        Args:
            qa_id: Identifier for the Q&A pair
            question: The question text
            answer: The answer text
            embedding: The ANSWER embedding
            is_ideal: Whether this is from the ideal document
            question_embedding: The QUESTION embedding (if available)
        """
        try:
            logger.info("========== DOCUMENT EMBEDDING STORAGE STAGE ==========")
            logger.info(f"Storing embeddings for QA pair {qa_id} (is_ideal: {is_ideal})")
            
            # Check if embedding is a numpy array or already a list
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            # Prepare document
            document = {
                "qa_id": qa_id,
                "question": question,
                "answer": answer,
                "embedding": embedding_list,  # This is the ANSWER embedding
                "embedding_type": "answer",  # Explicitly mark this as an answer embedding
                "is_ideal": is_ideal,
                "timestamp": datetime.now()
            }
            
            # Add question embedding if provided
            if question_embedding is not None:
                question_embedding_list = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
                document["question_embedding"] = question_embedding_list
            
            self.qa_embeddings.insert_one(document)
            logger.info(f"Successfully stored embeddings for QA pair {qa_id}")
            logger.info(f"Stored embeddings for QA pair {qa_id} (with question embedding: {question_embedding is not None})")
        except Exception as e:
            logger.error(f"Error storing embedding for Q&A pair {qa_id}: {e}")
    
    # Clears all embedding records from MongoDB to reset the system
    def clear_embeddings(self) -> None:
        """Clear all embeddings from MongoDB."""
        self.qa_embeddings.delete_many({})
    

    # Calculates cosine similarity between two embeddings for comparing text semantic similarity
    def compute_similarity(self, emb1, emb2) -> float:
        """
        Compute cosine similarity between two embeddings.
        Handles both numpy arrays and lists.
        """
        import numpy as np
        
        # Convert to numpy arrays if they're lists
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        
        # Ensure we're working with 1D arrays
        if len(emb1.shape) > 1:
            emb1 = emb1.flatten()
        if len(emb2.shape) > 1:
            emb2 = emb2.flatten()
        
        # Compute cosine similarity
        similarity = float(1 - cosine(emb1, emb2))
        
        # Ensure the result is a python float
        return float(similarity)
    
    # 2. Processes a DOCX file to extract Q&A pairs and store embeddings.
    def process_qa_document(self, file_path: str, is_ideal: bool = True) -> Dict[str, Dict[str, Any]]:
        """Process a DOCX file to extract Q&A pairs and store embeddings.
         
        Args:
            file_path: The path to the DOCX file
            is_ideal: Whether this is from the ideal document
            
        Returns:
            Dictionary of question-answer pairs
        """
        # 2.1. Extract text from document
        text = DocxProcessor.extract_text_from_docx(file_path)
        
        if not text:
            logger.error(f"No text extracted from {file_path}")
            return {}
        
        # 2.2.1 Extract Q&A pairs using regex
        qa_pairs = DocxProcessor.extract_qa_pairs(text)
        
        # 2.2.2 If no Q&A pairs found, try with LLM-based extraction
        if not qa_pairs:
            logger.info(f"No Q&A pairs found with regex, trying LLM extraction for {file_path}")
            qa_pairs = self.extract_qa_pairs_with_llm(text)
        
        # 2.3. Process each Q&A pair for embeddings
        qa_count = len(qa_pairs)
        logger.info(f"Extracted {qa_count} Q&A pairs from document")
        
        # Log which model we're using for embeddings
        if self.use_openai:
            logger.info(f"Using OpenAI for generating embeddings for {qa_count} Q&A pairs")
        else:
            logger.info(f"Using Ollama for generating embeddings for {qa_count} Q&A pairs")
        
        # Clear cache if OpenAI is selected to ensure fresh embeddings for each evaluation
        if self.use_openai:
            global _QUESTION_EMBEDDING_CACHE
            # We only clear OpenAI cache entries
            openai_keys = [k for k in _QUESTION_EMBEDDING_CACHE.keys() if k.startswith('openai_embed_')]
            for key in openai_keys:
                del _QUESTION_EMBEDDING_CACHE[key]
            logger.info(f"Cleared {len(openai_keys)} OpenAI embedding cache entries for fresh evaluation")
        
        # 2.4. Process each Q&A pair for embeddings and store embeddings in MongoDB
        for qa_id, qa_pair in qa_pairs.items():
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            # Skip if question or answer is empty
            if not question or not answer:
                logger.warning(f"Skipping {qa_id} due to empty question or answer")
                continue
            
            try:
                # d.1Generate embeddings for the Q&A pair

                # Generate embeddings for the answer
                if self.use_openai:
                    # Use bypass_cache=True to force a new API call every time for accurate scoring
                    answer_embedding = self.generate_embedding_openai(answer, bypass_cache=True)
                    logger.info(f"Generated new OpenAI embedding via API call for ANSWER in {qa_id}")
                else:
                    answer_embedding = self.generate_embedding(answer)
                    logger.debug(f"Generated Ollama embedding for ANSWER in {qa_id}")
                
                # Generate embeddings for the question
                if self.use_openai:
                    question_embedding = self.generate_embedding_openai(question, bypass_cache=True)
                    logger.info(f"Generated new OpenAI embedding via API call for QUESTION in {qa_id}")
                else:
                    question_embedding = self.generate_embedding(question)
                    logger.debug(f"Generated Ollama embedding for QUESTION in {qa_id}")
                
                # Check if we got valid embeddings
                if answer_embedding is None or (isinstance(answer_embedding, np.ndarray) and answer_embedding.size == 0):
                    logger.warning(f"Invalid answer embedding generated for {qa_id} - skipping")
                    continue
                    
                if question_embedding is None or (isinstance(question_embedding, np.ndarray) and question_embedding.size == 0):
                    logger.warning(f"Invalid question embedding generated for {qa_id} - skipping")
                    continue
                    
                # d.2 Store Q&A pair with embeddings only if it's an ideal Q&A pair
                if is_ideal:  # Only store ideal embeddings in DB
                    # Store both the answer and question embeddings in MongoDB
                    self.store_qa_embedding(
                        qa_id, 
                        question, 
                        answer, 
                        answer_embedding, 
                        is_ideal=True,
                        question_embedding=question_embedding
                    )
                    logger.info(f"Stored embeddings for ideal Q&A pair {qa_id}")
                
                # Add embeddings to the qa_pair for return
                answer_embedding_list = answer_embedding.tolist() if hasattr(answer_embedding, 'tolist') else answer_embedding
                question_embedding_list = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
                
                # Store both ways for backward compatibility during transition
                qa_pairs[qa_id]["answer_embedding"] = answer_embedding_list
                qa_pairs[qa_id]["embedding"] = answer_embedding_list  # Keep old field for backward compatibility
                qa_pairs[qa_id]["question_embedding"] = question_embedding_list
                
            except Exception as e:
                logger.error(f"Error processing embedding for Q&A pair {qa_id}: {e}")
                qa_pairs[qa_id]["embedding_error"] = str(e)
        
        return qa_pairs
    

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
            
            # 2.2 Process submission document
            logger.info(f"Processing submission document: {submission_path}")
            submission_qa_pairs = self.process_qa_document(submission_path, is_ideal=False)
            
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
            
            # Clean the mappings to ensure they can be JSON serialized
            cleaned_mappings = self._clean_for_json(qa_mappings)
            
            # Format the response to match what the UI expects
            result = {
                "status": "success",
                "eval_id": eval_id,
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
    def _map_qa_pairs(self, submission_qa_pairs: Dict[str, Dict[str, Any]], emphasize_embedding=True) -> List[Dict[str, Any]]:
        """Map submission Q&A pairs to ideal Q&A pairs using embeddings. 
        This method performs a two-phase matching:
        1. First it matches questions based on similarity to find the right pairing
        2. Then it assesses answer quality using detailed comparison metrics
        
        Args:
            submission_qa_pairs: Dictionary of submission Q&A pairs
            emphasize_embedding: If True, emphasizes embedding similarity over other metrics
            
        Returns:
            List of mappings between submission and ideal Q&A pairs
        """
        logger.info("========== RETRIEVAL STAGE ==========")
        logger.info(f"Mapping submission QA pairs to ideal QA pairs")
        
       
        
        # 3.1 Retrieve ideal Q&A pairs from MongoDB and student submission are passed as a parameter
        # 3.2 Check for missing question embeddings in ideal Q&A pairs and generate them if needed
        # 3.3 First phase - match questions based on similarity
        # 3.4 Second phase - assess answer quality using comprehensive comparison between submission and ideal mapping
        # 3.5 Extract similarity metrics
        # 3.6 Determine quality level based on combined similarity
        # 3.7 Create the mapping
        
        ideal_qa_pairs = self.retrieve_ideal_embeddings()
        logger.info(f"Retrieved {len(ideal_qa_pairs)} ideal Q&A pairs from MongoDB ({sum(1 for qa in ideal_qa_pairs.values() if 'question_embedding' in qa)} with question embeddings)")
        
        if not ideal_qa_pairs:
            logger.error("No ideal Q&A pairs found in the database")
            return []
        
        # Map student submissions to ideal answers
        logger.info(f"Mapping {len(submission_qa_pairs)} submissions to {len(ideal_qa_pairs)} ideal Q&A pairs")
        
        # Decide whether to emphasize embedding similarity for this evaluation
        if emphasize_embedding:
            logger.info("Using embedding-focused similarity for answer scoring")
            
        # Prepare for mapping
        mappings = []
        
        # Initialize collections for tracking best matches
        mapped_submission_ids = set()
        mapped_ideal_ids = set()
        
        logger.info("========== QUESTION MAPPING STAGE ==========")
        logger.info("Finding best matches between student and ideal questions")
        
        # Track which ideal questions have been matched to avoid duplicates
        matched_ideal_ids = set()
        
        # 3.2 Check for missing question embeddings in ideal Q&A pairs and generate them if needed
        self._ensure_question_embeddings(ideal_qa_pairs)
        
        # 3.3 First phase - match questions based on similarity
        for sub_id, sub_qa in submission_qa_pairs.items():
            submission_question = sub_qa.get("question", "")
            submission_answer = sub_qa.get("answer", "")
            
            if not submission_question or not submission_answer:
                logger.warning(f"Skipping submission {sub_id} - missing question or answer")
                continue
            
            if "question_embedding" not in sub_qa:
                logger.warning(f"Submission {sub_id} missing question embedding, generating it now")
                try:
                    # Generate question embedding on the fly
                    question_embedding = self.generate_embedding(submission_question)
                    sub_qa["question_embedding"] = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
                except Exception as e:
                    logger.error(f"Failed to generate question embedding for {sub_id}: {e}")
                    continue
            
            # Find the most similar ideal question
            best_match_id = None
            best_match_similarity = -1
            
            # 3.3.1 For each ideal Q&A pair, calculate the similarity between the submission question and the ideal question
            for ideal_id, ideal_qa in ideal_qa_pairs.items():

                # if question embedding is missing, generate it on the fly
                if "question_embedding" not in ideal_qa:
                    logger.warning(f"Ideal question {ideal_id} missing embedding, generating it now")
                    try:
                        # Generate question embedding on the fly
                        ideal_question = ideal_qa.get("question", "")
                        question_embedding = self.generate_embedding(ideal_question)
                        ideal_qa["question_embedding"] = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
                        
                        # Store in database for future use
                        self.qa_embeddings.update_one(
                            {"qa_id": ideal_id, "is_ideal": True},
                            {"$set": {"question_embedding": ideal_qa["question_embedding"]}}
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate question embedding for ideal {ideal_id}: {e}")
                        continue
                
                # 3.3.2 Calculate similarity between question embeddings
                question_similarity = self._calculate_similarity(
                    sub_qa["question_embedding"], 
                    ideal_qa["question_embedding"]
                )
                
                # Update best match if this is better
                if question_similarity > best_match_similarity:
                    best_match_similarity = question_similarity
                    best_match_id = ideal_id
            
            # 3.4 If we found a match and the question similarity is above the threshold
            # then we can proceed to the second phase
            if best_match_id and best_match_similarity > 0.5:  # Question similarity threshold
                ideal_qa = ideal_qa_pairs[best_match_id]
                ideal_question = ideal_qa.get("question", "")
                ideal_answer = ideal_qa.get("answer", "")
                
                logger.debug(f"Matched submission {sub_id} to ideal {best_match_id} (question sim: {best_match_similarity:.4f})")
                
                # 3.4 Second phase - assess answer quality using comprehensive comparison
                comparison_results = self.compare_answers(submission_answer, ideal_answer, emphasize_embedding=emphasize_embedding)
                
                # 3.5 Extract similarity metrics
                embedding_similarity = comparison_results["embedding_similarity"]
                text_similarity = comparison_results["text_similarity"]
                token_overlap = comparison_results["token_overlap"]
                combined_similarity = comparison_results["combined_similarity"]
                
                # Determine quality level based on combined similarity
                quality = "poor"
                if combined_similarity >= self.sim_threshold_high:
                    quality = "high"
                elif combined_similarity >= self.sim_threshold_medium:
                    quality = "medium"
                elif combined_similarity >= self.sim_threshold_low:
                    quality = "low"
                
                # Create the mapping
                mapping = {
                    "submission_id": sub_id,
                    "ideal_id": best_match_id,
                    "submission_question": submission_question,
                    "submission_answer": submission_answer,
                    "ideal_question": ideal_question,
                    "ideal_answer": ideal_answer,
                    "question_similarity": best_match_similarity,
                    "answer_similarity": combined_similarity,  # Use combined score as the main similarity measure
                    "embedding_similarity": embedding_similarity,
                    "text_similarity": text_similarity,
                    "token_overlap": token_overlap,
                    "quality": quality
                }
                
                mappings.append(mapping)
                matched_ideal_ids.add(best_match_id)
            
        # Add missing ideal questions (those that weren't matched to any submission)
        for ideal_id, ideal_qa in ideal_qa_pairs.items():
            if ideal_id not in matched_ideal_ids:
                # This ideal question wasn't matched to any submission
                ideal_question = ideal_qa.get("question", "")
                ideal_answer = ideal_qa.get("answer", "")
                
                logger.warning(f"Ideal question not answered: {ideal_question[:80]}...")
                
                # Add as a missing question
                mapping = {
                    "submission_id": None,
                    "ideal_id": ideal_id,
                    "submission_question": None,
                    "submission_answer": None,
                    "ideal_question": ideal_question,
                    "ideal_answer": ideal_answer,
                    "question_similarity": 0.0,
                    "answer_similarity": 0.0,
                    "embedding_similarity": 0.0,
                    "text_similarity": 0.0,
                    "token_overlap": 0.0,
                    "quality": "missing"
                }
                
                mappings.append(mapping)
            
        logger.info(f"Created {len(mappings)} mappings between submission and ideal Q&A pairs")
        logger.debug(f"Quality distribution: " + 
                    f"High={len([m for m in mappings if m['quality'] == 'high'])}, " +
                    f"Medium={len([m for m in mappings if m['quality'] == 'medium'])}, " +
                    f"Low={len([m for m in mappings if m['quality'] == 'low'])}, " +
                    f"Poor={len([m for m in mappings if m['quality'] == 'poor'])}, " +
                    f"Missing={len([m for m in mappings if m['quality'] == 'missing'])}")
                
        return mappings
        
    # Validates that all QA pairs have question embeddings and reports status
    def _ensure_question_embeddings(self, qa_pairs: Dict[str, Dict[str, Any]]) -> None:
        """
        Ensures all QA pairs have question embeddings by counting and reporting status.
        
        Args:
            qa_pairs: Dictionary of Q&A pairs
        """
        total = len(qa_pairs)
        with_embeddings = sum(1 for qa in qa_pairs.values() if "question_embedding" in qa)
        
        if with_embeddings < total:
            logger.warning(f"Only {with_embeddings}/{total} ideal QA pairs have question embeddings")
            # Missing embeddings will be handled individually during mapping
    
    # 3.2 Calculates enhanced cosine similarity between two embedding vectors with normalization
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate an enhanced cosine similarity between two vectors.
        
        This uses cosine similarity with additional normalization and 
        non-linear scaling to better differentiate high similarities.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Enhanced similarity score between 0 and 1
        """
        if vec1 is None or vec2 is None:
            logger.warning("Received None vector in similarity calculation")
            return 0.0
        
        # Convert to numpy arrays if not already
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Apply normalization for numerical stability
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("Zero norm vector detected in similarity calculation")
                return 0.0
            
            vec1_normalized = vec1 / norm1
            vec2_normalized = vec2 / norm2
            
            # Calculate raw cosine similarity
            raw_similarity = np.dot(vec1_normalized, vec2_normalized)
            
            # Apply non-linear scaling to emphasize differences in high similarity range
            # This amplifies differences between 0.7-1.0 range and reduces impact of low similarities
            enhanced_similarity = 0.0
            
            if raw_similarity > 0.7:
                # Scale the high similarity range (0.7-1.0) to (0.5-1.0)
                enhanced_similarity = 0.5 + 0.5 * ((raw_similarity - 0.7) / 0.3)
            else:
                # Compress the lower similarity range (0-0.7) to (0-0.5)
                enhanced_similarity = (raw_similarity / 0.7) * 0.5
            
            # Ensure the result is in [0, 1]
            enhanced_similarity = max(0.0, min(1.0, enhanced_similarity))
            
            # Log the calculation for debugging
            if random.random() < 0.05:  # Only log a small percentage of calculations
                logger.debug(f"Similarity: raw={raw_similarity:.4f}, enhanced={enhanced_similarity:.4f}")
            
            return enhanced_similarity
            
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            return 0.0
    
    # 3.3 Comprehensively compares student answers to ideal answers using multiple similarity metrics
    def compare_answers(self, student_answer: str, ideal_answer: str, emphasize_embedding=False) -> Dict[str, Any]:
        """
        Comprehensively compare a student answer to an ideal answer using multiple similarity metrics.
        
        Args:
            student_answer: The student's submitted answer
            ideal_answer: The ideal reference answer
            emphasize_embedding: If True, uses embedding similarity as the primary score
            
        Returns:
            Dict containing similarity metrics, key concepts, and quality assessment
        """
        logger.info("========== AUGMENTATION STAGE ==========")
        logger.info(f"Comparing answers using multiple similarity methods (emphasize_embedding={emphasize_embedding})")
        
        start_time = datetime.now()
        
        # Initialize results
        result = {
            "embedding_similarity": 0.0,
            "text_similarity": 0.0,
            "token_overlap": 0.0,
            "combined_similarity": 0.0
        }
        
        try:
            # Import needed modules locally to ensure they're available in this method
            import difflib
            from collections import Counter
            import re
            
            # Rest of method continues as normal...
            
            # 3.3.1 Normalize text (lower case, remove extra whitespace)
            # 3.3.2 Calculate semantic similarity using embeddings (cosine similarity)
            # 3.3.3 Calculate text similarity using SequenceMatcher (difflib)
            # 3.3.4 Calculate token overlap (Jaccard similarity on words)
            # 3.3.5 Calculate term match score
            # 3.3.6 Calculate combined similarity with weighted contributions
            
            def normalize_text(text):
                text = text.lower()
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            student_normalized = normalize_text(student_answer)
            ideal_normalized = normalize_text(ideal_answer)
            
            # Skip empty answers
            if not student_normalized or not ideal_normalized:
                logger.warning("Empty answer detected in comparison")
                return result
            
            # 3.3.2 Calculate semantic similarity using embeddings
            student_embedding = self.generate_embedding(student_normalized)
            ideal_embedding = self.generate_embedding(ideal_normalized)
            
            if student_embedding is not None and ideal_embedding is not None:
                result["embedding_similarity"] = self._calculate_similarity(
                    student_embedding, ideal_embedding
                )
                
            # 3.3.3 If emphasize_embedding is True, set combined_similarity directly to embedding_similarity
            # and return early with quality assessment
            if emphasize_embedding:
                result["combined_similarity"] = result["embedding_similarity"]
                result["quality"] = self._determine_quality(result["combined_similarity"])
                result["calculation_time"] = (datetime.now() - start_time).total_seconds()
                return result
            
            # 3.3.4 Calculate text similarity using SequenceMatcher
            sequence_similarity = difflib.SequenceMatcher(
                None, student_normalized, ideal_normalized
            ).ratio()
            result["text_similarity"] = sequence_similarity
            
            # 3.3.5 Calculate token overlap (Jaccard similarity on words)
            def tokenize(text):
                # Extract words, remove stopwords and punctuation
                words = re.findall(r'\b[a-z0-9]+\b', text)
                # Filter out very common words and very short words
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                             'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'that'}
                words = [w for w in words if w not in stopwords and len(w) > 2]
                return words
            
            student_tokens = tokenize(student_normalized)
            ideal_tokens = tokenize(ideal_normalized)
            
            # 3.3.6 Calculate Jaccard similarity
            if student_tokens and ideal_tokens:
                student_token_set = set(student_tokens)
                ideal_token_set = set(ideal_tokens)
                
                intersection = len(student_token_set.intersection(ideal_token_set))
                union = len(student_token_set.union(ideal_token_set))
                
                if union > 0:
                    result["token_overlap"] = intersection / union
                
            # Also consider term frequency for key concepts
            student_counts = Counter(student_tokens)
            ideal_counts = Counter(ideal_tokens)
            
            # Get the top N most frequent terms in ideal answer
            top_n = 10
            ideal_key_terms = dict(ideal_counts.most_common(top_n))
            
            # Check how many key terms appear in student answer with similar frequency
            term_match_score = 0
            for term, ref_count in ideal_key_terms.items():
                student_count = student_counts.get(term, 0)
                if student_count > 0:
                    # Score based on how close the frequency is
                    frequency_ratio = min(student_count / ref_count, 1.0) if ref_count > 0 else 0
                    term_match_score += frequency_ratio
                    
            # Normalize term match score
            if ideal_key_terms:
                term_match_score /= len(ideal_key_terms)
                # Blend with token overlap
                result["token_overlap"] = 0.5 * result["token_overlap"] + 0.5 * term_match_score
                
            # Calculate combined similarity with weighted contributions
            # - Embedding similarity is most important (semantic understanding)
            # - Text similarity helps catch structural similarities
            # - Token overlap ensures key concepts are present
            # Increase weight of embedding similarity from 0.6 to 0.8 to emphasize semantic understanding
            result["combined_similarity"] = (
                0.8 * result["embedding_similarity"] +
                0.1 * result["text_similarity"] + 
                0.1 * result["token_overlap"]
            )
            
            # Ensure combined score is in [0, 1]
            result["combined_similarity"] = max(0.0, min(1.0, result["combined_similarity"]))
            
            # Add quality assessment to result
            result["quality"] = self._determine_quality(result["combined_similarity"])
            result["calculation_time"] = (datetime.now() - start_time).total_seconds()
            
            # Debugging - log comparison details occasionally
            if random.random() < 0.1:  # Log ~10% of comparisons
                logger.debug(f"Answer comparison: emb={result['embedding_similarity']:.4f}, " +
                            f"text={result['text_similarity']:.4f}, " +
                            f"token={result['token_overlap']:.4f}, " +
                            f"combined={result['combined_similarity']:.4f}")
            
            # Log performance for monitoring
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Answer comparison completed in {elapsed_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in answer comparison: {e}")
            logger.error(traceback.format_exc())
            
            # On error, return initial empty result with error flag
            result["error"] = str(e)
            return result
    
    def _determine_quality(self, combined_similarity):
        if combined_similarity >= self.high_quality_threshold:
            return "high"
        elif combined_similarity >= self.medium_quality_threshold:
            return "medium"
        elif combined_similarity >= self.low_quality_threshold:
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
            
            logger.info(f"Connected to MongoDB database: {db_name}")
            return True
        except Exception as e:
            # Don't include the URI details in the error message
            logger.error(f"Failed to connect to MongoDB: Connection error")
            logger.debug(f"MongoDB connection error details: {e}")
            self.mongodb_client = None
            self.qa_embeddings = None
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
            quality = mapping.get("quality", "poor")
            
            if quality == "missing":
                # For missing answers, create an evaluation with empty student answers
                evaluations.append({
                    "question": mapping["ideal_question"],
                    "student_answer": "[No answer provided]",
                    "reference_answer": mapping["ideal_answer"],
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
                sub_id = mapping["submission_id"]
                answer_similarity = mapping.get("answer_similarity", 0.0)
                question_similarity = mapping.get("question_similarity", 0.0)
                
                # Generate the detailed feedback
                feedback = self._generate_answer_feedback(
                    submission_qa_pairs[sub_id]["answer"],
                    mapping["ideal_answer"],
                    quality
                )
                
                # Parse numerical score from feedback if available (format: "Numerical Score: 85")
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
                    "question": mapping["submission_question"],
                    "student_answer": submission_qa_pairs[sub_id]["answer"],
                    "reference_answer": mapping["ideal_answer"],
                    "quality": quality,
                    "similarity": answer_similarity,  # Primary score is answer similarity
                    "question_similarity": question_similarity,
                    "combined_score": answer_similarity,  # For consistency, use answer similarity as combined score
                    "numerical_score": numerical_score,
                    "key_concepts_present": key_concepts_present,
                    "key_concepts_missing": key_concepts_missing,
                    "feedback": feedback
                })
        
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