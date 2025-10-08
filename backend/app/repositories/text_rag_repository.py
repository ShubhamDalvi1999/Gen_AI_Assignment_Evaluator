from typing import Dict, Any, List, Optional
import numpy as np
import traceback
from datetime import datetime
import os
import json

from ..repositories.db import (
    get_qa_embeddings_collection,
    get_student_submissions_collection,
    get_counters_collection
)
from ..services.embedding_service import get_embedding, EmbeddingModel
from ..utils.docx_processor import DocxProcessor
from ..core.rag_logging import log_embedding_operation, log_retrieval_operation
from ..core.logging import db_logger as logger


class TextRAGRepository:
    """Repository for text-based RAG operations."""
    
    def __init__(self):
        self.use_openai = False
        
        # Define similarity thresholds
        self.question_similarity_threshold = 0.7
        self.high_quality_threshold = 0.92
        self.medium_quality_threshold = 0.75
        self.low_quality_threshold = 0.60
        
        # Set up local file storage for Q&A pairs
        self.qa_files_dir = os.path.join(os.getcwd(), "qa_files")
        self.setup_qa_directories()
    
    def setup_qa_directories(self):
        """Set up directories for storing Q&A files."""
        try:
            ideal_dir = os.path.join(self.qa_files_dir, "ideal")
            student_dir = os.path.join(self.qa_files_dir, "student")
            evaluations_dir = os.path.join(self.qa_files_dir, "evaluations")
            
            for directory in [ideal_dir, student_dir, evaluations_dir]:
                os.makedirs(directory, exist_ok=True)
                
            logger.info(f"Q&A directories set up at: {self.qa_files_dir}")
        except Exception as e:
            logger.error(f"Error setting up Q&A directories: {e}")
    
    def generate_submission_id(self) -> int:
        """Generate a unique submission ID."""
        try:
            collection = get_counters_collection()
            if collection is not None:
                # Use findOneAndUpdate for atomic increment
                result = collection.find_one_and_update(
                    {"_id": "student_submission_id"},
                    {"$inc": {"value": 1}},
                    upsert=True,
                    return_document=True
                )
                submission_id = result["value"]
                logger.info(f"Generated submission ID: {submission_id}")
                return submission_id
            else:
                logger.warning("MongoDB connection unavailable - using timestamp-based ID")
                return int(datetime.now().timestamp())
        except Exception as e:
            logger.error(f"Error generating submission ID: {e}")
            return int(datetime.now().timestamp())
    
    @log_embedding_operation("qa_embedding_generation")
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the configured model."""
        model = EmbeddingModel.OPENAI if self.use_openai else EmbeddingModel.OLLAMA
        return get_embedding(text, model)
    
    def process_qa_document(self, file_path: str, is_ideal: bool = True, submission_id: int = None) -> Dict[str, Dict[str, Any]]:
        """Process a DOCX file to extract Q&A pairs and store embeddings."""
        
        # Extract text from document
        text = DocxProcessor.extract_text_from_docx(file_path)
        
        if not text:
            logger.error(f"No text extracted from {file_path}")
            return {}
        
        # Extract Q&A pairs from text
        qa_pairs = DocxProcessor.extract_qa_pairs(text)
        
        if not qa_pairs:
            logger.warning(f"No Q&A pairs found in document: {file_path}")
            return {}
            
        # Normalize and deduplicate
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
        
        if is_ideal:
            # Save ideal Q&A pairs
            json_path = os.path.join(self.qa_files_dir, "ideal", f"{base_name}.json")
            self.save_qa_pairs_to_json(processed_qa, json_path)
        else:
            # Save student Q&A pairs with submission ID
            json_path = os.path.join(self.qa_files_dir, "student", f"{base_name}_submission_{submission_id}.json")
            self.save_qa_pairs_to_json(processed_qa, json_path)
        
        return processed_qa
    
    def store_qa_embedding(self, qa_id: str, question: str, answer: str, 
                          answer_embedding: np.ndarray, is_ideal: bool = True,
                          question_embedding: Optional[np.ndarray] = None, 
                          submission_id: Optional[int] = None):
        """Store Q&A embedding in MongoDB."""
        
        try:
            if is_ideal:
                collection = get_qa_embeddings_collection()
            else:
                collection = get_student_submissions_collection()
            
            if collection is not None:
                document = {
                    "qa_id": qa_id,
                    "question": question,
                    "answer": answer,
                    "embedding": answer_embedding.tolist(),
                    "is_ideal": is_ideal,
                    "timestamp": datetime.now()
                }
                
                if question_embedding is not None:
                    document["question_embedding"] = question_embedding.tolist()
                
                if not is_ideal and submission_id is not None:
                    document["submission_id"] = submission_id
                
                result = collection.insert_one(document)
                logger.debug(f"Stored Q&A embedding: {qa_id} (ID: {result.inserted_id})")
            else:
                logger.warning(f"Q&A embedding not stored for {qa_id} - MongoDB connection unavailable")
        except Exception as e:
            logger.error(f"Error storing Q&A embedding: {e}")
            logger.error(traceback.format_exc())
    
    @log_retrieval_operation("qa_matching")
    def find_best_qa_match(self, question: str, answer: str) -> Optional[Dict[str, Any]]:
        """Find the best matching ideal Q&A pair for a given question and answer."""
        
        try:
            collection = get_qa_embeddings_collection()
            if collection is None:
                logger.warning("MongoDB connection unavailable - cannot find Q&A matches")
                return None
            
            # Generate embeddings for the input
            question_embedding = self.generate_embedding(question)
            answer_embedding = self.generate_embedding(answer)
            
            # Get all ideal Q&A pairs from MongoDB
            ideal_qa_pairs = list(collection.find({"is_ideal": True}))
            
            if not ideal_qa_pairs:
                logger.warning("No ideal Q&A pairs found in database")
                return None
            
            best_match = None
            best_score = 0
            
            for ideal_qa in ideal_qa_pairs:
                try:
                    # Calculate question similarity
                    if "question_embedding" in ideal_qa:
                        ideal_question_emb = np.array(ideal_qa["question_embedding"])
                        question_similarity = self._compute_similarity(question_embedding, ideal_question_emb)
                    else:
                        question_similarity = 0
                    
                    # Calculate answer similarity
                    ideal_answer_emb = np.array(ideal_qa["embedding"])
                    answer_similarity = self._compute_similarity(answer_embedding, ideal_answer_emb)
                    
                    # Combined score (weighted more towards answer similarity)
                    combined_score = (question_similarity * 0.3) + (answer_similarity * 0.7)
                    
                    # Check if this is a better match
                    if (question_similarity >= self.question_similarity_threshold and 
                        combined_score > best_score):
                        best_match = {
                            "qa_id": ideal_qa["qa_id"],
                            "question": ideal_qa["question"],
                            "answer": ideal_qa["answer"],
                            "question_similarity": question_similarity,
                            "answer_similarity": answer_similarity,
                            "combined_score": combined_score
                        }
                        best_score = combined_score
                        
                except Exception as e:
                    logger.error(f"Error comparing with ideal Q&A {ideal_qa.get('qa_id', 'unknown')}: {e}")
                    continue
            
            if best_match:
                logger.debug(f"Found best match: {best_match['qa_id']} (score: {best_score:.3f})")
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding best Q&A match: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def save_qa_pairs_to_json(self, qa_pairs: Dict[str, Dict[str, Any]], file_path: str):
        """Save Q&A pairs to a JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_qa = {}
            for qa_id, qa_data in qa_pairs.items():
                serializable_qa[qa_id] = {
                    "question": qa_data["question"],
                    "answer": qa_data["answer"],
                    "embedding": qa_data["embedding"].tolist() if isinstance(qa_data["embedding"], np.ndarray) else qa_data["embedding"],
                    "question_embedding": qa_data["question_embedding"].tolist() if isinstance(qa_data.get("question_embedding"), np.ndarray) else qa_data.get("question_embedding")
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_qa, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved Q&A pairs to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving Q&A pairs to JSON: {e}")
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            from scipy.spatial.distance import cosine
            similarity = float(1 - cosine(emb1, emb2))
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
