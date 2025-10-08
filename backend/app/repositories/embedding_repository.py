from typing import List, Dict, Any, Optional
import numpy as np
import traceback
from datetime import datetime

from ..repositories.db import (
    get_embeddings_collection,
    get_qa_embeddings_collection,
    get_student_submissions_collection,
    get_counters_collection
)
from ..core.logging import db_logger as logger
from ..core.rag_logging import log_retrieval_operation


class EmbeddingRepository:
    """Repository for embedding-related database operations."""
    
    def store_embedding(self, function_name: str, code: str, embedding: np.ndarray) -> None:
        """Store embedding in MongoDB."""
        try:
            collection = get_embeddings_collection()
            if collection is not None:
                logger.info("========== EMBEDDING STORAGE STAGE ==========")
                logger.info(f"Storing embedding for function: {function_name} (vector dimension: {len(embedding)})")
                
                logger.debug(f"Storing embedding for function: {function_name}")
                document = {
                    "function_name": function_name,
                    "code": code,
                    "embedding": embedding.tolist(),
                    "timestamp": datetime.now()
                }
                result = collection.insert_one(document)
                logger.info(f"Successfully stored embedding for function: {function_name} (ID: {result.inserted_id})")
                logger.debug(f"Embedding stored for function: {function_name} (ID: {result.inserted_id})")
            else:
                logger.warning(f"Embedding not stored for function {function_name} - MongoDB connection unavailable")
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            logger.error(traceback.format_exc())
    
    @log_retrieval_operation("context_retrieval")
    def retrieve_similar_contexts(self, query_embedding: np.ndarray, top_k: int = 9) -> List[Dict[str, Any]]:
        """Retrieve similar contexts from MongoDB."""
        logger.info(f"Retrieving similar contexts (top_{top_k})")
        start_time = datetime.now()
        
        try:
            collection = get_embeddings_collection()
            if collection is not None:
                # Get all stored embeddings from MongoDB
                logger.debug("Querying MongoDB for stored embeddings")
                stored_docs = list(collection.find({}, {"embedding": 1, "code": 1, "function_name": 1}))
                logger.debug(f"Retrieved {len(stored_docs)} embeddings from MongoDB for similarity comparison")
                
                if not stored_docs:
                    logger.warning("No stored embeddings found in MongoDB")
                    return []
                
                # Calculate similarities
                logger.debug("Calculating similarity scores between query and stored embeddings")
                calc_start = datetime.now()
                similarities = []
                for doc in stored_docs:
                    similarity = self._compute_similarity(query_embedding, np.array(doc["embedding"]))
                    doc["similarity"] = similarity
                    similarities.append((similarity, doc))
                
                calc_time = (datetime.now() - calc_start).total_seconds()
                logger.debug(f"Calculated {len(similarities)} similarity scores in {calc_time:.2f}s")
                    
                # Sort by similarity and get top k
                similarities.sort(reverse=True, key=lambda x: x[0])
                result = [doc for _, doc in similarities[:top_k]]
                
                # Log result summary
                if result:
                    top_similarity = similarities[0][0] if similarities else 0
                    logger.info(f"Retrieved {len(result)} similar contexts (top similarity: {top_similarity:.4f})")
                else:
                    logger.info("No similar contexts found")
                
                total_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Similarity retrieval completed in {total_time:.2f}s")
                
                return result
            else:
                logger.warning("MongoDB connection unavailable - cannot retrieve similar contexts")
                return []
        except Exception as e:
            logger.error(f"Error retrieving similar contexts: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def clear_embeddings(self) -> None:
        """Clear all embeddings from the database."""
        try:
            collection = get_embeddings_collection()
            if collection is not None:
                delete_result = collection.delete_many({})
                logger.info(f"Cleared {delete_result.deleted_count} embeddings from database")
            else:
                logger.warning("MongoDB connection unavailable - cannot clear embeddings")
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
            logger.error(traceback.format_exc())
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            from scipy.spatial.distance import cosine
            similarity = float(1 - cosine(emb1, emb2))
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
