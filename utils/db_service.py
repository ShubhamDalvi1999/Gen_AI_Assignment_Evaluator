import numpy as np
from pymongo import MongoClient
import logging
from enum import Enum
import json
import os
import sys
import traceback
import logging
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import traceback
from utils.embedding_service import compute_similarity


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
