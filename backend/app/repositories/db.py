from pymongo import MongoClient
from typing import Optional
import logging
from ..core.config import settings
from ..core.logging import db_logger

# Global MongoDB client
mongo_client: Optional[MongoClient] = None
db = None
embeddings_collection = None
qa_embeddings_collection = None
student_submissions_collection = None
counters_collection = None


def get_mongodb_client() -> Optional[MongoClient]:
    """Get MongoDB client instance."""
    return mongo_client


def get_db():
    """Get database instance."""
    return db


def get_embeddings_collection():
    """Get embeddings collection."""
    return embeddings_collection


def get_qa_embeddings_collection():
    """Get Q&A embeddings collection."""
    return qa_embeddings_collection


def get_student_submissions_collection():
    """Get student submissions collection."""
    return student_submissions_collection


def get_counters_collection():
    """Get counters collection."""
    return counters_collection


def init_mongodb() -> bool:
    """Initialize MongoDB connection and collections."""
    global mongo_client, db, embeddings_collection, qa_embeddings_collection
    global student_submissions_collection, counters_collection
    
    try:
        # Check if MongoDB URI is available
        if not settings.mongodb_uri:
            db_logger.error("MONGODB_URI not found in configuration")
            return False
        
        # Connect to MongoDB
        db_logger.info("Attempting to connect to MongoDB")
        mongo_client = MongoClient(settings.mongodb_uri)
        
        # Test connection
        mongo_client.admin.command('ping')
        db_logger.info(f"Successfully connected to MongoDB database: {settings.mongodb_db_name}")
        
        # Initialize database and collections
        db = mongo_client[settings.mongodb_db_name]
        embeddings_collection = db[settings.mongodb_embeddings_collection]
        qa_embeddings_collection = db[settings.mongodb_qa_collection]
        student_submissions_collection = db["student_submissions"]
        counters_collection = db["counters"]
        
        # Create indexes for efficient querying
        _create_indexes()
        
        # Initialize counter for student submission IDs if it doesn't exist
        if counters_collection.count_documents({"_id": "student_submission_id"}) == 0:
            counters_collection.insert_one({"_id": "student_submission_id", "value": 0})
            db_logger.info("Initialized student submission ID counter")
        
        return True
        
    except Exception as e:
        db_logger.error(f"Failed to connect to MongoDB: {e}")
        mongo_client = None
        db = None
        embeddings_collection = None
        qa_embeddings_collection = None
        student_submissions_collection = None
        counters_collection = None
        return False


def _create_indexes():
    """Create database indexes for performance."""
    try:
        # Create indexes for student_submissions collection
        existing_indexes = [idx['name'] for idx in student_submissions_collection.list_indexes()]
        
        if "submission_id_1" not in existing_indexes:
            db_logger.info("Creating submission_id index on student_submissions collection")
            student_submissions_collection.create_index([("submission_id", 1)])
        
        if "submission_id_1_qa_id_1" not in existing_indexes:
            db_logger.info("Creating compound index on student_submissions collection")
            student_submissions_collection.create_index([("submission_id", 1), ("qa_id", 1)], unique=True)
        
        if "timestamp_-1" not in existing_indexes:
            db_logger.info("Creating timestamp index on student_submissions collection")
            student_submissions_collection.create_index([("timestamp", -1)])
            
    except Exception as e:
        db_logger.warning(f"Error creating indexes: {e}")


def close_mongodb():
    """Close MongoDB connection."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        db_logger.info("MongoDB connection closed")
