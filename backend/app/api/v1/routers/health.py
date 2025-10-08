from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ....repositories.db import get_mongodb_client

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check MongoDB connection
        mongo_client = get_mongodb_client()
        if mongo_client:
            # Test the connection
            mongo_client.admin.command('ping')
            db_status = "connected"
        else:
            db_status = "disconnected"
        
        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
