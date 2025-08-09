"""
API endpoints for chat, health, and status using FastAPI.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from models.query import QueryRequest, QueryResponse
from pydantic import BaseModel
from services.dependencies import get_rag_service
from services.rag_service import RagService
from utils.logging_config import setup_logging

log = setup_logging("api_endpoints.log", logging.DEBUG)

router = APIRouter(prefix="/api/v1", tags=["chat"])


class HealthResponse(BaseModel):
    status: str
    message: str


class StatusResponse(BaseModel):
    status: str
    collection_count: int
    model_info: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="API is running")


@router.get("/status", response_model=StatusResponse)
def get_status(service: RagService = Depends(get_rag_service)):
    """Get system status and collection information."""
    try:
        # Get collection count from the service
        collection_info = service.get_collection_info()
        return StatusResponse(
            status="operational",
            collection_count=collection_info.get("count", 0),
            model_info=collection_info.get("model_info", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/query", response_model=QueryResponse)
def query_handler(
        req: QueryRequest,
        service: RagService = Depends(get_rag_service)
):
    """Process a query and return a response with citations."""
    try:
        log.info(f"Received query: {req.query}")
        log.info(f"Chat history: {req.chat_history}")
        response = service.answer_query(req.query, req.chat_history)
        log.info(f"Response: {response}")
        return response
    except Exception as e:
        log.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
