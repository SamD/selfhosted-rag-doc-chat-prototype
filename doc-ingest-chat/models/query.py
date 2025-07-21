#!/usr/bin/env python3
"""
Query models for the API endpoints.
"""
from typing import List, Dict
from pydantic import BaseModel


# class QueryRequest(BaseModel):
#     """Request model for querying the RAG system."""
#     query: str
#     chat_history: Optional[List[str]] = None
#
#
# class QueryResponse(BaseModel):
#     """Response model for query results."""
#     answer: str
#     sources: Optional[List[str]] = None
#     confidence: Optional[float] = None


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str
    chat_history: List[Dict]

class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str
    chat_history: List[Dict]
    debug: str
