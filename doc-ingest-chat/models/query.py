#!/usr/bin/env python3
"""
Query models for the API endpoints.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""

    query: str
    session_id: str = ""


class QueryResponse(BaseModel):
    """Response model for query results."""

    answer: str
    session_id: str
    debug: str
