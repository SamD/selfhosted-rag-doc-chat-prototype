#!/usr/bin/env python3
"""
Dependencies for FastAPI services.
"""
from functools import lru_cache
from services.rag_service import RagService


class DependenciesService:
    """Dependencies for FastAPI services as static methods."""
    
    @staticmethod
    @lru_cache()
    def get_rag_service() -> RagService:
        """Get a cached instance of the RAG service."""
        return RagService()

# Expose static methods as module-level functions after class definition
get_rag_service = DependenciesService.get_rag_service 