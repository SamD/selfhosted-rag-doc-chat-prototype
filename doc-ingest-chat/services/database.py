#!/usr/bin/env python3
"""
Database service for ChromaDB operations.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import CHROMA_COLLECTION, E5_MODEL_PATH, CHROMA_HOST, CHROMA_PORT, LLAMA_USE_GPU

device = "cuda" if LLAMA_USE_GPU else "cpu"
print(f"Device: {device}")


class DatabaseService:
    """Database service for ChromaDB operations as static methods."""
    
    @staticmethod
    def get_db():
        """Get ChromaDB instance with embeddings."""
        embeddings = HuggingFaceEmbeddings(
            model_name=E5_MODEL_PATH,
            model_kwargs={
                "device": device,
                "trust_remote_code": True
            },
            encode_kwargs={"normalize_embeddings": True}
        )

        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT
        )

        return Chroma(
            client=client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings
        )

# Expose static methods as module-level functions after class definition
get_db = DatabaseService.get_db 