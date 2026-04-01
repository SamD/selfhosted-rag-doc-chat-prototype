#!/usr/bin/env python3
"""
Database service for vector database operations (ChromaDB or Qdrant).
Includes singleton caching for models and clients (per-process).
"""

import logging
import uuid
from typing import Optional

import chromadb
from config.settings import (
    EMBEDDING_MODEL_PATH,
    LLAMA_USE_GPU,
    USE_QDRANT,
    VECTOR_DB_COLLECTION,
    VECTOR_DB_HOST,
    VECTOR_DB_PORT,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

log = logging.getLogger("ingest.database")

device = "cuda" if LLAMA_USE_GPU else "cpu"

# Per-process singletons
_EMBEDDINGS_CACHE: Optional[HuggingFaceEmbeddings] = None
_QDRANT_CLIENT_CACHE: Optional[QdrantClient] = None
_CHROMA_CLIENT_CACHE: Optional[chromadb.HttpClient] = None

class VectorStoreWrapper:
    """Wrapper around vector store to provide unified interface for ChromaDB and Qdrant."""

    def __init__(self, vectorstore, db_type):
        self.vectorstore = vectorstore
        self.db_type = db_type

    def add_texts(self, texts, metadatas=None, ids=None):
        """Add texts to the vector store."""
        if self.db_type == "qdrant" and ids is not None:
            namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
            ids = [uuid.uuid5(namespace, str(id_)) for id_ in ids]
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)

    def as_retriever(self, **kwargs):
        return self.vectorstore.as_retriever(**kwargs)

    def delete(self, where=None, ids=None):
        if self.db_type == "qdrant":
            if where:
                if "source_file" in where:
                    from qdrant_client.models import FieldCondition, Filter, MatchValue
                    filter_condition = Filter(must=[FieldCondition(key="metadata.source_file", match=MatchValue(value=where["source_file"]))])
                    self.vectorstore.client.delete(collection_name=VECTOR_DB_COLLECTION, points_selector=filter_condition)
            elif ids:
                self.vectorstore.delete(ids=ids)
        else:
            if where:
                self.vectorstore.delete(where=where)
            elif ids:
                self.vectorstore.delete(ids=ids)

    def get_collection_count(self):
        if self.db_type == "qdrant":
            result = self.vectorstore.client.count(collection_name=VECTOR_DB_COLLECTION)
            return result.count
        else:
            return self.vectorstore._collection.count()


class DatabaseService:
    """Database service for vector database operations as static methods."""

    @staticmethod
    def get_embeddings() -> HuggingFaceEmbeddings:
        """Get or initialize the singleton embedding model (per process)."""
        global _EMBEDDINGS_CACHE
        if _EMBEDDINGS_CACHE is None:
            log.info(f"🚀 Loading embedding model into {device}: {EMBEDDING_MODEL_PATH}")
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH, 
                model_kwargs={"device": device, "trust_remote_code": True}, 
                encode_kwargs={"normalize_embeddings": True}
            )
        return _EMBEDDINGS_CACHE

    @staticmethod
    def get_qdrant_client() -> QdrantClient:
        global _QDRANT_CLIENT_CACHE
        if _QDRANT_CLIENT_CACHE is None:
            log.info(f"🛰️ Initializing Qdrant client: {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
            _QDRANT_CLIENT_CACHE = QdrantClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
        return _QDRANT_CLIENT_CACHE

    @staticmethod
    def get_chroma_client() -> chromadb.HttpClient:
        global _CHROMA_CLIENT_CACHE
        if _CHROMA_CLIENT_CACHE is None:
            log.info(f"📡 Initializing Chroma client: {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
            _CHROMA_CLIENT_CACHE = chromadb.HttpClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
        return _CHROMA_CLIENT_CACHE

    @staticmethod
    def get_chromadb():
        embeddings = DatabaseService.get_embeddings()
        client = DatabaseService.get_chroma_client()
        vectorstore = Chroma(client=client, collection_name=VECTOR_DB_COLLECTION, embedding_function=embeddings)
        return VectorStoreWrapper(vectorstore, "chroma")

    @staticmethod
    def get_qdrant():
        embeddings = DatabaseService.get_embeddings()
        client = DatabaseService.get_qdrant_client()

        from qdrant_client.http.exceptions import UnexpectedResponse
        try:
            client.get_collection(VECTOR_DB_COLLECTION)
        except UnexpectedResponse as e:
            if "doesn't exist" in str(e) or e.status_code == 404:
                from qdrant_client.models import Distance, VectorParams
                log.info(f"Creating collection '{VECTOR_DB_COLLECTION}'")
                test_embedding = embeddings.embed_query("test")
                dim = len(test_embedding)
                try:
                    client.create_collection(
                        collection_name=VECTOR_DB_COLLECTION,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                except UnexpectedResponse as create_error:
                    if create_error.status_code != 409 and "already exists" not in str(create_error):
                        raise
            else:
                raise

        vectorstore = QdrantVectorStore(client=client, collection_name=VECTOR_DB_COLLECTION, embedding=embeddings)
        return VectorStoreWrapper(vectorstore, "qdrant")

    @staticmethod
    def get_db():
        if USE_QDRANT:
            return DatabaseService.get_qdrant()
        else:
            return DatabaseService.get_chromadb()


get_db = DatabaseService.get_db
