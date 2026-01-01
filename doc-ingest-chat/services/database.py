#!/usr/bin/env python3
"""
Database service for vector database operations (ChromaDB or Qdrant).
"""

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
from utils.logging_config import setup_logging

log = setup_logging("database_service.log")

device = "cuda" if LLAMA_USE_GPU else "cpu"
log.info(f"Device: {device}")
log.info(f"Vector DB Type: {'Qdrant' if USE_QDRANT else 'ChromaDB'}")


class VectorStoreWrapper:
    """Wrapper around vector store to provide unified interface for ChromaDB and Qdrant."""

    def __init__(self, vectorstore, db_type):
        self.vectorstore = vectorstore
        self.db_type = db_type

    def add_texts(self, texts, metadatas=None, ids=None):
        """Add texts to the vector store."""
        # Qdrant requires UUIDs or integers, not strings. Convert string IDs to UUIDs.
        if self.db_type == "qdrant" and ids is not None:
            import uuid

            # Use UUID5 (SHA-1 based) for deterministic UUID generation from string IDs
            namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace
            ids = [uuid.uuid5(namespace, str(id_)) for id_ in ids]
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)

    def as_retriever(self, **kwargs):
        """Return a retriever interface."""
        return self.vectorstore.as_retriever(**kwargs)

    def delete(self, where=None, ids=None):
        """Delete documents by metadata filter or IDs."""
        if self.db_type == "qdrant":
            # Qdrant uses delete() with filter or IDs
            if where:
                # Convert ChromaDB-style where clause to Qdrant filter
                # For now, support source_file filter
                if "source_file" in where:
                    from qdrant_client.models import FieldCondition, Filter, MatchValue

                    filter_condition = Filter(must=[FieldCondition(key="metadata.source_file", match=MatchValue(value=where["source_file"]))])
                    self.vectorstore.client.delete(collection_name=VECTOR_DB_COLLECTION, points_selector=filter_condition)
            elif ids:
                self.vectorstore.delete(ids=ids)
        else:
            # ChromaDB delete
            if where:
                self.vectorstore.delete(where=where)
            elif ids:
                self.vectorstore.delete(ids=ids)

    def get_collection_count(self):
        """Get the total count of documents in the collection."""
        if self.db_type == "qdrant":
            # Qdrant uses client.count()
            result = self.vectorstore.client.count(collection_name=VECTOR_DB_COLLECTION)
            return result.count
        else:
            # ChromaDB uses _collection.count()
            return self.vectorstore._collection.count()


class DatabaseService:
    """Database service for vector database operations as static methods."""

    @staticmethod
    def _get_embeddings():
        """Get HuggingFace embeddings model."""
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={"device": device, "trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True})

    @staticmethod
    def get_chromadb():
        """Get ChromaDB instance with embeddings."""
        embeddings = DatabaseService._get_embeddings()

        client = chromadb.HttpClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)

        vectorstore = Chroma(client=client, collection_name=VECTOR_DB_COLLECTION, embedding_function=embeddings)

        return VectorStoreWrapper(vectorstore, "chroma")

    @staticmethod
    def get_qdrant():
        """Get Qdrant instance with embeddings."""
        embeddings = DatabaseService._get_embeddings()

        client = QdrantClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)

        # Create collection if it doesn't exist (required for langchain-qdrant 0.2.1)
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            client.get_collection(VECTOR_DB_COLLECTION)
            log.info(f"Collection '{VECTOR_DB_COLLECTION}' already exists")
        except UnexpectedResponse as e:
            # Collection doesn't exist (404), create it
            if "doesn't exist" in str(e) or e.status_code == 404:
                from qdrant_client.models import Distance, VectorParams

                log.info(f"Creating collection '{VECTOR_DB_COLLECTION}'")
                # Get embedding dimension by encoding a test string
                test_embedding = embeddings.embed_query("test")
                embedding_dimension = len(test_embedding)
                try:
                    client.create_collection(
                        collection_name=VECTOR_DB_COLLECTION,
                        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
                    )
                    log.info(f"Collection '{VECTOR_DB_COLLECTION}' created successfully (dimension: {embedding_dimension})")
                except UnexpectedResponse as create_error:
                    # Handle race condition: if collection was created by another process (409 Conflict), that's fine
                    if create_error.status_code == 409 or "already exists" in str(create_error):
                        log.info(f"Collection '{VECTOR_DB_COLLECTION}' already exists (created by another process)")
                    else:
                        raise
            else:
                raise

        vectorstore = QdrantVectorStore(client=client, collection_name=VECTOR_DB_COLLECTION, embedding=embeddings)

        return VectorStoreWrapper(vectorstore, "qdrant")

    @staticmethod
    def get_db():
        """Get vector database instance based on configuration."""
        if USE_QDRANT:
            log.info(f"Connecting to Qdrant at {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
            return DatabaseService.get_qdrant()
        else:
            log.info(f"Connecting to ChromaDB at {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
            return DatabaseService.get_chromadb()


# Expose static methods as module-level functions after class definition
get_db = DatabaseService.get_db
