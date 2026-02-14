#!/usr/bin/env python3
"""
Database service for vector database operations (ChromaDB or Qdrant).
"""
from typing import Any, Dict, List, Optional

import chromadb
from config.settings import EMBEDDING_MODEL_PATH, LLAMA_USE_GPU, QDRANT_DENSE_WEIGHT, QDRANT_RETRIEVER_K, QDRANT_SPARSE_WEIGHT, USE_QDRANT, VECTOR_DB_COLLECTION, VECTOR_DB_HOST, VECTOR_DB_PORT
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, SparseVector
from utils.logging_config import setup_logging

log = setup_logging("database_service.log")

device = "cuda" if LLAMA_USE_GPU else "cpu"
log.info(f"Device: {device}")
log.info(f"Vector DB Type: {'Qdrant' if USE_QDRANT else 'ChromaDB'}")

# Cache embeddings models per process to avoid loading multiple times on GPU
_embeddings_cache = None
_sparse_embeddings_cache = None

WEIGHTS = {"dense": QDRANT_DENSE_WEIGHT, "sparse": QDRANT_SPARSE_WEIGHT}

# Build prefetches with higher limits to allow sparse to "check" dense bias
def build_prefetches(dense_vector: List[float], sparse_vector: Any) -> List[Prefetch]:
    return [
        Prefetch(
            query=dense_vector,
            using="dense",
            limit=50  # Increased to provide more candidates for fusion
        ),
        # Higher limit for sparse ensures keywords are caught even if semantically "distant"
        Prefetch(
            query=SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist()
            ),
            using="sparse",
            limit=100
        ),
    ]

# Use DBSF (Distribution-Based Score Fusion) to eliminate E5/BM25 scale bias
f_query = models.FusionQuery(fusion=models.Fusion.DBSF)

class QdrantHybridRetriever(BaseRetriever):
    """Hybrid dense + sparse retriever for Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any
    collection_name: str
    embeddings: Any
    sparse_embeddings: Any

    def __init__(self, client, collection_name, embeddings, sparse_embeddings, **kwargs):
        super().__init__(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            **kwargs
        )


    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Required method for BaseRetriever."""

        dense = self.embeddings.embed_query(query)
        sparse = list(self.sparse_embeddings.embed([query]))[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=build_prefetches(dense, sparse),
            query=f_query,
            limit=QDRANT_RETRIEVER_K,
            with_payload=True
        )
        return [
            Document(
                page_content=hit.payload.get("page_content", ""),
                metadata=hit.payload.get("metadata", {})
            )
            for hit in results.points
        ]

class VectorStoreWrapper:
    """Wrapper around vector store to provide unified interface for ChromaDB and Qdrant."""

    def __init__(self, vectorstore: Any, db_type: str) -> None:
        self.vectorstore = vectorstore
        self.db_type = db_type

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> Any:
        """Add texts to the vector store."""
        if self.db_type == "qdrant" and ids is not None:
            import uuid
            namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
            ids = [uuid.uuid5(namespace, str(id_)) for id_ in ids]
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever interface."""
        if isinstance(self.vectorstore, BaseRetriever):
            return self.vectorstore
        return self.vectorstore.as_retriever(**kwargs)

    def delete(self, where: Optional[Dict[str, Any]] = None, ids: Optional[List[str]] = None) -> None:
        """Delete documents by metadata filter or IDs."""
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

    def get_collection_count(self) -> int:
        """Get the total count of documents in the collection."""
        if self.db_type == "qdrant":
            result = self.vectorstore.client.count(collection_name=VECTOR_DB_COLLECTION)
            return result.count
        else:
            return self.vectorstore._collection.count()

    def upsert(self, collection_name: str, points: List[Dict[str, Any]]) -> Any:
        """Upsert points to Qdrant collection (Qdrant only)."""
        if self.db_type != "qdrant":
            raise NotImplementedError("upsert is only supported for Qdrant")
        import uuid
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        for point in points:
            if isinstance(point.get("id"), str):
                point["id"] = uuid.uuid5(namespace, point["id"])
        return self.vectorstore.client.upsert(collection_name=collection_name, points=points)


class DatabaseServiceQdrantSparseTesting:
    """Database service for vector database operations as static methods."""

    @staticmethod
    def get_embeddings() -> Any:
        global _embeddings_cache
        if _embeddings_cache is None:
            log.info(f"Loading embeddings model on {device}...")
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                model_kwargs={"device": device, "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True}
            )
            log.info(f"Embeddings model loaded successfully on {device}")
        return _embeddings_cache

    @staticmethod
    def get_sparse_embeddings() -> Any:
        global _sparse_embeddings_cache
        if _sparse_embeddings_cache is None:
            from fastembed import SparseTextEmbedding
            log.info("Loading sparse embeddings model...")
            _sparse_embeddings_cache = SparseTextEmbedding("Qdrant/bm25")
            log.info("Sparse embeddings model loaded successfully")
        return _sparse_embeddings_cache


    @staticmethod
    def get_chromadb() -> "VectorStoreWrapper":
        embeddings = DatabaseServiceQdrantSparseTesting.get_embeddings()
        client = chromadb.HttpClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
        vectorstore = Chroma(client=client, collection_name=VECTOR_DB_COLLECTION, embedding_function=embeddings)
        return VectorStoreWrapper(vectorstore, "chroma")

    @staticmethod
    def get_qdrant() -> "VectorStoreWrapper":
        embeddings = DatabaseServiceQdrantSparseTesting.get_embeddings()
        sparse_embeddings = DatabaseServiceQdrantSparseTesting.get_sparse_embeddings()
        client = QdrantClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)

        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            client.get_collection(VECTOR_DB_COLLECTION)
            log.info(f"Collection '{VECTOR_DB_COLLECTION}' already exists")
        except UnexpectedResponse as e:
            if "doesn't exist" in str(e) or e.status_code == 404:
                from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

                log.info(f"Creating collection '{VECTOR_DB_COLLECTION}'")
                test_embedding = embeddings.embed_query("test")
                embedding_dimension = len(test_embedding)
                try:
                    client.create_collection(
                        collection_name=VECTOR_DB_COLLECTION,
                        vectors_config={
                            "dense": VectorParams(
                                on_disk=True,
                                size=embedding_dimension,
                                distance=Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(
                                index=SparseIndexParams(
                                    on_disk=True,  # Optimization: RAM-based index prevents hanging
                                    full_scan_threshold=1000
                                )
                            )
                        }
                    )
                    log.info(f"Collection '{VECTOR_DB_COLLECTION}' created successfully (dimension: {embedding_dimension})")
                except UnexpectedResponse as create_error:
                    if create_error.status_code == 409 or "already exists" in str(create_error):
                        log.info(f"Collection '{VECTOR_DB_COLLECTION}' already exists (created by another process)")
                    else:
                        raise
            else:
                raise

        retriever = QdrantHybridRetriever(
            client=client,
            collection_name=VECTOR_DB_COLLECTION,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        return VectorStoreWrapper(retriever, "qdrant")

    @staticmethod
    def get_db() -> "VectorStoreWrapper":
        if USE_QDRANT:
            log.info(f"Connecting to Qdrant at {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
            return DatabaseServiceQdrantSparseTesting.get_qdrant()
        else:
            log.info(f"Connecting to ChromaDB at {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
            return DatabaseServiceQdrantSparseTesting.get_chromadb()


get_db = DatabaseServiceQdrantSparseTesting.get_db