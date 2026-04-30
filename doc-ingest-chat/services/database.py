import logging
import os
import random
import time
from typing import Any, List, Optional, Tuple

import chromadb
import duckdb
from config import settings
from config.settings import (
    EMBEDDING_MODEL_PATH,
    LLAMA_USE_GPU,
    USE_QDRANT,
    VECTOR_DB_COLLECTION,
    VECTOR_DB_HOST,
    VECTOR_DB_PORT,
)
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

log = logging.getLogger("ingest.database")

device = "cuda" if LLAMA_USE_GPU else "cpu"

# Per-process singletons
_EMBEDDINGS_CACHE: Optional[HuggingFaceEmbeddings] = None
_QDRANT_CLIENT_CACHE: Optional[QdrantClient] = None
_CHROMA_CLIENT_CACHE: Optional[chromadb.HttpClient] = None


class DatabaseService:
    """
    Centralized database service for DuckDB and Vector Store management.
    Handles file-level locking for DuckDB and singleton caching for Vector models.
    """

    # --- DUCKDB LOGIC (Relational / Lifecycle) ---

    @staticmethod
    def _run_statement(sql: str, params: tuple = (), fetch: bool = False, fetch_all: bool = False, db_path: str = None) -> Any:
        """
        Internal driver that handles the 'DuckDB Lock' retry loop.
        """
        target_db = db_path or settings.DUCKDB_FILE
        max_retries = 20
        base_delay = 0.2

        for attempt in range(max_retries):
            con = None
            try:
                con = duckdb.connect(target_db)
                if fetch:
                    res = con.execute(sql, params).fetchone()
                    cols = [desc[0] for desc in con.description] if con.description else []
                    return res, cols
                elif fetch_all:
                    res = con.execute(sql, params).fetchall()
                    cols = [desc[0] for desc in con.description] if con.description else []
                    return res, cols
                else:
                    con.execute(sql, params)
                    return True
            except (duckdb.IOException, duckdb.InternalException) as e:
                err_msg = str(e).lower()
                if "lock" in err_msg or "used by another process" in err_msg:
                    delay = base_delay * (2**attempt) + (random.random() * 0.2)
                    log.warning(f"⏳ DuckDB file locked, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()

        raise RuntimeError(f"💥 Failed to acquire DuckDB lock after {max_retries} attempts.")

    @staticmethod
    def execute(sql: str, params: tuple = ()) -> bool:
        """Executes a DDL or DML statement (CREATE, INSERT, UPDATE, DELETE)."""
        return DatabaseService._run_statement(sql, params)

    @staticmethod
    def query(sql: str, params: tuple = (), fetch_all: bool = True) -> Tuple[Any, List[str]]:
        """Executes a read-only query (SELECT) and returns (results, columns)."""
        if fetch_all:
            return DatabaseService._run_statement(sql, params, fetch_all=True)
        return DatabaseService._run_statement(sql, params, fetch=True)

    @staticmethod
    def get_schema_path() -> str:
        """Robustly locates the schema.sql file."""
        search_paths = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql", "schema.sql"), os.path.join(os.getcwd(), "doc-ingest-chat", "sql", "schema.sql"), "doc-ingest-chat/sql/schema.sql", "sql/schema.sql"]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return search_paths[0]

    @staticmethod
    def init_db(db_path: Optional[str] = None) -> None:
        """One-time startup task: Synchronizes the DuckDB schema."""
        target_db = db_path or settings.DUCKDB_FILE
        schema_path = DatabaseService.get_schema_path()

        if not os.path.exists(schema_path):
            log.error(f"❌ Schema file not found: {schema_path}")
            return

        try:
            log.info(f"💾 Initializing DuckDB schema at {target_db}")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()
            DatabaseService._run_statement(schema_sql, db_path=target_db)
            log.info("✅ DuckDB schema initialized.")
        except Exception as e:
            log.error(f"💥 Failed to initialize DuckDB schema: {e}")

    @staticmethod
    def get_duckdb():
        """Returns a standard DuckDB connection (No retries)."""
        return duckdb.connect(settings.DUCKDB_FILE)

    # --- VECTOR STORE LOGIC (Semantic Search) ---

    @staticmethod
    def get_embeddings() -> HuggingFaceEmbeddings:
        """Get or initialize the singleton embedding model (per process)."""
        global _EMBEDDINGS_CACHE
        if _EMBEDDINGS_CACHE is None:
            from utils.exceptions import ConfigurationError

            if not os.path.exists(EMBEDDING_MODEL_PATH):
                raise ConfigurationError(f"EMBEDDING_MODEL_PATH not found at {EMBEDDING_MODEL_PATH}")

            log.info(f"🚀 Loading embedding model into {device}: {EMBEDDING_MODEL_PATH}")
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={"device": device, "trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True})
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
    def get_vectorstore():
        """Get vector database instance based on configuration."""
        if USE_QDRANT:
            return DatabaseService.get_qdrant()
        else:
            return DatabaseService.get_chromadb()


class VectorStoreWrapper(VectorStore):
    """
    A wrapper that allows switching between ChromaDB and Qdrant at runtime.
    """

    def __init__(self, vectorstore: VectorStore, db_type: str = "qdrant"):
        self.vectorstore = vectorstore
        self.db_type = db_type

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> "VectorStoreWrapper":
        raise NotImplementedError("Use the specific vectorstore.from_texts instead.")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        # Qdrant requires UUIDs or integers, not strings. Convert string IDs to UUIDs.
        if self.db_type == "qdrant" and kwargs.get("ids") is not None:
            import uuid

            namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
            kwargs["ids"] = [uuid.uuid5(namespace, str(id_)) for id_ in kwargs["ids"]]

        return self.vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any):
        return self.vectorstore.similarity_search(query, k, **kwargs)

    def as_retriever(self, **kwargs: Any):
        return self.vectorstore.as_retriever(**kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if self.db_type == "qdrant":
            if "where" in kwargs:
                where = kwargs.pop("where")
                if "source_file" in where:
                    from qdrant_client.models import FieldCondition, Filter, MatchValue

                    filter_condition = Filter(must=[FieldCondition(key="metadata.source_file", match=MatchValue(value=where["source_file"]))])
                    self.vectorstore.client.delete(collection_name=VECTOR_DB_COLLECTION, points_selector=filter_condition)
            elif ids:
                self.vectorstore.delete(ids=ids)
        else:
            return self.vectorstore.delete(ids, **kwargs)

    def get_collection_count(self) -> int:
        """Helper to get point/document count from the underlying store."""
        try:
            if self.db_type == "chroma" or hasattr(self.vectorstore, "_collection"):
                return self.vectorstore._collection.count()
            elif self.db_type == "qdrant" or (hasattr(self.vectorstore, "client") and hasattr(self.vectorstore, "collection_name")):
                res = self.vectorstore.client.get_collection(self.vectorstore.collection_name)
                return res.points_count
        except Exception:
            pass
        return 0


# Convenience aliases
get_duckdb = DatabaseService.get_duckdb
get_vectorstore = DatabaseService.get_vectorstore
init_db = DatabaseService.init_db
execute = DatabaseService.execute
query = DatabaseService.query
